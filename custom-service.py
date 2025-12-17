import json
import time
import os
from typing import Annotated, List

# 引入必要的库
import openai
import PIL
from PIL import Image
from pydantic import BaseModel
from openai import OpenAI

# 引入 Marker 的基类和工具
from marker.services.openai import OpenAIService
from marker.schema.blocks import Block
from marker.logger import get_logger

logger = get_logger()

class PipelineQwenService(OpenAIService):
    """
    自定义服务类：多模态流水线 (Pipeline)
    
    逻辑：
    1. 拦截 Marker 传入的图片。
    2. 使用【本地 Vision 模型】(如 Qwen-VL) 识别图片内容，转化为文字。
    3. 将识别到的文字拼接到 Prompt 中。
    4. 将纯文本请求发送给【主 LLM】(如 Qwen3-32B)。
    """

    
    def __init__(self, config=None):
        # 初始化父类，确保 Qwen3 的配置被正确加载
        super().__init__(config)

        # =========================================================
        # 配置区域：本地视觉模型 (Qwen-VL via LM Studio) 的配置
        # =========================================================
        # LM Studio 运行在本地 1234 端口
        vision_base_url: str = "http://10.8.2.63:12345"
        # vision_api_key: str = "lm-studio" 
        vision_model_name: str = "qwen3-vl-8b-instruct" # LM Studio 中加载的模型名称
        
        # 初始化一个专门用于连接视觉模型的客户端
        # 注意：这与连接主 LLM 的 self.get_client() 是分开的
        self.vision_client = OpenAI(
            base_url=self.vision_base_url,
            api_key=self.vision_api_key
        )

    def run_vision_model_locally(self, images: List[Image.Image]) -> str:
        """
       调用本地 Vision 模型处理图片
        """
        descriptions = []
        
        for idx, img in enumerate(images):
            try:
                # 1. 将 PIL 图片转换为 Base64 字符串
                # Marker 的父类中提供了 img_to_base64 工具函数
                base64_image = self.img_to_base64(img)
                
                # 2. 构造发送给 Qwen-VL (LM Studio) 的消息
                # 这是标准的 OpenAI 多模态格式
                vision_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "OCR this image and describe the layout structure accurately. Extract all text."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ]

                # 3. 调用本地视觉模型
                logger.info(f"正在调用本地视觉模型处理第 {idx+1} 张图片...")
                response = self.vision_client.chat.completions.create(
                    model=self.vision_model_name,
                    messages=vision_messages,
                    max_tokens=2048,  # 根据需要调整
                    temperature=0.1,  # OCR 任务建议低温度
                )
                
                # 4. 获取视觉模型的文本回复
                content = response.choices[0].message.content
                descriptions.append(f"[Image {idx+1} Content]:\n{content}")
                
            except Exception as e:
                logger.error(f"本地视觉模型处理失败: {e}")
                descriptions.append(f"[Image {idx+1} Error]: Could not process image.")

        # 将所有图片的描述合并成一个字符串返回
        return "\n\n".join(descriptions)

    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | List[PIL.Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        """
        重写父类的调用方法 (Entry Point)
        """
        # 继承重试次数和超时设置
        if max_retries is None:
            max_retries = self.max_retries
        if timeout is None:
            timeout = self.timeout

        # 获取连接【主 LLM (Qwen3)】的客户端
        # 这个 client 使用的是命令行传入的 --openai_base_url 和 --openai_api_key
        main_client = self.get_client()

        # =================================================
        # 第一阶段：视觉处理 (如果存在图片)
        # =================================================
        vision_context = ""
        if image:
            # 确保图片是列表格式
            if isinstance(image, Image.Image):
                images_list = [image]
            else:
                images_list = image
            
            # 调用我们定义的本地视觉处理函数
            # 这里会发生阻塞，直到本地 Qwen-VL 处理完图片
            vision_context = self.run_vision_model_locally(images_list)

        # =================================================
        # 第二阶段：Prompt 组装
        # =================================================
        final_prompt = prompt
        
        # 如果视觉模型提取到了内容，将其追加到原始 Prompt 后面
        # 这样 Qwen3 就能"看到"图片里的文字了
        if vision_context:
            final_prompt = (
                f"{prompt}\n\n"
                f"--- ADDITIONAL CONTEXT FROM IMAGES ---\n"
                f"The following text was extracted from the images associated with this block:\n"
                f"{vision_context}\n"
                f"--- END CONTEXT ---\n"
                f"Use the context above to fulfill the original request."
            )

        # =================================================
        # 第三阶段：构造纯文本请求
        # =================================================
        # 关键点：这里只发送 text，不发送 image_url，避免 400 错误
        messages = [
            {
                "role": "user",
                "content": final_prompt 
            }
        ]

        # =================================================
        # 第四阶段：发送给主 LLM (Qwen3) 并解析 JSON
        # =================================================
        total_tries = max_retries + 1
        for tries in range(1, total_tries + 1):
            try:
                # 调用 Qwen3-32B
                response = main_client.beta.chat.completions.parse(
                    extra_headers={
                        "X-Title": "Marker",
                    },
                    model=self.openai_model, # 使用命令行配置的模型名
                    messages=messages,       # 发送修改后的纯文本消息
                    timeout=timeout,
                    response_format=response_schema, # 强制结构化输出 (Pydantic)
                )
                
                # 获取回复文本
                response_text = response.choices[0].message.content
                
                # 记录 Token 使用量 (用于统计)
                total_tokens = response.usage.total_tokens
                if block:
                    block.update_metadata(
                        llm_tokens_used=total_tokens, llm_request_count=1
                    )
                
                # 解析 JSON 并返回
                return json.loads(response_text)
                
            except Exception as e:
                # 错误处理逻辑
                logger.warning(f"主 LLM ({self.openai_model}) 调用失败: {e}")
                
                if tries == total_tries:
                    logger.error("达到最大重试次数，放弃处理该 Block。")
                    break
                
                # 简单的退避策略
                time.sleep(tries * self.retry_wait_time)

        # 如果全失败了，返回空字典
        return {}
