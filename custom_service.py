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
from marker.services. openai import OpenAIService
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

    # =========================================================
    # 配置区域：按照 Marker 的方式定义为类属性
    # 这些属性可以通过命令行参数 --vision_base_url 等传入
    # =========================================================
    vision_base_url:  Annotated[
        str, "The base url for the vision model (e.g., LM Studio)."
    ] = "http://10.8.2.63:12345"
    
    vision_api_key: Annotated[
        str, "The API key for the vision model service."
    ] = "lm-studio"  # LM Studio 默认不需要真实 key
    
    vision_model_name: Annotated[
        str, "The model name for the vision model."
    ] = "qwen3-vl-8b-instruct"

    def get_vision_client(self) -> OpenAI:
        """获取视觉模型客户端"""
        return OpenAI(
            base_url=self.vision_base_url,
            api_key=self. vision_api_key
        )

    def run_vision_model_locally(self, images: List[Image.Image]) -> str:
        """
        调用本地 Vision 模型处理图片
        """
        descriptions = []
        vision_client = self.get_vision_client()
        
        for idx, img in enumerate(images):
            try:
                # 1. 将 PIL 图片转换为 Base64 字符串
                base64_image = self.img_to_base64(img)
                
                # 2. 构造发送给 Qwen-VL (LM Studio) 的消息
                vision_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text":  "OCR this image and describe the layout structure accurately.  Extract all text. "},
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
                response = vision_client.chat.completions.create(
                    model=self.vision_model_name,
                    messages=vision_messages,
                    max_tokens=2048,
                    temperature=0.1,
                )
                
                # 4. 获取视觉模型的文本回复
                content = response.choices[0].message.content
                descriptions.append(f"[Image {idx+1} Content]:\n{content}")
                
            except Exception as e:
                logger.error(f"本地视觉模型处理失败:  {e}")
                descriptions.append(f"[Image {idx+1} Error]:  Could not process image.")

        return "\n\n". join(descriptions)

    def __call__(
        self,
        prompt: str,
        image:  PIL.Image.Image | List[PIL.Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        """
        重写父类的调用方法 (Entry Point)
        """
        if max_retries is None: 
            max_retries = self. max_retries
        if timeout is None:
            timeout = self.timeout

        # 获取连接【主 LLM (Qwen3)】的客户端
        main_client = self.get_client()

        # =================================================
        # 第一阶段：视觉处理 (如果存在图片)
        # =================================================
        vision_context = ""
        if image: 
            if isinstance(image, Image.Image):
                images_list = [image]
            else:
                images_list = image
            
            vision_context = self.run_vision_model_locally(images_list)

        # =================================================
        # 第二阶段：Prompt 组装
        # =================================================
        final_prompt = prompt
        
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
                response = main_client.beta.chat.completions.parse(
                    extra_headers={
                        "X-Title": "Marker",
                    },
                    model=self.openai_model,
                    messages=messages,
                    timeout=timeout,
                    response_format=response_schema,
                )
                
                response_text = response.choices[0].message.content
                
                total_tokens = response.usage.total_tokens
                if block:
                    block.update_metadata(
                        llm_tokens_used=total_tokens, llm_request_count=1
                    )
                
                return json.loads(response_text)
                
            except Exception as e: 
                logger.warning(f"主 LLM ({self.openai_model}) 调用失败: {e}")
                
                if tries == total_tries: 
                    logger.error("达到最大重试次数，放弃处理该 Block。")
                    break
                
                time.sleep(tries * self.retry_wait_time)

        return {}
