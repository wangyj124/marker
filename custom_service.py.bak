import json
import time
from typing import List

import PIL
from PIL import Image
from pydantic import BaseModel
from openai import OpenAI

from marker.services.openai import OpenAIService
from marker.schema. blocks import Block
from marker.logger import get_logger

logger = get_logger()


class PipelineQwenService(OpenAIService):
    """
    自定义服务类：多模态流水线 (Pipeline)
    
    工作流程：
    1. 接收 Marker 传入的动态 prompt 和图片
    2. 用视觉模型处理图片，生成图片描述
    3. 将图片描述 + 原始动态 prompt 组合
    4. 发送给主 LLM 处理
    """

    # =========================================================
    # 视觉模型配置
    # =========================================================
    VISION_BASE_URL = "http://10.8.2.63:12345/v1/"
    VISION_MODEL_NAME = "qwen3-vl-8b-instruct"
    VISION_API_KEY = "lm-studio"

    # 视觉模型的 OCR 提示词（可自定义）
    VISION_OCR_PROMPT = """You are an OCR and document analysis expert. 
Analyze this image and: 
1. Extract ALL text content accurately
2. Describe the layout structure (tables, headers, lists, etc.)
3. Preserve the original formatting as much as possible

Output the extracted content in a structured format."""

    def get_vision_client(self) -> OpenAI:
        """获取视觉模型客户端"""
        return OpenAI(
            base_url=self.VISION_BASE_URL,
            api_key=self.VISION_API_KEY
        )

    def process_image_with_vision(self, img:  Image.Image) -> str:
        """
        使用视觉模型处理单张图片
        
        Args:
            img: PIL Image 对象
            
        Returns:
            图片的文字描述
        """
        try:
            vision_client = self.get_vision_client()
            base64_image = self.img_to_base64(img)

            vision_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text":  self.VISION_OCR_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ]

            response = vision_client.chat. completions.create(
                model=self.VISION_MODEL_NAME,
                messages=vision_messages,
                max_tokens=4096,
                temperature=0.1,
            )

            return response.choices[0].message. content

        except Exception as e:
            logger.error(f"视觉模型处理失败:  {e}")
            return f"[Error: Could not process image - {e}]"

    def process_images_to_context(self, images: List[Image.Image]) -> str:
        """
        处理多张图片，生成上下文描述
        
        这个方法替代了原来的 format_image_for_llm，
        将图片转换为文字描述而不是 base64 编码
        """
        if not images:
            return ""

        descriptions = []
        logger.info(f"=== 视觉模型处理 {len(images)} 张图片 ===")

        for idx, img in enumerate(images):
            logger.info(f"处理第 {idx+1}/{len(images)} 张图片...")
            description = self.process_image_with_vision(img)
            descriptions.append(f"[Image {idx+1} Content]:\n{description}")
            logger.info(f"图片 {idx+1} 处理完成，描述长度: {len(description)} 字符")

        return "\n\n".join(descriptions)

    def __call__(
        self,
        prompt: str,  # Marker 传入的动态 prompt（来自各种 Processor）
        image: PIL.Image.Image | List[PIL.Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        """
        重写父类的调用方法
        
        关键点：
        - prompt 是动态的，由 Marker 的各种 Processor 生成
        - 我们不修改 prompt 的内容，只是在前面添加图片描述
        - 保持与原始 OpenAIService 相同的接口和返回格式
        """
        if max_retries is None:
            max_retries = self.max_retries
        if timeout is None:
            timeout = self. timeout

        main_client = self.get_client()

        logger.info(f"=== PipelineQwenService 调用 ===")
        logger.info(f"主 LLM Model: {self.openai_model}")
        logger.info(f"动态 Prompt 长度: {len(prompt)} 字符")
        logger.info(f"是否有图片:  {image is not None}")

        # =================================================
        # 第一阶段：视觉处理 - 将图片转换为文字描述
        # =================================================
        vision_context = ""
        if image: 
            # 统一转换为列表
            if isinstance(image, Image.Image):
                images_list = [image]
            else:
                images_list = image

            # 使用视觉模型处理图片
            vision_context = self.process_images_to_context(images_list)

        # =================================================
        # 第二阶段：组装最终 Prompt
        # 结构：图片描述 + 原始动态 Prompt
        # =================================================
        if vision_context:
            # 将图片描述添加到原始 prompt 前面
            final_prompt = (
                f"=== IMAGE CONTENT (extracted by vision model) ===\n"
                f"{vision_context}\n"
                f"=== END IMAGE CONTENT ===\n\n"
                f"{prompt}"  # 保持原始 prompt 不变
            )
        else:
            final_prompt = prompt

        logger.info(f"最终 Prompt 长度: {len(final_prompt)} 字符")

        # =================================================
        # 第三阶段：构造消息（纯文本，不包含图片）
        # =================================================
        messages = [
            {
                "role":  "user",
                "content": final_prompt
            }
        ]

        # =================================================
        # 第四阶段：调用主 LLM
        # =================================================
        total_tries = max_retries + 1
        for tries in range(1, total_tries + 1):
            try:
                logger. info(f"调用主 LLM (第 {tries}/{total_tries} 次)...")

                # 尝试使用 beta.chat.completions.parse（如果模型支持）
                response = main_client.beta.chat.completions.parse(
                    extra_headers={"X-Title": "Marker"},
                    model=self.openai_model,
                    messages=messages,
                    timeout=timeout,
                    response_format=response_schema,
                )

                response_text = response.choices[0].message.content
                logger.info(f"主 LLM 调用成功！响应长度: {len(response_text)} 字符")

                total_tokens = response.usage.total_tokens
                logger.info(f"消耗 tokens: {total_tokens}")

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
