"""
Pipeline Qwen Service - 视觉模型 + 文本模型流水线

工作流程：
1. 使用 Qwen3-VL 处理图片，生成结构化描述
2. 将图片描述 + Marker 动态 prompt 组合
3. 发送给 Qwen3 文本模型处理

命令行使用：
    marker_single your. pdf --use_llm \
        --llm_service marker. services.pipeline_qwen.PipelineQwenService \
        --vision_base_url http://10.8.2.63:12345/v1/ \
        --vision_model qwen3-vl-8b-instruct \
        --openai_base_url http://10.8.2.63:12345/v1/ \
        --openai_model qwen3-14b

Python 使用：
    from marker.services.pipeline_qwen import PipelineQwenService
    
    service = PipelineQwenService(config={
        "vision_base_url":  "http://10.8.2.63:12345/v1/",
        "vision_model": "qwen3-vl-8b-instruct",
        "openai_base_url": "http://10.8.2.63:12345/v1/",
        "openai_model": "qwen3-14b",
    })
"""

import json
import time
from typing import Annotated, List

import openai
import PIL
from PIL import Image
from pydantic import BaseModel
from openai import APITimeoutError, RateLimitError, OpenAI

from marker.logger import get_logger
from marker.schema. blocks import Block
from marker.services.openai import OpenAIService

logger = get_logger()


class PipelineQwenService(OpenAIService):
    """
    视觉模型 + 文本模型流水线服务
    
    继承 OpenAIService，复用其图片处理逻辑（Base64 编码），
    但将图片先发送给视觉模型处理，再将描述发送给文本模型。
    """
    
    # =========================================================
    # 视觉模型配置（Qwen3-VL）
    # =========================================================
    vision_base_url: Annotated[
        str,
        "视觉模型 API 地址"
    ] = "http://localhost:1234/v1/"
    
    vision_model: Annotated[
        str,
        "视觉模型名称"
    ] = "qwen3-vl-8b-instruct"
    
    vision_api_key: Annotated[
        str,
        "视觉模型 API Key"
    ] = "lm-studio"
    
    vision_timeout: Annotated[
        int,
        "视觉模型请求超时时间（秒）"
    ] = 120
    
    vision_max_tokens: Annotated[
        int,
        "视觉模型最大输出 token 数"
    ] = 4096
    
    vision_image_format: Annotated[
        str,
        "发送给视觉模型的图像格式（png 或 jpeg，不支持 webp）"
    ] = "png"
    
    vision_prompt: Annotated[
        str,
        "视觉模型的 OCR/描述提示词"
    ] = """You are an OCR and document analysis expert. 
Analyze this image and: 
1. Extract ALL text content accurately, preserving the original language
2. Describe the layout structure (tables, headers, lists, figures, etc.)
3. For tables, extract data in a structured format
4. For figures/charts, describe what they show
5. Preserve the original formatting as much as possible

Output the extracted content in a clear, structured format."""

    # =========================================================
    # 主文本模型配置（继承自 OpenAIService）
    # openai_base_url, openai_model, openai_api_key 等
    # =========================================================
    
    # 覆盖默认值
    openai_base_url: Annotated[
        str,
        "主文本模型 API 地址"
    ] = "http://localhost:1234/v1/"
    
    openai_model: Annotated[
        str,
        "主文本模型名称"
    ] = "qwen3-14b"
    
    openai_api_key:  Annotated[
        str,
        "主文本模型 API Key"
    ] = "lm-studio"
    
    # 图像格式（用于 Base64 编码）
    openai_image_format: Annotated[
        str,
        "图像编码格式"
    ] = "png"

    def get_vision_client(self) -> OpenAI:
        """获取视觉模型客户端"""
        return OpenAI(
            base_url=self.vision_base_url,
            api_key=self.vision_api_key
        )

    def process_image_with_vision(self, img: Image.Image) -> str:
        """
        使用视觉模型处理单张图片
        
        与 openai.py 的 process_images 逻辑一致：
        1. 将图片转为 Base64
        2. 构造 OpenAI 兼容的消息格式
        3. 发送请求并获取描述
        
        Args:
            img: PIL Image 对象
            
        Returns: 
            图片的文字描述
        """
        try:
            vision_client = self.get_vision_client()
            
            # 使用与 openai.py 相同的 Base64 编码逻辑
            img_fmt = self.vision_image_format
            base64_image = self. img_to_base64(img, format=img_fmt. upper())

            # 构造与 openai.py process_images 相同格式的消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": self.vision_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{img_fmt. lower()};base64,{base64_image}"
                            },
                        },
                    ],
                }
            ]

            response = vision_client.chat. completions.create(
                model=self.vision_model,
                messages=messages,
                max_tokens=self.vision_max_tokens,
                timeout=self.vision_timeout,
                temperature=0.1,
            )

            return response.choices[0].message. content

        except Exception as e:
            logger.error(f"视觉模型处理失败: {e}")
            return f"[Error: Could not process image - {e}]"

    def process_images_to_context(self, images: List[Image.Image]) -> str:
        """
        处理多张图片，生成上下文描述
        
        Args:
            images: PIL Image 列表
            
        Returns: 
            合并的图片描述文本
        """
        if not images:
            return ""

        descriptions = []
        logger.info(f"视觉模型处理 {len(images)} 张图片")

        for idx, img in enumerate(images):
            logger.debug(f"处理第 {idx+1}/{len(images)} 张图片...")
            description = self.process_image_with_vision(img)
            descriptions.append(f"[Image {idx+1} Content]:\n{description}")
            logger.debug(f"图片 {idx+1} 处理完成，描述长度: {len(description)} 字符")

        return "\n\n".join(descriptions)

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
        流水线调用：视觉模型处理图片 -> 文本模型生成结果
        
        Args: 
            prompt:  Marker 传入的动态 prompt
            image: 图片或图片列表
            block:  当前处理的 Block
            response_schema: 响应的 Pydantic schema
            max_retries: 最大重试次数
            timeout: 超时时间
            
        Returns:
            解析后的 JSON 字典
        """
        if max_retries is None:
            max_retries = self.max_retries
        if timeout is None:
            timeout = self. timeout

        logger.debug(f"PipelineQwenService:  vision={self.vision_model}, text={self.openai_model}")

        # =========================================================
        # 第一阶段：视觉处理 - 将图片转换为文字描述
        # =========================================================
        vision_context = ""
        if image: 
            images_list = [image] if isinstance(image, Image.Image) else image
            vision_context = self.process_images_to_context(images_list)

        # =========================================================
        # 第二阶段：组装最终 Prompt
        # =========================================================
        if vision_context:
            final_prompt = (
                f"=== IMAGE CONTENT (extracted by vision model) ===\n"
                f"{vision_context}\n"
                f"=== END IMAGE CONTENT ===\n\n"
                f"{prompt}"
            )
        else:
            final_prompt = prompt

        # =========================================================
        # 第三阶段：构造纯文本消息
        # =========================================================
        messages = [
            {
                "role": "user",
                "content": final_prompt
            }
        ]

        # =========================================================
        # 第四阶段：调用主文本 LLM
        # =========================================================
        main_client = self.get_client()
        total_tries = max_retries + 1
        
        for tries in range(1, total_tries + 1):
            try:
                # 尝试结构化输出
                response = self._call_text_model(
                    main_client, messages, response_schema, timeout
                )

                if response: 
                    response_text = response.choices[0].message.content
                    total_tokens = response.usage.total_tokens if response.usage else 0
                    
                    logger.debug(f"主 LLM 响应长度: {len(response_text)}, tokens: {total_tokens}")

                    if block:
                        block.update_metadata(
                            llm_tokens_used=total_tokens, llm_request_count=1
                        )

                    return self._parse_response(response_text, response_schema)

            except (APITimeoutError, RateLimitError) as e:
                if tries == total_tries:
                    logger.error(f"请求错误:  {e}. 已达最大重试次数。")
                    break
                wait_time = tries * self.retry_wait_time
                logger.warning(f"请求错误:  {e}. {wait_time} 秒后重试...")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"主 LLM 调用失败: {e}")
                break

        return {}

    def _call_text_model(
        self,
        client: openai.OpenAI,
        messages: list,
        response_schema: type[BaseModel],
        timeout: int
    ):
        """
        调用文本模型，支持结构化输出回退
        """
        # 首先尝试 beta. chat.completions.parse
        try:
            return client.beta.chat.completions.parse(
                extra_headers={"X-Title": "Marker"},
                model=self.openai_model,
                messages=messages,
                timeout=timeout,
                response_format=response_schema,
            )
        except Exception as e: 
            logger.debug(f"结构化输出不支持，回退到普通调用: {e}")
        
        # 回退：普通调用 + JSON schema 提示
        schema_example = response_schema.model_json_schema()
        system_message = {
            "role": "system",
            "content": (
                f"You must respond in JSON format matching this schema:\n"
                f"{json.dumps(schema_example, indent=2)}\n"
                f"Output only valid JSON, no other text."
            )
        }
        
        return client.chat.completions.create(
            model=self.openai_model,
            messages=[system_message] + messages,
            timeout=timeout,
        )

    def _parse_response(self, response_text: str, response_schema: type[BaseModel]) -> dict:
        """
        解析模型响应，处理各种 JSON 格式
        """
        # 直接解析
        try:
            return json. loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # 提取 JSON 代码块
        if "```json" in response_text:
            try:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            except:
                pass
        
        if "```" in response_text:
            try:
                json_str = response_text.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            except: 
                pass
        
        # 正则匹配 JSON 对象
        import re
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        logger.warning(f"无法解析 JSON:  {response_text[: 200]}...")
        return {}
