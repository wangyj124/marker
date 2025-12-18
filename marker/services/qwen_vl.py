"""
QwenVL Service - 适配本地部署的 Qwen-VL 模型（LMStudio / vLLM / Ollama）

命令行使用: 
    marker_single your.pdf --use_llm \
        --llm_service marker.services.qwen_vl.QwenVLService \
        --openai_base_url http://localhost:1234/v1 \
        --openai_model qwen2.5-vl-7b-instruct

Python 使用: 
    from marker.services.qwen_vl import QwenVLService
    
    service = QwenVLService(config={
        "openai_base_url": "http://localhost:1234/v1",
        "openai_model": "qwen2.5-vl-7b-instruct",
    })
"""

import base64
import io
import json
import time
from typing import Annotated, List

import openai
import PIL
from PIL import Image
from pydantic import BaseModel
from openai import APITimeoutError, RateLimitError

from marker.logger import get_logger
from marker.schema. blocks import Block
from marker.services.openai import OpenAIService

logger = get_logger()


def decode_base64_to_image(base64_str:  str) -> Image.Image:
    """
    将 Base64 字符串转为 RGB 图像
    
    Args:
        base64_str: Base64 编码的图像字符串，可包含 data:image/xxx;base64, 前缀
        
    Returns: 
        PIL.Image. Image: RGB 格式的图像对象
        
    Raises:
        ValueError: 如果 Base64 数据无效
    """
    try: 
        # 去掉 data:image/xxx;base64, 前缀
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]

        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return image. convert("RGB")
    except Exception as e:
        raise ValueError(f"无效的 Base64 图像数据: {e}")


class QwenVLService(OpenAIService):
    """
    适配本地部署 Qwen-VL 模型的服务类
    
    继承自 OpenAIService，完全兼容命令行参数: 
    - --openai_base_url: LMStudio/vLLM 服务地址
    - --openai_model: 模型名称
    - --openai_api_key: API Key（本地部署通常不需要）
    - --openai_image_format: 图像格式（推荐 png）
    
    新增参数: 
    - --qwen_preprocess_image: 是否预处理图像
    - --qwen_max_image_size: 图像最大尺寸
    """
    
    # 覆盖默认值，适配本地部署
    openai_base_url: Annotated[
        str, 
        "LMStudio/vLLM 本地服务地址"
    ] = "http://localhost:1234/v1"
    
    openai_model:  Annotated[
        str, 
        "Qwen-VL 模型名称"
    ] = "qwen2.5-vl-7b-instruct"
    
    openai_api_key:  Annotated[
        str, 
        "API Key（本地部署设置任意值）"
    ] = "lm-studio"
    
    openai_image_format:  Annotated[
        str, 
        "图像格式，推荐 png 以获得更好兼容性"
    ] = "png"
    
    # Qwen-VL 特定配置
    qwen_preprocess_image: Annotated[
        bool,
        "是否预处理图像（调整尺寸、标准化色彩空间）"
    ] = True
    
    qwen_max_image_size: Annotated[
        int,
        "图像最大边长，超过将等比缩放"
    ] = 1280
    
    timeout: Annotated[
        int,
        "请求超时时间（秒），本地推理通常需要更长时间"
    ] = 120
    
    def process_images(self, images: List[Image.Image]) -> List[dict]:
        """
        处理图像为 OpenAI 兼容的消息格式
        
        针对 Qwen-VL 优化: 
        1. 使用 PNG 格式
        2. 可选的图像预处理
        3. 尺寸限制以控制显存使用
        """
        if isinstance(images, Image.Image):
            images = [images]

        img_fmt = self.openai_image_format
        processed_images = []
        
        for img in images:
            if self.qwen_preprocess_image:
                img = self._preprocess_image(img)
            
            base64_str = self. img_to_base64(img, format=img_fmt. upper())
            
            processed_images.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{img_fmt.lower()};base64,{base64_str}",
                },
            })
        
        return processed_images
    
    def _preprocess_image(self, img:  Image.Image) -> Image.Image:
        """
        预处理图像以确保兼容性和性能
        """
        # 确保是 RGB 格式
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        
        # 限制最大尺寸
        max_size = self.qwen_max_image_size
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"图像缩放:  {img.size} -> {new_size}")
        
        return img

    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | List[PIL.Image.Image] | None,
        block:  Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout:  int | None = None,
    ):
        """
        调用 Qwen-VL 模型进行推理
        
        与父类的区别: 
        1. 更长的默认超时
        2. 结构化输出回退机制
        3. 更详细的错误处理
        """
        if max_retries is None:
            max_retries = self.max_retries

        if timeout is None:
            timeout = self.timeout

        client = self.get_client()
        image_data = self.format_image_for_llm(image)

        messages = [
            {
                "role": "user",
                "content": [
                    *image_data,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        total_tries = max_retries + 1
        for tries in range(1, total_tries + 1):
            try:
                response = self._call_model(client, messages, response_schema, timeout)
                
                if response is not None:
                    response_text = response.choices[0].message.content
                    total_tokens = response.usage.total_tokens if response.usage else 0
                    
                    if block:
                        block.update_metadata(
                            llm_tokens_used=total_tokens, llm_request_count=1
                        )
                    
                    # 尝试解析 JSON
                    return self._parse_response(response_text, response_schema)
                    
            except (APITimeoutError, RateLimitError) as e:
                if tries == total_tries:
                    logger.error(f"请求错误: {e}.  已达最大重试次数。(尝试 {tries}/{total_tries})")
                    break
                else:
                    wait_time = tries * self.retry_wait_time
                    logger.warning(f"请求错误: {e}.  {wait_time} 秒后重试...  (尝试 {tries}/{total_tries})")
                    time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Qwen-VL 推理失败: {e}")
                break

        return {}
    
    def _call_model(
        self,
        client: openai.OpenAI,
        messages: list,
        response_schema: type[BaseModel],
        timeout: int
    ):
        """
        调用模型，支持结构化输出回退
        """
        # 首先尝试标准的结构化输出（beta. chat.completions.parse）
        try:
            return client.beta.chat.completions.parse(
                model=self.openai_model,
                messages=messages,
                timeout=timeout,
                response_format=response_schema,
            )
        except Exception as e: 
            logger.debug(f"结构化输出不支持，尝试普通调用: {e}")
        
        # 回退：使用普通调用 + JSON schema 提示
        schema_example = response_schema.model_json_schema()
        system_message = {
            "role": "system",
            "content": (
                f"你必须严格按照以下 JSON schema 格式响应，只输出 JSON，不要有其他内容:\n"
                f"{json.dumps(schema_example, indent=2, ensure_ascii=False)}"
            )
        }
        
        messages_with_schema = [system_message] + messages
        
        return client.chat.completions.create(
            model=self.openai_model,
            messages=messages_with_schema,
            timeout=timeout,
        )
    
    def _parse_response(self, response_text: str, response_schema: type[BaseModel]) -> dict:
        """
        解析模型响应，处理各种 JSON 格式
        """
        try:
            return json.loads(response_text)
        except json.JSONDecodeError: 
            pass
        
        # 尝试提取 JSON 代码块
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
        
        # 尝试找到 JSON 对象
        import re
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match. group())
            except:
                pass
        
        logger.warning(f"无法解析 JSON 响应: {response_text[: 200]}...")
        return {}

    def get_client(self) -> openai.OpenAI:
        """获取 OpenAI 客户端"""
        return openai. OpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_base_url,
        )
