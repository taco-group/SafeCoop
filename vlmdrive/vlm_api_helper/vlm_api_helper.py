from openai import OpenAI, AsyncOpenAI, APIConnectionError, RateLimitError
import httpx
import json
from PIL import Image
import numpy as np
import os, io, base64, asyncio
from typing import List, Optional

class VLMAPIHelper:

    def __init__(self, api_key, api_base_url, api_model_name, image_placeholder="<IMAGE_PLACEHOLDER>"):
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.api_model_name = api_model_name
        self.IMAGE_PLACEHOLDER = image_placeholder

    def encode_image_array(self, img_array, format="PNG"):
        """Converts a numpy array to base64 encoded JPEG or PNG."""
        img = Image.fromarray(img_array.astype('uint8'))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return base64_image

    def infer(self, images=[], text=None, sys_message=None):
        
        assert len(text) < 12800, "Text length exceeds 12800 characters limit."
        os.environ['HTTPX_PROXIES'] = ''
        os.environ['no_proxy'] = '*'
        
        if text is None and not images:
            raise ValueError("Either 'text' or 'images' must be provided.")
        if not images:
            images = []
            

        try:
            http_client = httpx.Client(transport=httpx.HTTPTransport(retries=3))
        except Exception:
            http_client = httpx.Client(transport=httpx.HTTPTransport(retries=3))
        client = OpenAI(base_url=self.api_base_url, api_key=self.api_key, http_client=http_client)

        # Check if text contains image placeholders.
        if self.IMAGE_PLACEHOLDER in text:
            num_placeholders = text.count(self.IMAGE_PLACEHOLDER)
            if num_placeholders != len(images):
                print(f"Number of images ({len(images)}) does not match number of image placeholders ({num_placeholders}) in text.")
                raise ValueError(f"Number of images ({len(images)}) does not match number of image placeholders ({num_placeholders}) in text.")

            text_parts = text.split(self.IMAGE_PLACEHOLDER)
            content = []

            for i in range(num_placeholders):
                if text_parts[i]:
                    content.append({"type": "text", "text": text_parts[i]})
                
                img_input = images[i]

                # 判断img_input是ndarray还是路径
                if isinstance(img_input, np.ndarray):
                    base64_image = self.encode_image_array(img_input)
                elif isinstance(img_input, str) and os.path.exists(img_input):
                    with open(img_input, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                else:
                    raise ValueError(f"Invalid image input at index {i}: Must be numpy array or existing file path.")

                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })

            if text_parts[-1]:
                content.append({"type": "text", "text": text_parts[-1]})
        else:
            content = [{"type": "text", "text": text}]
            for i, img_input in enumerate(images):
                if isinstance(img_input, np.ndarray):
                    base64_image = self.encode_image_array(img_input)
                elif isinstance(img_input, str) and os.path.exists(img_input):
                    with open(img_input, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                else:
                    raise ValueError(f"Invalid image input at index {i}: Must be numpy array or existing file path.")

                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })

        messages = [{"role": "user", "content": content}]
        if sys_message:
            messages.insert(0, {"role": "system", "content": sys_message})

        params = {"model": self.api_model_name, "messages": messages, "max_tokens": 4096}

        # Give Three Attempts to Get a Response
        for i in range(3):
            try:
                result = client.chat.completions.create(**params)
                content = result.choices[0].message.content
                break
            except:
                if i == 2:
                    raise Exception("Failed to get a response from the API after three attempts.")
        return content
    
    

class VLMAPIHelperAsync:
    def __init__(self, api_key, api_base_url, api_model_name, image_placeholder="<IMAGE_PLACEHOLDER>"):
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.api_model_name = api_model_name
        self.IMAGE_PLACEHOLDER = image_placeholder

        # one client reused across calls (connection pooling)
        self._httpx = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=3),
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
            trust_env=False  # ignore system proxy settings (safe for localhost clusters)
        )
        self._oaiclient = AsyncOpenAI(base_url=api_base_url, api_key=api_key, http_client=self._httpx)

        # optional: keep env clean, but trust_env=False above already disables proxy usage
        os.environ['no_proxy'] = '*'

    async def aclose(self):
        await self._httpx.aclose()

    @staticmethod
    def _encode_image_array(img_array: np.ndarray, format="PNG") -> str:
        img = Image.fromarray(img_array.astype('uint8'))
        buf = io.BytesIO()
        img.save(buf, format=format)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def _build_content(self, images: List, text: Optional[str]):
        if text is None and not images:
            raise ValueError("Either 'text' or 'images' must be provided.")
        if not images:
            images = []

        def to_b64(img_input):
            if isinstance(img_input, np.ndarray):
                return self._encode_image_array(img_input)
            elif isinstance(img_input, str) and os.path.exists(img_input):
                with open(img_input, "rb") as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            else:
                raise ValueError("Invalid image input: must be ndarray or existing file path.")

        if text and self.IMAGE_PLACEHOLDER in text:
            n = text.count(self.IMAGE_PLACEHOLDER)
            if n != len(images):
                raise ValueError(f"#images ({len(images)}) != #placeholders ({n})")
            parts, content = text.split(self.IMAGE_PLACEHOLDER), []
            for i in range(n):
                if parts[i]:
                    content.append({"type": "text", "text": parts[i]})
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{to_b64(images[i])}"}})
            if parts[-1]:
                content.append({"type": "text", "text": parts[-1]})
            return content

        content = [{"type": "text", "text": text or ""}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{to_b64(img)}"}})
        return content

    async def ainfer(self, images: List = [], text: Optional[str] = None, sys_message: Optional[str] = None, *,
                     max_tokens: int = 4096, timeout_s: float = 75.0, max_retries: int = 3) -> str:
        assert len(text) < 12800, "Text length exceeds 12800 characters limit."
        """Async inference with simple exponential backoff."""
        content = self._build_content(images, text)
        messages = [{"role": "user", "content": content}]
        if sys_message:
            messages.insert(0, {"role": "system", "content": sys_message})

        params = {"model": self.api_model_name, 
                  "messages": messages, 
                  "max_tokens": max_tokens,
                  }

        delay = 0.2
        for attempt in range(1, max_retries + 1):
            try:
                resp = await asyncio.wait_for(
                    self._oaiclient.chat.completions.create(**params),
                    timeout=timeout_s
                )
                return resp.choices[0].message.content
            except (APIConnectionError, RateLimitError, httpx.TimeoutException, asyncio.TimeoutError) as e:
                if attempt == max_retries:
                    raise
                await asyncio.sleep(delay)