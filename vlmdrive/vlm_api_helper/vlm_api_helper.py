import os
from openai import OpenAI
import httpx
import base64
import json
from PIL import Image
import io
import numpy as np

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

    def infer(self, images, text, sys_message=None):
        os.environ['HTTPX_PROXIES'] = ''
        os.environ['no_proxy'] = '*'

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

        params = {"model": self.api_model_name, "messages": messages, "max_tokens": 2048}

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