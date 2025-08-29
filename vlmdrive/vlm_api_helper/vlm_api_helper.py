from openai import OpenAI, AsyncOpenAI, APIConnectionError, RateLimitError
import httpx
import json
from PIL import Image
import numpy as np
import os, io, base64, asyncio
from typing import List, Optional

import os
import base64
import time
from typing import List, Optional, Union, Dict, Any

import httpx
import numpy as np
from openai import OpenAI


class VLMAPIHelper:
    """.
    - provider: "openai", "openrouter" or others (OpenAI-compatible)
    - For OpenAI:
        * default: chat.completions.create (broadest compatibility)
        * if use_responses=True: responses.create (to use reasoning/verbosity for GPT-5)
    - For Others:
        * uses OpenAI SDK with base_url
    """

    # IMAGE_PLACEHOLDER = "<IMAGE_PLACEHOLDER>"

    def __init__(
        self,
        provider: str = "openai",              # "openai" | "openrouter" | ...
        api_key: Optional[str] = None,
        api_model_name: str = "gpt-4o-mini",
        use_responses: bool = True,           # use Responses API (OpenAI only)
        timeout_s: float = 60.0,
        retries: int = 3,
        api_base_url: Optional[str] = None,        # override if you have a proxy
        image_placeholder="<IMAGE_PLACEHOLDER>",
    ):
        self.provider = provider.lower().strip()
        self.api_model_name = api_model_name
        self.use_responses = use_responses if self.provider == "openai" else False
        self.retries = max(1, retries)
        self.IMAGE_PLACEHOLDER = image_placeholder

        # Decide API key & base_url
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY") if self.provider == "openai" else None
            
        api_base_url = api_base_url  # could be None -> SDK default

        # httpx client for connection pooling
        self._httpx = httpx.Client(
            transport=httpx.HTTPTransport(retries=3),
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
        )

        if self.provider == "openai":
            self.client = OpenAI()
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url=api_base_url,
                http_client=self._httpx,
            )


    # ---------- Utilities ----------

    @staticmethod
    def _encode_image_array(img: np.ndarray, fmt: str = "jpeg") -> str:
        """
        Encode numpy image (H,W,3 or 4, uint8) to base64.
        """
        import cv2
        ok, buf = cv2.imencode(f".{fmt}", img)
        if not ok:
            raise ValueError("Failed to encode numpy image.")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    @staticmethod
    def _encode_image_path(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _build_content_parts(
        self,
        text: str,
        images: List[Union[np.ndarray, str]],
        fmt: str = "jpeg",
    ) -> List[Dict[str, Any]]:
        """
        Build OpenAI-compatible content list with text + image_url parts.
        Supports <IMAGE_PLACEHOLDER> alignment or appends images after text if no placeholder.
        """
        if self.IMAGE_PLACEHOLDER in text:
            num_placeholders = text.count(self.IMAGE_PLACEHOLDER)
            if num_placeholders != len(images):
                raise ValueError(
                    f"#images ({len(images)}) != #placeholders ({num_placeholders})"
                )

            text_parts = text.split(self.IMAGE_PLACEHOLDER)
            content = []
            for i in range(num_placeholders):
                if text_parts[i]:
                    content.append({"type": "text", "text": text_parts[i]})

                img_input = images[i]
                if isinstance(img_input, np.ndarray):
                    b64 = self._encode_image_array(img_input, fmt=fmt)
                elif isinstance(img_input, str) and os.path.exists(img_input):
                    b64 = self._encode_image_path(img_input)
                else:
                    raise ValueError(f"Invalid image at index {i}: ndarray or existing file path required")

                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{fmt};base64,{b64}"}
                })

            if text_parts[-1]:
                content.append({"type": "text", "text": text_parts[-1]})
            return content

        # no placeholder: put text first, then images
        content = [{"type": "text", "text": text}]
        for i, img_input in enumerate(images):
            if isinstance(img_input, np.ndarray):
                b64 = self._encode_image_array(img_input, fmt=fmt)
            elif isinstance(img_input, str) and os.path.exists(img_input):
                b64 = self._encode_image_path(img_input)
            else:
                raise ValueError(f"Invalid image at index {i}: ndarray or existing file path required")

            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{fmt};base64,{b64}"}
            })
        return content

    def _build_responses_content_parts(
        self,
        text: str,
        images: List[Union[np.ndarray, str]],
        fmt: str = "jpeg",
    ) -> List[Dict[str, Any]]:
        """
        Build content parts for **Responses API** (types must be input_text / input_image).
        Mirrors placeholder alignment logic from _build_content_parts.
        """
        if self.IMAGE_PLACEHOLDER in text:
            n = text.count(self.IMAGE_PLACEHOLDER)
            if n != len(images):
                raise ValueError(f"#images ({len(images)}) != #placeholders ({n})")
            parts = text.split(self.IMAGE_PLACEHOLDER)
            content: List[Dict[str, Any]] = []
            for i in range(n):
                if parts[i]:
                    content.append({"type": "input_text", "text": parts[i]})
                img_input = images[i]
                if isinstance(img_input, np.ndarray):
                    b64 = self._encode_image_array(img_input, fmt=fmt)
                    url = f"data:image/{fmt};base64,{b64}"
                elif isinstance(img_input, str) and os.path.exists(img_input):
                    b64 = self._encode_image_path(img_input)
                    url = f"data:image/{fmt};base64,{b64}"
                else:
                    raise ValueError(f"Invalid image at index {i}: ndarray or existing file path required")
                content.append({
                    "type": "input_image",
                    "image_url": url
                })
            if parts[-1]:
                content.append({"type": "input_text", "text": parts[-1]})
            return content

        # No placeholder: text first, then images
        content: List[Dict[str, Any]] = []
        if text:
            content.append({"type": "input_text", "text": text})
        for img_input in images:
            if isinstance(img_input, np.ndarray):
                b64 = self._encode_image_array(img_input, fmt=fmt)
                url = f"data:image/{fmt};base64,{b64}"
            elif isinstance(img_input, str) and os.path.exists(img_input):
                b64 = self._encode_image_path(img_input)
                url = f"data:image/{fmt};base64,{b64}"
            else:
                raise ValueError("Invalid image input: ndarray or existing file path required")
            content.append({
                "type": "input_image",
                "image_url": url
            })
        return content

    @staticmethod
    def _is_gpt5_like(model: str) -> bool:
        m = model.lower()
        return m.startswith("gpt-5") or "/gpt-5" in m 

    # ---------- Public API ----------

    def infer(
        self,
        images: Optional[List[Union[np.ndarray, str]]] = None,
        text: str = None,
        sys_message: Optional[str] = None,
        max_output_tokens: int = None,
        reasoning_effort: Optional[str] = "low",   # "minimal" | "low" | "medium" | "high"
        verbosity: Optional[str] = None,          # "low" | "medium" | "high"
    ) -> str:
        """
        Generate a response from the selected provider.
        - text: user text, may include <IMAGE_PLACEHOLDER>
        - images: list of numpy arrays or file paths
        - sys_message: optional system prompt
        - reasoning_effort / verbosity:
            * OpenAI + Responses API only (if self.use_responses=True AND model is GPT-5-ish)
            * Ignored otherwise (silently)
        """
        images = images or []

        # Build content
        # OpenAI path
        if self.provider == "openai":
            if self.use_responses and self._is_gpt5_like(self.api_model_name):
                # ---- Responses API (to use reasoning/verbosity) ----
                # shape input in a messages-like structure
                user_content = self._build_responses_content_parts(text or "", images)
                input_payload = [{"role": "user", "content": user_content}]
                if sys_message:
                    input_payload.insert(0, {"role": "system", "content": [{"type": "input_text", "text": sys_message}]})

                payload = {
                    "model": self.api_model_name,
                    "input": input_payload,
                    "max_output_tokens": max_output_tokens,
                }
                if reasoning_effort:
                    payload["reasoning"] = {"effort": reasoning_effort}
                if verbosity:
                    payload["text"] = {"verbosity": verbosity}

                return self._with_retries(self._call_openai_responses, payload)

            else:
                # ---- Chat Completions API (no reasoning/verbosity here) ----
                content = self._build_content_parts(text, images)
                messages = [{"role": "user", "content": content}]
                if sys_message:
                    messages.insert(0, {"role": "system", "content": sys_message})
                payload = {
                    "model": self.api_model_name,
                    "messages": messages,
                    "max_tokens": max_output_tokens,
                }
                return self._with_retries(self._call_openai_chat, payload)

        else:
            # Use Chat Completions for widest compatibility across models.
            #  but it's vendor/model-specific. We deliberately keep it simple and stable.)
            content = self._build_content_parts(text, images)
            messages = [{"role": "user", "content": content}]
            if sys_message:
                messages.insert(0, {"role": "system", "content": sys_message})
            payload = {
                "model": self.api_model_name,           # e.g., "openai/gpt-4o-mini", "openai/gpt-5"
                "messages": messages,
                "max_tokens": max_output_tokens,
            }
            return self._with_retries(self._call_openai_chat, payload)

    # ---------- Low-level call wrappers ----------

    def _call_openai_chat(self, payload: Dict[str, Any]) -> str:
        resp = self.client.chat.completions.create(**payload)
        return resp.choices[0].message.content

    def _call_openai_responses(self, payload: Dict[str, Any]) -> str:
        resp = self.client.responses.create(**payload)
        # SDK exposes a convenience: resp.output_text
        # If not present in your SDK version, you can reconstruct from resp.output
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text
        # Fallback: try to stitch text segments together
        if hasattr(resp, "output") and resp.output:
            parts = []
            for seg in resp.output:
                if getattr(seg, "type", None) == "output_text":
                    parts.append(getattr(seg, "text", "") or "")
            if parts:
                return "".join(parts)
        # Last resort
        raise RuntimeError("Unexpected Responses payload; no output_text/output segments present")

    def _with_retries(self, fn, payload: Dict[str, Any]) -> str:
        last_e = None
        for attempt in range(self.retries):
            try:
                return fn(payload)
            except Exception as e:
                last_e = e
                # simple backoff
                time.sleep(0.2 * (attempt + 1))
        raise last_e
    

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
                    self._oaiclient.chat.completions.create(
                        **params,
                        reasoning={
                            "effort": "low"
                        },
                        text={
                            "verbosity": "low"
                        }
                        ),
                    timeout=timeout_s
                )
                return resp.choices[0].message.content
            except (APIConnectionError, RateLimitError, httpx.TimeoutException, asyncio.TimeoutError) as e:
                if attempt == max_retries:
                    raise
                await asyncio.sleep(delay)
            except Exception as e:
                # import traceback; traceback.print_exc()
                print("Retrying without reasoning and text verbosity parameters...")
                resp = await asyncio.wait_for(
                    self._oaiclient.chat.completions.create(**params),
                    timeout=timeout_s
                )
                return resp.choices[0].message.content