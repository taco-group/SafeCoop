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
    """
    Minimal async mirror of VLMAPIHelper.

    - provider: "openai", "openrouter" or other OpenAI-compatible servers
    - If provider == "openai" and use_responses=True and model looks like GPT-5,
      we use the Responses API to enable reasoning/verbosity knobs.
    - Otherwise we fall back to Chat Completions (broadest compatibility).
    """

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        api_model_name: str = "gpt-4o-mini",
        use_responses: bool = True,
        timeout_s: float = 60.0,
        retries: int = 3,
        api_base_url: Optional[str] = None,
        image_placeholder: str = "<IMAGE_PLACEHOLDER>",
    ):
        self.provider = provider.lower().strip()
        self.api_model_name = api_model_name
        self.use_responses = use_responses if self.provider == "openai" else False
        self.retries = max(1, retries)
        self.timeout_s = timeout_s
        self.IMAGE_PLACEHOLDER = image_placeholder

        # default key routing
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY") if self.provider == "openai" else None

        # single async httpx client for connection pooling
        self._httpx = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=3),
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
            trust_env=False,
        )

        if self.provider == "openai":
            # Let SDK use its defaults when talking to OpenAI
            self._oaiclient = AsyncOpenAI()
        else:
            # OpenAI-compatible servers (e.g., OpenRouter/proxy)
            self._oaiclient = AsyncOpenAI(
                api_key=api_key,
                base_url=api_base_url,
                http_client=self._httpx,
            )

        # avoid system proxies accidentally hijacking localhost
        os.environ["no_proxy"] = "*"

    async def aclose(self):
        await self._httpx.aclose()

    # ---------- Utilities (same behavior as sync helper) ----------

    @staticmethod
    def _encode_image_array(img: np.ndarray, fmt: str = "jpeg") -> str:
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
        if self.IMAGE_PLACEHOLDER in (text or ""):
            n = text.count(self.IMAGE_PLACEHOLDER)
            if n != len(images):
                raise ValueError(f"#images ({len(images)}) != #placeholders ({n})")
            parts = text.split(self.IMAGE_PLACEHOLDER)
            out: List[Dict[str, Any]] = []
            for i in range(n):
                if parts[i]:
                    out.append({"type": "text", "text": parts[i]})
                img = images[i]
                if isinstance(img, np.ndarray):
                    b64 = self._encode_image_array(img, fmt)
                elif isinstance(img, str) and os.path.exists(img):
                    b64 = self._encode_image_path(img)
                else:
                    raise ValueError(f"Invalid image at index {i}: ndarray or existing file path required")
                out.append({"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{b64}"}})
            if parts[-1]:
                out.append({"type": "text", "text": parts[-1]})
            return out

        # no placeholders: text then images
        out: List[Dict[str, Any]] = []
        if text:
            out.append({"type": "text", "text": text})
        for i, img in enumerate(images or []):
            if isinstance(img, np.ndarray):
                b64 = self._encode_image_array(img, fmt)
            elif isinstance(img, str) and os.path.exists(img):
                b64 = self._encode_image_path(img)
            else:
                raise ValueError(f"Invalid image at index {i}: ndarray or existing file path required")
            out.append({"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{b64}"}})
        return out

    def _build_responses_content_parts(
        self,
        text: str,
        images: List[Union[np.ndarray, str]],
        fmt: str = "jpeg",
    ) -> List[Dict[str, Any]]:
        if self.IMAGE_PLACEHOLDER in (text or ""):
            n = text.count(self.IMAGE_PLACEHOLDER)
            if n != len(images):
                raise ValueError(f"#images ({len(images)}) != #placeholders ({n})")
            parts = text.split(self.IMAGE_PLACEHOLDER)
            out: List[Dict[str, Any]] = []
            for i in range(n):
                if parts[i]:
                    out.append({"type": "input_text", "text": parts[i]})
                img = images[i]
                if isinstance(img, np.ndarray):
                    b64 = self._encode_image_array(img, fmt)
                elif isinstance(img, str) and os.path.exists(img):
                    b64 = self._encode_image_path(img)
                else:
                    raise ValueError(f"Invalid image at index {i}: ndarray or existing file path required")
                out.append({"type": "input_image", "image_url": f"data:image/{fmt};base64,{b64}"})
            if parts[-1]:
                out.append({"type": "input_text", "text": parts[-1]})
            return out

        out: List[Dict[str, Any]] = []
        if text:
            out.append({"type": "input_text", "text": text})
        for i, img in enumerate(images or []):
            if isinstance(img, np.ndarray):
                b64 = self._encode_image_array(img, fmt)
            elif isinstance(img, str) and os.path.exists(img):
                b64 = self._encode_image_path(img)
            else:
                raise ValueError(f"Invalid image at index {i}: ndarray or existing file path required")
            out.append({"type": "input_image", "image_url": f"data:image/{fmt};base64,{b64}"})
        return out

    @staticmethod
    def _is_gpt5_like(model: str) -> bool:
        m = model.lower()
        return m.startswith("gpt-5") or "/gpt-5" in m

    # ---------- Public API ----------

    async def ainfer(
        self,
        images: Optional[List[Union[np.ndarray, str]]] = None,
        text: str = None,
        sys_message: Optional[str] = None,
        max_output_tokens: int = None,
        reasoning_effort: Optional[str] = "low",   # "minimal" | "low" | "medium" | "high"
        verbosity: Optional[str] = None,          # "low" | "medium" | "high"
        fmt: str = "jpeg",
    ) -> str:
        """
        Async generation with retries. Mirrors VLMAPIHelper.infer behavior.
        """
        images = images or []

        if self.provider == "openai" and self.use_responses and self._is_gpt5_like(self.api_model_name):
            # Responses API path
            user_content = self._build_responses_content_parts(text or "", images, fmt=fmt)
            input_payload = [{"role": "user", "content": user_content}]
            if sys_message:
                input_payload.insert(0, {"role": "system", "content": [{"type": "input_text", "text": sys_message}]})

            payload: Dict[str, Any] = {
                "model": self.api_model_name,
                "input": input_payload,
                "max_output_tokens": max_output_tokens,
            }
            if reasoning_effort:
                payload["reasoning"] = {"effort": reasoning_effort}
            if verbosity:
                payload["text"] = {"verbosity": verbosity}

            return await self._with_retries_async(self._call_openai_responses_async, payload)

        # Chat Completions path (default / non-OpenAI / non-GPT-5)
        content = self._build_content_parts(text or "", images, fmt=fmt)
        messages: List[Dict[str, Any]] = [{"role": "user", "content": content}]
        if sys_message:
            messages.insert(0, {"role": "system", "content": sys_message})
        payload = {
            "model": self.api_model_name,
            "messages": messages,
            "max_tokens": max_output_tokens,
        }
        return await self._with_retries_async(self._call_openai_chat_async, payload)

    # ---------- Low-level async call wrappers ----------

    async def _call_openai_chat_async(self, payload: Dict[str, Any]) -> str:
        resp = await self._oaiclient.chat.completions.create(**payload)
        return resp.choices[0].message.content

    async def _call_openai_responses_async(self, payload: Dict[str, Any]) -> str:
        resp = await self._oaiclient.responses.create(**payload)
        # Prefer SDK convenience if available
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text
        # Fallback to stitching text segments
        if hasattr(resp, "output") and resp.output:
            parts = []
            for seg in resp.output:
                if getattr(seg, "type", None) == "output_text":
                    parts.append(getattr(seg, "text", "") or "")
            if parts:
                return "".join(parts)
        raise RuntimeError("Unexpected Responses payload; no output_text/output segments present")

    async def _with_retries_async(self, fn, payload: Dict[str, Any]) -> str:
        delay = 0.2
        last_e = None
        for attempt in range(self.retries):
            try:
                return await asyncio.wait_for(fn(payload), timeout=self.timeout_s)
            except (APIConnectionError, RateLimitError, httpx.TimeoutException, asyncio.TimeoutError) as e:
                last_e = e
                if attempt == self.retries - 1:
                    raise
                await asyncio.sleep(delay * (attempt + 1))
            except Exception as e:
                # If the server rejects reasoning/text fields, retry once without them (common portability hack)
                last_e = e
                # remove optional fields if present; safe no-ops otherwise
                payload.pop("reasoning", None)
                payload.pop("text", None)
                try:
                    return await asyncio.wait_for(fn(payload), timeout=self.timeout_s)
                except Exception as inner:
                    last_e = inner
                    if attempt == self.retries - 1:
                        raise
                    await asyncio.sleep(delay * (attempt + 1))
        # Should not reach here
        raise last_e