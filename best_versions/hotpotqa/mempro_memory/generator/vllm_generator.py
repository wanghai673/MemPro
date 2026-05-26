# mempro_memory/generator/vllm_generator.py
import time
import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from tqdm import tqdm

from mempro_memory.generator.base import AbsGenerator  
# 如果你有专门的 Config dataclass，也可像 OpenAIGenerator 一样 from_config

class VLLMGenerator(AbsGenerator):
    """
    使用 vLLM 的 OpenAI 兼容端点，并通过 guided_json 做结构化输出。
    与 OpenAIGenerator 的差异：不构造 response_format.json_schema，
    而是在 extra_body 里放 guided_json / guided_grammar / guided_regex 等。
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_name", "qwen2.5-14b-instruct")
        self.api_key = config.get("api_key", "empty")
        self.base_url = config.get("base_url", "http://localhost:8000/v1")
        self.n = config.get("n", 1)
        self.temperature = config.get("temperature", 0.0)
        self.top_p = config.get("top_p", 1.0)
        self.top_k = config.get("top_k")
        self.min_p = config.get("min_p")
        self.max_tokens = config.get("max_tokens", 10000)
        self.thread_count = config.get("thread_count")
        self.system_prompt = config.get("system_prompt")
        self.timeout = config.get("timeout", 60.0)
        self.use_schema = config.get("use_schema", False)

        # 兼容 openai SDK 的环境变量（可选）
        if self.api_key is not None:
            os.environ["OPENAI_API_KEY"] = self.api_key
        if self.base_url is not None:
            os.environ["OPENAI_BASE_URL"] = self.base_url

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url.rstrip("/"))
        # 某些 openai 版本支持 with_options
        self._cclient = (
            self._client.with_options(timeout=self.timeout)
            if hasattr(self._client, "with_options") else self._client
        )

    def _build_messages(
        self,
        prompt: Optional[str],
        messages: Optional[List[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        if (prompt is None) and (not messages):
            raise ValueError("Either prompt or messages is required.")
        if (prompt is not None) and messages:
            raise ValueError("Pass either prompt or messages, not both.")
        # 构造 messages
        if messages is None:
            messages = [{"role": "user", "content": prompt}]  # type: ignore[arg-type]
        if self.system_prompt and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        return messages

    def _extract_message_text(self, resp: Any) -> str:
        try:
            message = resp.choices[0].message
        except Exception:
            return ""

        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            text = content
        else:
            text = ""

        if not text and hasattr(message, "model_dump"):
            try:
                message_dump = message.model_dump()
            except Exception:
                message_dump = {}
            if isinstance(message_dump, dict):
                dump_content = message_dump.get("content")
                if isinstance(dump_content, str) and dump_content.strip():
                    text = dump_content

        if text and "</think>" in text:
            text = text.split("</think>")[-1]

        return text.strip()

    @staticmethod
    def _extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        left = text.find("{")
        right = text.rfind("}")
        if left == -1 or right == -1 or right <= left:
            return None
        try:
            return json.loads(text[left:right + 1])
        except Exception:
            return None

    def generate_single(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        极简 Chat 调用（vLLM OpenAI 兼容端点）
        - 二选一：prompt 文本 或 messages 列表
        - 若传 schema：通过 extra_body={"guided_json": schema} 进行结构化输出
        返回：
          {"text": str, "json": dict|None, "response": dict}
        """
        msgs = self._build_messages(prompt, messages)

        # vLLM 结构化输出的推荐用法：guided_json
        # 也可换成 guided_grammar / guided_regex / guided_choice
        extra_body: Dict[str, Any] = {}
        if schema is not None and self.use_schema:
            extra_body["guided_json"] = schema

        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": msgs,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.top_k is not None:
            params["top_k"] = self.top_k
        if self.min_p is not None:
            params["min_p"] = self.min_p
        if extra_params:
            # 用户自定义的其他 vLLM 扩展参数（如 guided_choice 等）
            params.update(extra_params)
        if extra_body:
            params["extra_body"] = {**params.get("extra_body", {}), **extra_body}

        times = 0
        while True:
            try:
                resp = self._cclient.chat.completions.create(**params)
                break
            except Exception as e:
                print(str(e), "times:", times)
                times += 1
                if times > 3:  # 最多重试3次
                    raise e
                time.sleep(5)

        text = self._extract_message_text(resp)

        out: Dict[str, Any] = {"text": text, "json": None, "response": resp.model_dump()}
        if schema is not None:
            # vLLM 的 guided_json 会尽量保证合法 JSON，但仍建议做一次解析
            out["json"] = self._extract_json_payload(text)
        return out

    def generate_batch(
        self,
        prompts: Optional[List[str]] = None,
        messages_list: Optional[List[List[Dict[str, str]]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量生成响应
        返回格式: [{"text": str, "json": dict|None, "response": dict}, ...]
        """
        if (prompts is None) and (not messages_list):
            raise ValueError("Either prompts or messages_list is required.")
        if (prompts is not None) and messages_list:
            raise ValueError("Pass either prompts or messages_list, not both.")

        if prompts is not None:
            if isinstance(prompts, str):
                prompts = [prompts]
            messages_list = [[{"role": "user", "content": p}] for p in prompts]

        thread_count = self.thread_count or cpu_count()

        def _worker(msgs):
            return self.generate_single(
                messages=msgs,
                schema=schema,
                extra_params=extra_params,
            )

        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            results = list(tqdm(executor.map(_worker, messages_list), total=len(messages_list)))
        return results

    @classmethod
    def from_config(cls, config) -> "VLLMGenerator":
        """与 OpenAIGenerator 的 from_config 对齐；config 可传 dict 或你的 dataclass"""
        if hasattr(config, "__dict__"):
            return cls(config.__dict__)
        return cls(config)
