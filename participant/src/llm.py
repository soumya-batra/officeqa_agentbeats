import os
import hashlib
import json
from pathlib import Path

from config import SolverConfig

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class LLMClient:
    def __init__(self, config: SolverConfig):
        self._config = config
        self._cache_path = config.llm_cache_path
        self._cache = self._load_cache()

    def complete(self, *, system_prompt: str, prompt: str) -> str:
        provider = self._config.llm_provider
        cache_key = self._cache_key(system_prompt=system_prompt, prompt=prompt, provider=provider)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        use_openai = (
            OPENAI_AVAILABLE
            and os.environ.get("OPENAI_API_KEY")
            and (provider == "openai" or (provider == "" and not os.environ.get("ANTHROPIC_API_KEY")))
        )
        use_anthropic = (
            ANTHROPIC_AVAILABLE
            and os.environ.get("ANTHROPIC_API_KEY")
            and (provider == "anthropic" or (provider == "" and not use_openai))
        )

        if use_openai:
            response = self._complete_openai(system_prompt=system_prompt, prompt=prompt)
            self._store_cache(cache_key, response)
            return response
        if use_anthropic:
            response = self._complete_anthropic(system_prompt=system_prompt, prompt=prompt)
            self._store_cache(cache_key, response)
            return response
        response = "<FINAL_ANSWER>Unable to determine - no LLM configured</FINAL_ANSWER>"
        self._store_cache(cache_key, response)
        return response

    def _cache_key(self, *, system_prompt: str, prompt: str, provider: str) -> str:
        payload = json.dumps(
            {
                "provider": provider,
                "openai_model": self._config.openai_model,
                "anthropic_model": self._config.anthropic_model,
                "system_prompt": system_prompt,
                "prompt": prompt,
                "reasoning_effort": self._config.reasoning_effort,
                "web_search": self._config.enable_web_search,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_cache(self) -> dict[str, str]:
        if self._cache_path is None or not self._cache_path.exists():
            return {}
        try:
            with self._cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(key): str(value) for key, value in payload.items()}

    def _store_cache(self, cache_key: str, response: str) -> None:
        if self._cache_path is None or cache_key in self._cache:
            return
        self._cache[cache_key] = response
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("w", encoding="utf-8") as handle:
            json.dump(self._cache, handle, indent=2, sort_keys=True)

    def _complete_openai(self, *, system_prompt: str, prompt: str) -> str:
        client = OpenAI()
        model = self._config.openai_model
        if model.startswith("gpt-5"):
            kwargs = {
                "model": model,
                "instructions": system_prompt,
                "input": [{"role": "user", "content": prompt}],
                "tools": [{"type": "web_search"}] if self._config.enable_web_search else None,
            }
            if self._config.reasoning_effort:
                kwargs["reasoning"] = {"effort": self._config.reasoning_effort}
            else:
                kwargs["temperature"] = 0
            response = client.responses.create(**kwargs)
            return response.output_text or ""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    def _complete_anthropic(self, *, system_prompt: str, prompt: str) -> str:
        client = anthropic.Anthropic()
        kwargs = {
            "model": self._config.anthropic_model,
            "max_tokens": self._config.anthropic_max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        if self._config.enable_web_search:
            kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 10}]
        response = client.messages.create(**kwargs)
        text_parts = [block.text for block in response.content if hasattr(block, "text")]
        return "\n".join(text_parts) if text_parts else ""
