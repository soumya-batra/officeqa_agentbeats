import os
import hashlib
import json
import time
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

try:
    from google import genai
    from google.genai import types as genai_types

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


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

        use_gemini = (
            GEMINI_AVAILABLE
            and os.environ.get("GOOGLE_API_KEY")
            and (provider == "gemini" or (provider == "" and os.environ.get("GOOGLE_API_KEY") and not os.environ.get("OPENAI_API_KEY")))
        )
        use_openai = (
            OPENAI_AVAILABLE
            and os.environ.get("OPENAI_API_KEY")
            and (provider == "openai" or (provider == "" and not os.environ.get("ANTHROPIC_API_KEY") and not use_gemini))
        )
        use_anthropic = (
            ANTHROPIC_AVAILABLE
            and os.environ.get("ANTHROPIC_API_KEY")
            and (provider == "anthropic" or (provider == "" and not use_openai and not use_gemini))
        )

        if use_gemini:
            response = self._complete_gemini(system_prompt=system_prompt, prompt=prompt)
            self._store_cache(cache_key, response)
            return response
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
                "gemini_model": self._config.gemini_model,
                "system_prompt": system_prompt,
                "prompt": prompt,
                "reasoning_effort": self._config.reasoning_effort,
                "enable_web_search": self._config.enable_web_search,
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
        max_retries = 5
        for attempt in range(max_retries):
            try:
                return self._complete_openai_once(client=client, model=model, system_prompt=system_prompt, prompt=prompt)
            except Exception as e:
                err_str = str(e).lower()
                if "429" in str(e) or "rate_limit" in err_str or "insufficient_quota" in err_str:
                    wait = 2 ** attempt * 10  # 10s, 20s, 40s, 80s, 160s
                    import logging
                    logging.getLogger(__name__).warning(f"Rate limit hit, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                elif "invalid_prompt" in err_str or "usage policy" in err_str:
                    # Content policy filter triggered — truncate context and retry
                    import logging
                    logging.getLogger(__name__).warning(f"Content policy filter hit, truncating prompt (attempt {attempt+1})")
                    # Reduce prompt size by removing the last ~30% of context
                    lines = prompt.split("\n")
                    prompt = "\n".join(lines[:int(len(lines) * 0.7)])
                else:
                    raise
        return self._complete_openai_once(client=client, model=model, system_prompt=system_prompt, prompt=prompt)

    def _complete_openai_once(self, *, client, model: str, system_prompt: str, prompt: str) -> str:
        if model.startswith("gpt-5"):
            tools = []
            if self._config.enable_web_search:
                tools.append({"type": "web_search"})
            tools.append({"type": "code_interpreter", "container": {"type": "auto"}})
            kwargs = {
                "model": model,
                "instructions": system_prompt,
                "input": [{"role": "user", "content": prompt}],
                "tools": tools or None,
            }
            if self._config.reasoning_effort:
                kwargs["reasoning"] = {"effort": self._config.reasoning_effort}
            else:
                kwargs["temperature"] = 0
            print("\n" + "="*80)
            print("SYSTEM PROMPT (instructions):")
            print("-"*80)
            print(system_prompt)
            print("="*80)
            print("USER PROMPT (input):")
            print("-"*80)
            print(prompt)
            print("="*80 + "\n")
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

    def complete_cheap(self, *, system_prompt: str, prompt: str) -> str:
        """Call a cheap/fast model with JSON response format.

        Used for pre-processing steps like query rewriting. Not cached.
        Falls back to the main provider if unavailable.
        """
        import os
        provider = self._config.llm_provider
        # Gemini path — use gemini-3-flash for fast reformulation tasks
        if (provider == "gemini" or (not os.environ.get("OPENAI_API_KEY"))) and GEMINI_AVAILABLE and os.environ.get("GOOGLE_API_KEY"):
            client = genai.Client()
            model = "gemini-3-flash-preview"
            config = genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0,
                max_output_tokens=1024,
                response_mime_type="application/json",
            )
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config,
                    )
                    parts = []
                    for candidate in response.candidates or []:
                        for part in candidate.content.parts or []:
                            if hasattr(part, "text") and part.text:
                                parts.append(part.text)
                    return "\n".join(parts) if parts else (response.text or "")
                except Exception as e:
                    err_str = str(e).lower()
                    if "429" in str(e) or "rate" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                        wait = 2 ** attempt * 10
                        import logging
                        logging.getLogger(__name__).warning(f"Gemini rate limit in complete_cheap, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                        time.sleep(wait)
                    else:
                        raise
            # Final attempt
            response = client.models.generate_content(model=model, contents=prompt, config=config)
            return response.text or ""
        if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or ""
        # fallback: use main provider without JSON enforcement
        return self.complete(system_prompt=system_prompt, prompt=prompt)

    def _complete_gemini(self, *, system_prompt: str, prompt: str) -> str:
        client = genai.Client()
        model = self._config.gemini_model
        tools = []
        if self._config.enable_web_search:
            tools.append(genai_types.Tool(google_search=genai_types.GoogleSearch()))
        # code_execution disabled — suspected cause of hanging in Docker env
        # tools.append(genai_types.Tool(code_execution=genai_types.ToolCodeExecution()))

        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=tools,
            temperature=0,
        )
        if self._config.reasoning_effort:
            config.thinking_config = genai_types.ThinkingConfig(
                thinking_budget={"low": 1024, "medium": 8192, "high": 16384}.get(
                    self._config.reasoning_effort, 8192
                )
            )

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                # Extract text parts (skip thinking parts)
                parts = []
                for candidate in response.candidates or []:
                    for part in candidate.content.parts or []:
                        if hasattr(part, "text") and part.text:
                            parts.append(part.text)
                return "\n".join(parts) if parts else (response.text or "")
            except Exception as e:
                err_str = str(e).lower()
                retryable = (
                    "429" in str(e) or "rate" in err_str or "quota" in err_str
                    or "resource_exhausted" in err_str or "disconnected" in err_str
                    or "remoteprotocolerror" in err_str or "connection" in err_str
                    or "500" in str(e) or "503" in str(e) or "504" in str(e) or "unavailable" in err_str
                    or "internal" in err_str or "too long" in err_str
                    or "exceeds the maximum" in err_str
                )
                if retryable:
                    wait = min(2 ** attempt * 5, 30)
                    import logging
                    logging.getLogger(__name__).warning(f"Gemini error, retrying in {wait}s (attempt {attempt+1}/{max_retries}): {e}")
                    time.sleep(wait)
                else:
                    raise
        # Final attempt without retry
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        return response.text or ""

    def _complete_anthropic(self, *, system_prompt: str, prompt: str) -> str:
        client = anthropic.Anthropic()
        tools = []
        if self._config.enable_web_search:
            tools.append({"type": "web_search_20250305", "name": "web_search", "max_uses": 10})
        # Always enable code execution for calculation-heavy questions
        tools.append({"type": "code_execution_20250522", "name": "code_execution"})
        kwargs = {
            "model": self._config.anthropic_model,
            "max_tokens": self._config.anthropic_max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "tools": tools,
        }
        response = client.messages.create(**kwargs)
        text_parts = [block.text for block in response.content if hasattr(block, "text")]
        return "\n".join(text_parts) if text_parts else ""
