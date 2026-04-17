import os
import hashlib
import json
import re
import time
from pathlib import Path

# Kimi sometimes emits native tool-call markup in message.content instead of
# populating the OpenAI-standard `tool_calls` field. Parse it ourselves so we
# still execute the tool and feed the result back.
_KIMI_TOOL_CALL_RE = re.compile(
    r"<\|tool_call_begin\|>\s*(\S+?)\s*<\|tool_call_argument_begin\|>\s*(.*?)\s*<\|tool_call_end\|>",
    re.DOTALL,
)
_KIMI_SECTION_WRAPPER_RE = re.compile(
    r"<\|tool_calls_section_begin\|>.*?<\|tool_calls_section_end\|>",
    re.DOTALL,
)
_KIMI_ORPHAN_MARKER_RE = re.compile(r"<\|tool_call[^|]*\|>")

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


def _sandboxed_python(code: str, namespace: dict | None = None) -> str:
    import io
    import contextlib
    import subprocess

    if namespace is None:
        namespace = {"__name__": "__main__"}

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, namespace)
        out = buf.getvalue() or "(no stdout)"
    except ModuleNotFoundError as e:
        module = e.name or str(e).split("'")[1] if "'" in str(e) else str(e)
        print(f"[AUTO-INSTALL] pip install {module}", flush=True)
        subprocess.run(["pip", "install", "-q", module], capture_output=True, timeout=60)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, namespace)
            out = buf.getvalue() or "(no stdout)"
        except Exception as e2:
            out = f"Error: {type(e2).__name__}: {e2}"
    except Exception as e:
        out = f"Error: {type(e).__name__}: {e}"
    if len(out) > 4000:
        out = out[:4000] + "\n...[truncated]"
    return out


def _tavily_search(query: str) -> str:
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return "Error: TAVILY_API_KEY not set"
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        resp = client.search(query, max_results=3)
        parts = []
        for r in resp.get("results", []):
            parts.append(f"[{r.get('title','')}]\n{r.get('content','')}")
        return "\n\n".join(parts)[:4000] if parts else "No results found"
    except Exception as e:
        return f"Search error: {e}"


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

        use_nebius = (
            OPENAI_AVAILABLE
            and os.environ.get("NEBIUS_API_KEY")
            and (provider == "nebius" or (provider == "" and os.environ.get("NEBIUS_API_KEY") and not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GOOGLE_API_KEY")))
        )
        use_gemini = (
            GEMINI_AVAILABLE
            and os.environ.get("GOOGLE_API_KEY")
            and (provider == "gemini" or (provider == "" and os.environ.get("GOOGLE_API_KEY") and not os.environ.get("OPENAI_API_KEY") and not use_nebius))
        )
        use_openai = (
            OPENAI_AVAILABLE
            and os.environ.get("OPENAI_API_KEY")
            and (provider == "openai" or (provider == "" and not os.environ.get("ANTHROPIC_API_KEY") and not use_gemini and not use_nebius))
        )
        use_anthropic = (
            ANTHROPIC_AVAILABLE
            and os.environ.get("ANTHROPIC_API_KEY")
            and (provider == "anthropic" or (provider == "" and not use_openai and not use_gemini and not use_nebius))
        )

        if use_nebius:
            response = self._complete_nebius(system_prompt=system_prompt, prompt=prompt)
            self._store_cache(cache_key, response)
            return response
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
                "nebius_model": self._config.nebius_model,
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
        if not response or not response.strip():
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
            print(f"\n[LLM] model={model} prompt_len={len(prompt)} chars")
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

    def _complete_nebius(self, *, system_prompt: str, prompt: str) -> str:
        client = OpenAI(
            api_key=os.environ["NEBIUS_API_KEY"],
            base_url=self._config.nebius_base_url,
        )
        model = self._config.nebius_model
        print(f"\n[LLM] provider=nebius model={model} prompt_len={len(prompt)} chars", flush=True)

        tools = [{
            "type": "function",
            "function": {
                "name": "execute_python",
                "description": (
                    "Execute Python code and return its stdout. Use this for ANY non-trivial "
                    "arithmetic: sums, percentages, regressions, KL divergence, rounding to N "
                    "decimal places. The sandbox has math, statistics, and json available. "
                    "Always print() the final value you want to read back."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"},
                    },
                    "required": ["code"],
                },
            },
        }]
        if os.environ.get("TAVILY_API_KEY"):
            tools.append({
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": (
                        "Search the web for factual information not in the retrieved context. "
                        "Useful for historical dates, bureau names, exchange rates, and other "
                        "reference facts."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                        },
                        "required": ["query"],
                    },
                },
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        MAX_TOOL_ITERS = 5
        MAX_WEB_SEARCHES = 2
        cur_messages = list(messages)
        sandbox_ns = {"__name__": "__main__"}
        web_search_count = 0

        for iteration in range(MAX_TOOL_ITERS + 1):
            kwargs = {
                "model": model,
                "messages": cur_messages,
                "temperature": 1.0,
                "top_p": 0.95,
                "max_tokens": 12000,
            }
            if iteration < MAX_TOOL_ITERS:
                kwargs["tools"] = tools
            response = self._nebius_call_with_retry(client, kwargs)

            choice = response.choices[0]
            msg = choice.message
            tool_calls = getattr(msg, "tool_calls", None) or []
            print(
                f"[KIMI DEBUG iter={iteration}] finish_reason={choice.finish_reason} "
                f"content_len={len(msg.content or '')} content_repr={repr((msg.content or '')[:200])} "
                f"tool_calls={len(tool_calls)} "
                f"role={getattr(msg, 'role', '?')}",
                flush=True,
            )

            if tool_calls:
                parsed_calls = [
                    (tc.id, tc.function.name, tc.function.arguments or "")
                    for tc in tool_calls
                ]
                clean_content = msg.content or ""
            elif msg.content and "<|tool_call_begin|>" in msg.content:
                parsed_calls, clean_content = self._parse_kimi_native_tool_calls(msg.content)
                if parsed_calls:
                    print(
                        f"[KIMI native-markup tool calls parsed: {len(parsed_calls)} iter={iteration}]",
                        flush=True,
                    )
            else:
                parsed_calls, clean_content = [], msg.content or ""

            if not parsed_calls:
                if not clean_content and iteration > 0:
                    print(
                        f"[KIMI empty after tool calls iter={iteration}] sending nudge",
                        flush=True,
                    )
                    cur_messages.append({"role": "assistant", "content": ""})
                    cur_messages.append({
                        "role": "user",
                        "content": "You used tools above but produced no text. "
                        "Please provide your REASONING and FINAL_ANSWER now.",
                    })
                    nudge_resp = self._nebius_call_with_retry(client, {
                        "model": model,
                        "messages": cur_messages,
                        "temperature": 1.0,
                        "top_p": 0.95,
                        "max_tokens": 12000,
                    })
                    nudge_content = nudge_resp.choices[0].message.content or ""
                    print(
                        f"[KIMI nudge response] len={len(nudge_content)} "
                        f"repr={repr(nudge_content[:200])}",
                        flush=True,
                    )
                    return nudge_content
                return clean_content

            cur_messages.append({
                "role": "assistant",
                "content": clean_content,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": args_json},
                    }
                    for call_id, name, args_json in parsed_calls
                ],
            })

            for call_id, name, args_json in parsed_calls:
                try:
                    args = json.loads(args_json or "{}")
                except Exception as parse_err:
                    result = f"Error parsing tool arguments: {parse_err}"
                else:
                    if name == "web_search":
                        query = args.get("query", "")
                        if web_search_count >= MAX_WEB_SEARCHES:
                            print(f"\n[KIMI web_search CAPPED iter={iteration}] {query}", flush=True)
                            result = "Search limit reached. Use the retrieved context and any prior search results to answer."
                        else:
                            web_search_count += 1
                            print(f"\n[KIMI web_search iter={iteration} ({web_search_count}/{MAX_WEB_SEARCHES})] {query}", flush=True)
                            result = _tavily_search(query)
                            print(f"[KIMI search result]\n{result[:400]}", flush=True)
                    else:
                        code = args.get("code", "")
                        print(f"\n[KIMI execute_python iter={iteration}]\n{code[:800]}", flush=True)
                        result = _sandboxed_python(code, namespace=sandbox_ns)
                        print(f"[KIMI tool result]\n{result[:400]}", flush=True)
                cur_messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": result,
                })
        else:
            return msg.content or ""

        return ""

    @staticmethod
    def _parse_kimi_native_tool_calls(content: str) -> tuple[list[tuple[str, str, str]], str]:
        calls: list[tuple[str, str, str]] = []
        for m in _KIMI_TOOL_CALL_RE.finditer(content):
            call_id = m.group(1).strip()
            args_json = m.group(2).strip()
            try:
                args = json.loads(args_json or "{}")
            except Exception:
                args = {}
            if "query" in args and "code" not in args:
                name = "web_search"
            else:
                name = "execute_python"
            calls.append((call_id, name, args_json))
        clean = _KIMI_SECTION_WRAPPER_RE.sub("", content)
        clean = _KIMI_ORPHAN_MARKER_RE.sub("", clean).strip()
        return calls, clean

    def _nebius_call_with_retry(self, client, kwargs):
        # Retry 429 "Model is busy" and 400 "Connection error" only.
        # Do NOT retry 504 (timeout) or context-blowout 400s (won't shrink).
        # Budget: 3 attempts, backoffs 3s/8s → +11s worst-case per call,
        # safely inside the 300s per-question judge timeout.
        backoffs = [3, 8]
        max_attempts = len(backoffs) + 1
        for attempt in range(max_attempts):
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as e:
                err_text = str(e)
                err_lower = err_text.lower()
                status = getattr(e, "status_code", None) or getattr(
                    getattr(e, "response", None), "status_code", None
                )
                # Never retry: 504, context blowouts
                if status == 504 or "504" in err_text or "gateway timeout" in err_lower:
                    self._log_nebius_error(e)
                    raise
                if "splited_prompt_len" in err_lower or "max_seq_len" in err_lower:
                    self._log_nebius_error(e)
                    raise
                # Retryable: 429 / "Model is busy" / "Connection error"
                is_busy = status == 429 or "model is busy" in err_lower or "21354" in err_text
                is_conn = "connection error" in err_lower
                if (is_busy or is_conn) and attempt < max_attempts - 1:
                    wait = backoffs[attempt]
                    import logging
                    kind = "busy" if is_busy else "conn"
                    logging.getLogger(__name__).warning(
                        f"Nebius retryable error ({kind}), retry in {wait}s (attempt {attempt+1}/{max_attempts})"
                    )
                    time.sleep(wait)
                    continue
                self._log_nebius_error(e)
                raise

    def _log_nebius_error(self, e):
        import logging, traceback
        log = logging.getLogger(__name__)
        log.error("=== NEBIUS ERROR ===")
        log.error("type=%s", type(e).__name__)
        log.error("repr=%r", e)
        log.error("str=%s", str(e))
        for attr in ("status_code", "code", "message", "body", "response", "request_id"):
            val = getattr(e, attr, None)
            if val is not None:
                log.error("e.%s = %r", attr, val)
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                log.error("response.status_code=%s", getattr(resp, "status_code", None))
                log.error("response.headers=%s", dict(getattr(resp, "headers", {}) or {}))
                log.error("response.text=%s", getattr(resp, "text", None))
            except Exception as inner:
                log.error("could not introspect response: %r", inner)
        log.error("traceback:\n%s", traceback.format_exc())
        log.error("=== END NEBIUS ERROR ===")

    def complete_cheap(self, *, system_prompt: str, prompt: str) -> str:
        """Call a cheap/fast model with JSON response format.

        Used for pre-processing steps like query rewriting. Not cached.
        Falls back to the main provider if unavailable.
        """
        import os
        provider = self._config.llm_provider
        # Nebius path — reuse the configured Kimi model for cheap calls
        if (provider == "nebius" or (provider == "" and os.environ.get("NEBIUS_API_KEY") and not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"))) and OPENAI_AVAILABLE:
            client = OpenAI(
                api_key=os.environ["NEBIUS_API_KEY"],
                base_url=self._config.nebius_base_url,
            )
            response = client.chat.completions.create(
                model=self._config.nebius_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or ""
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
        tools.append(genai_types.Tool(code_execution=genai_types.ToolCodeExecution()))

        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=tools,
            temperature=0,
        )
        if self._config.reasoning_effort:
            config.thinking_config = genai_types.ThinkingConfig(
                thinking_budget={"low": 4096, "medium": 8192, "high": 16384}.get(
                    self._config.reasoning_effort, 4096
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
                all_part_kinds = []
                for candidate in response.candidates or []:
                    for part in candidate.content.parts or []:
                        all_part_kinds.append(type(part).__name__)
                        if hasattr(part, "text") and part.text:
                            parts.append(part.text)
                result = "\n".join(parts) if parts else (response.text or "")
                if not result.strip():
                    import logging
                    logging.getLogger(__name__).warning(
                        "Gemini returned empty text. Part kinds: %s", all_part_kinds
                    )
                return result
            except Exception as e:
                err_str = str(e).lower()
                retryable = (
                    "429" in str(e) or "rate" in err_str or "quota" in err_str
                    or "resource_exhausted" in err_str or "disconnected" in err_str
                    or "remoteprotocolerror" in err_str or "connection" in err_str
                    or "500" in str(e) or "503" in str(e) or "504" in str(e) or "unavailable" in err_str
                    or "internal" in err_str or "too long" in err_str
                    or "exceeds the maximum" in err_str
                    or "timed out" in err_str or "timeout" in err_str
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
