"""Translate the harness message and tool contracts into each provider's HTTP protocol."""

import json
import time
from contextlib import closing
from http.client import HTTPException
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4

from domain import ContextLimitError, ModelError


class _Overloaded(ModelError):
    """The provider's model was unavailable after transient retries."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # A model endpoint must never forward credentials to a redirect target.
        return None


def _blocks(message):
    content = message["content"]
    return [{"type": "text", "text": content}] if isinstance(content, str) else content


def _tools(tools, schema_field):
    return [
        {
            "name": tool["name"],
            "description": tool.get("description", ""),
            schema_field: tool.get("input_schema", tool.get("parameters", {})),
        }
        for tool in tools
    ]


def _assistant(blocks, stop):
    if not isinstance(blocks, list):
        raise ModelError("Expected model content blocks")
    if stop == "max_tokens":
        # The last call may be truncated even if the partial input parses as JSON.
        blocks = [b for b in blocks if b["type"] == "text"]
    calls = [b for b in blocks if b["type"] == "tool_use"]
    seen = set()
    for call in calls:
        if (
            not isinstance(call.get("id"), str)
            or not call["id"]
            or call["id"] in seen
            or not isinstance(call.get("name"), str)
            or not call["name"]
            or not isinstance(call.get("input"), dict)
        ):
            raise ModelError("Invalid tool call identity or arguments")
        seen.add(call["id"])
    if stop == "tool_use" and not calls:
        raise ModelError("Model requested tool use but supplied no calls")
    return {"role": "assistant", "content": blocks, "stop_reason": "tool_use" if calls else stop}


def _openai_messages(instruction, messages):
    wire = [{"role": "system", "content": instruction}] if instruction else []
    for message in messages:
        blocks = _blocks(message)
        text = "\n".join(b["text"] for b in blocks if b["type"] == "text")
        if message["role"] == "assistant":
            entry = {"role": "assistant", "content": text or None}
            calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"], "arguments": json.dumps(b["input"], ensure_ascii=False)},
                }
                for b in blocks
                if b["type"] == "tool_use"
            ]
            if calls:
                entry["tool_calls"] = calls
            wire.append(entry)
        else:
            for block in blocks:
                if block["type"] == "tool_result":
                    wire.append(
                        {"role": "tool", "tool_call_id": block["tool_use_id"], "content": block["content"]}
                    )
            if text:
                wire.append({"role": "user", "content": text})
    return wire


def _openai_reply(data):
    choice = data["choices"][0]
    message = choice["message"]
    text = message.get("content") or message.get("refusal")
    blocks = [{"type": "text", "text": text}] if text else []
    finish = choice["finish_reason"]
    if finish == "length":
        return _assistant(blocks, "max_tokens")
    if finish not in ("stop", "tool_calls"):
        raise ModelError(f"OpenAI stopped with {finish}")
    for call in message.get("tool_calls") or []:
        function = call["function"]
        blocks.append(
            {
                "type": "tool_use",
                "id": call["id"],
                "name": function["name"],
                "input": json.loads(function["arguments"]),
            }
        )
    return _assistant(blocks, "tool_use" if finish == "tool_calls" else "end_turn")


def _google_messages(messages):
    wire = []
    calls = {}
    for message in messages:
        parts = []
        for b in _blocks(message):
            kind = b["type"]
            if kind == "tool_use":
                function = {"id": b["id"], "name": b["name"], "args": b["input"]}
                part = b.get("_gemini_part", {"functionCall": function})
                calls[b["id"]] = part["functionCall"]
            elif kind == "tool_result":
                function = calls[b["tool_use_id"]]
                result = {
                    "name": function["name"],
                    "response": {"error" if b.get("is_error") else "output": b["content"]},
                }
                if "id" in function:
                    result["id"] = function["id"]
                part = {"functionResponse": result}
            elif kind == "text":
                part = {**b.get("_gemini_metadata", {}), "text": b["text"]}
            elif kind == "gemini_part":
                part = b["part"]
            else:
                continue
            parts.append(part)
        if parts:
            role = "model" if message["role"] == "assistant" else "user"
            wire.append({"role": role, "parts": parts})
    return wire


def _google_reply(data):
    candidates = data.get("candidates")
    if not candidates:
        reason = data.get("promptFeedback", {}).get("blockReason", "no candidate")
        raise ModelError(f"Gemini rejected the request ({reason})")
    candidate = candidates[0]
    finish = candidate.get("finishReason")
    if finish not in ("STOP", "MAX_TOKENS"):
        raise ModelError(f"Gemini stopped with {finish}")
    blocks = []
    for part in candidate.get("content", {}).get("parts", []):
        if "functionCall" in part:
            call = part["functionCall"]
            # Older models omit the wire ID. Keep an internal ID for linking results,
            # and retain the original signed part for verbatim protocol replay.
            block = {
                "type": "tool_use",
                "id": call.get("id") or uuid4().hex,
                "name": call["name"],
                "input": call.get("args", {}),
                "_gemini_part": part,
            }
        elif "text" in part and not part.get("thought"):
            block = {"type": "text", "text": part["text"]}
            metadata = {k: v for k, v in part.items() if k != "text"}
            if metadata:
                block["_gemini_metadata"] = metadata
        else:
            block = {"type": "gemini_part", "part": part}
        blocks.append(block)
    return _assistant(blocks, "max_tokens" if finish == "MAX_TOKENS" else "end_turn")


def _anthropic_messages(messages):
    wire = []
    for message in messages:
        blocks = []
        for block in _blocks(message):
            if block["type"] != "gemini_part":
                blocks.append({k: v for k, v in block.items() if not k.startswith("_gemini")})
        if blocks:
            wire.append({"role": message["role"], "content": blocks})
    return wire


class Model:
    def __init__(
        self, provider: str, model: str, api_key: str, effort: str = "off", fallback_model: str | None = None
    ):
        self.provider = "google" if provider == "gemini" else provider
        if self.provider not in {"deepseek", "anthropic", "openai", "google"}:
            raise ValueError(f"Unknown model provider {provider!r}")
        if effort not in {"off", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError(f"Unknown reasoning effort {effort!r}")
        if not model or not api_key:
            raise ValueError(f"Missing {provider} model ID or API key")
        self.model = model
        self.effort = effort
        self.fallback_model = fallback_model
        self._api_key = api_key

    def generate(
        self, instruction: str, messages: list[dict], tools: list[dict], max_tokens: int = 4096
    ) -> dict:
        try:
            if self.provider == "openai":
                return self._openai(instruction, messages, tools, max_tokens)
            if self.provider == "google":
                return self._google(instruction, messages, tools, max_tokens)
            return self._anthropic(instruction, messages, tools, max_tokens)
        except _Overloaded:
            if not self.fallback_model or self.fallback_model == self.model:
                raise
            alternate = Model(self.provider, self.fallback_model, self._api_key, self.effort)
            return alternate.generate(instruction, messages, tools, max_tokens)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            raise ModelError(
                f"Invalid {self.provider} message or model response ({type(exc).__name__})"
            ) from exc

    def _anthropic(self, instruction, messages, tools, max_tokens):
        deepseek = self.provider == "deepseek"
        host = "api.deepseek.com" if deepseek else "api.anthropic.com"
        path = "/anthropic/v1/messages" if deepseek else "/v1/messages"
        body = {"model": self.model, "max_tokens": max_tokens, "messages": _anthropic_messages(messages)}
        if instruction:
            body["system"] = instruction
        if tools:
            body["tools"] = _tools(tools, "input_schema")
        if self.effort == "off":
            if self.model.startswith(("claude-opus-5-5", "claude-fable", "claude-mythos")):
                # These Claude models always reason. Request their lowest effort.
                body["output_config"] = {"effort": "low"}
            elif self.model.startswith("claude-sonnet-5-5"):
                body.update(thinking={"type": "between_tools"}, output_config={"effort": "low"})
            else:
                body["thinking"] = {"type": "disabled"}
        else:
            effort = self.effort
            if deepseek:
                effort = {"medium": "high", "xhigh": "high"}.get(effort, effort)
            elif self.model.startswith(("claude-opus-4-6", "claude-sonnet-4-6")) and effort == "xhigh":
                effort = "high"
            body["thinking"] = {"type": "enabled" if deepseek else "adaptive"}
            body["output_config"] = {"effort": effort}
        headers = {"x-api-key": self._api_key, "anthropic-version": "2023-06-01"}
        data = self._post(host, path, body, headers)
        stop = data["stop_reason"]
        if stop not in {"end_turn", "tool_use", "max_tokens", "stop_sequence", "refusal"}:
            raise ModelError(f"{self.provider} stopped with {stop}")
        return _assistant(data["content"], stop if stop in ("max_tokens", "tool_use") else "end_turn")

    def _openai(self, instruction, messages, tools, max_tokens):
        body = {
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "messages": _openai_messages(instruction, messages),
        }
        if self.effort != "off" or self.model.startswith(("gpt-5", "gpt-6")):
            effort = "none" if self.effort == "off" else self.effort
            if self.model.startswith(("gpt-5.2", "gpt-5.4")) and effort == "max":
                effort = "xhigh"
            body["reasoning_effort"] = effort
        if tools:
            body["tools"] = [{"type": "function", "function": f} for f in _tools(tools, "parameters")]
        data = self._post(
            "api.openai.com", "/v1/chat/completions", body, {"Authorization": f"Bearer {self._api_key}"}
        )
        return _openai_reply(data)

    def _google(self, instruction, messages, tools, max_tokens):
        if self.model.startswith("gemini-2.5"):
            budgets = {
                "off": 128 if "pro" in self.model else 0,
                "low": 512,
                "medium": 1024,
                "high": 2048,
                "xhigh": 4096,
                "max": 8192,
            }
            thinking = {"thinkingBudget": budgets[self.effort]}
        else:
            level = {"off": "low", "xhigh": "high", "max": "high"}.get(self.effort, self.effort)
            thinking = {"thinkingLevel": level.upper()}
        body = {
            "contents": _google_messages(messages),
            "generationConfig": {"maxOutputTokens": max_tokens, "thinkingConfig": thinking},
        }
        if instruction:
            body["systemInstruction"] = {"parts": [{"text": instruction}]}
        if tools:
            body["tools"] = [{"functionDeclarations": _tools(tools, "parametersJsonSchema")}]
        path = f"/v1beta/models/{quote(self.model, safe='')}:generateContent"
        data = self._post("generativelanguage.googleapis.com", path, body, {"x-goog-api-key": self._api_key})
        return _google_reply(data)

    def _post(self, host, path, body, headers):
        payload = json.dumps(body, ensure_ascii=False).encode()
        headers = {"Content-Type": "application/json", **headers}
        opener = build_opener(_NoRedirect)
        for attempt in range(3):
            # ProxyHandler mutates the target of a Request when setting up its tunnel.
            request = Request(f"https://{host}{path}", data=payload, headers=headers)
            try:
                try:
                    response = opener.open(request, timeout=180)
                except HTTPError as exc:
                    response = exc
                with closing(response):
                    status, raw = response.status, response.read()
            except (OSError, HTTPException) as exc:
                if attempt == 2:
                    raise ModelError(f"Could not reach {self.provider} ({type(exc).__name__})") from exc
            else:
                if 200 <= status < 300:
                    return json.loads(raw)
                if status == 401:
                    raise ModelError(
                        f"{self.provider} HTTP 401：API 密钥未通过认证。请检查启动目录的 .env，更新密钥后重启 agent。"
                    )
                if status not in (408, 429, 500, 502, 503, 504, 529) or attempt == 2:
                    detail = raw.decode(errors="replace").replace(self._api_key, "[redacted]")[:1000]
                    limit_phrases = (
                        "prompt is too long",
                        "context_length_exceeded",
                        "maximum context length",
                        "input token count exceeds",
                        "exceeds the maximum number of tokens",
                    )
                    error = _Overloaded if status in (503, 529) else ModelError
                    if status == 413 or (status == 400 and any(p in detail.lower() for p in limit_phrases)):
                        error = ContextLimitError
                    raise error(f"{self.provider} HTTP {status}: {detail}")
            time.sleep(2**attempt)
