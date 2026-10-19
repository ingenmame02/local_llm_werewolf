import json
import os
import re
import socket
import subprocess
from dataclasses import dataclass
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class OllamaError(Exception):
    kind: str
    message: str
    attempts: int
    status_code: int | None = None

    def __str__(self) -> str:
        status = f" (HTTP {self.status_code})" if self.status_code is not None else ""
        return f"Ollama {self.kind} error after {self.attempts} attempt(s){status}: {self.message}"

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "kind": self.kind,
            "message": self.message,
            "attempts": self.attempts,
            "status_code": self.status_code,
        }


Transport = Callable[[Request, float], bytes]


def _urlopen_transport(request: Request, timeout: float) -> bytes:
    with urlopen(request, timeout=timeout) as response:
        return response.read()


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, object]
    call_id: str | None = None


@dataclass(frozen=True)
class ChatResponse:
    content: str
    attempts: int
    done_reason: str | None = None
    content_length: int | None = None
    thinking_present: bool = False
    tool_calls: tuple[ToolCall, ...] = ()

    def to_metadata(self) -> dict[str, str | int | bool | None]:
        return {
            "done_reason": self.done_reason,
            "content_length": (
                self.content_length
                if self.content_length is not None
                else len(self.content)
            ),
            "thinking_present": self.thinking_present,
        }


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        max_retries: int = 2,
        transport: Transport | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("Timeout must be greater than zero.")
        if not 0 <= max_retries <= 2:
            raise ValueError("At most 2 retries are supported.")
        if not base_url.strip():
            raise ValueError("Ollama base URL cannot be empty.")

        self.endpoint = f"{base_url.rstrip('/')}/api/chat"
        self.tags_endpoint = f"{base_url.rstrip('/')}/api/tags"
        self.timeout = timeout
        self.max_retries = max_retries
        self._transport = transport or _urlopen_transport

    def ping(self) -> bool:
        request = Request(self.tags_endpoint, headers={"Content-Type": "application/json"}, method="GET")
        try:
            self._transport(request, min(self.timeout, 5.0))
            return True
        except Exception:
            return False

    def chat_with_metadata(
        self,
        model: str,
        messages: list[dict[str, object]],
        options: dict[str, object] | None = None,
        tools: list[dict[str, object]] | None = None,
    ) -> ChatResponse:
        content = self.chat(
            model=model, messages=messages, options=options, tools=tools
        )
        return getattr(
            self,
            "_last_chat_response",
            ChatResponse(
                content=content,
                attempts=getattr(self, "_last_attempts", 1),
                content_length=len(content),
            ),
        )

    def chat(
        self,
        model: str,
        messages: list[dict[str, object]],
        options: dict[str, object] | None = None,
        tools: list[dict[str, object]] | None = None,
    ) -> str:
        if not model.strip():
            raise ValueError("Model name cannot be empty.")
        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if options:
            payload["options"] = options
        if tools:
            payload["tools"] = tools
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        for attempt in range(1, self.max_retries + 2):
            self._last_attempts = attempt
            try:
                response_body = self._transport(request, self.timeout)
                response = json.loads(response_body)
                if not isinstance(response, dict):
                    raise ValueError("The response body is not a JSON object.")
                message = response["message"]
                if not isinstance(message, dict):
                    raise ValueError("The response message is not a JSON object.")
                content = message["content"]
                if not isinstance(content, str):
                    raise ValueError("The response message content is not a string.")
                raw_tool_calls = message.get("tool_calls", [])
                if not isinstance(raw_tool_calls, list):
                    raise ValueError("The response message tool_calls is not an array.")
                tool_calls: list[ToolCall] = []
                for raw_tool_call in raw_tool_calls:
                    if not isinstance(raw_tool_call, dict):
                        raise ValueError("A response tool call is not an object.")
                    function = raw_tool_call.get("function")
                    if not isinstance(function, dict):
                        raise ValueError("A response tool call has no function object.")
                    name = function.get("name")
                    arguments = function.get("arguments")
                    if not isinstance(name, str) or not isinstance(arguments, dict):
                        raise ValueError("A response tool call has invalid function data.")
                    call_id = raw_tool_call.get("id")
                    tool_calls.append(
                        ToolCall(
                            name=name,
                            arguments=arguments,
                            call_id=call_id if isinstance(call_id, str) else None,
                        )
                    )
                done_reason = response.get("done_reason")
                thinking = message.get("thinking")
                self._last_chat_response = ChatResponse(
                    content=content,
                    attempts=attempt,
                    done_reason=done_reason if isinstance(done_reason, str) else None,
                    content_length=len(content),
                    thinking_present=bool(thinking) or "<think" in content.casefold(),
                    tool_calls=tuple(tool_calls),
                )
                return content
            except HTTPError as error:
                error_body = error.read().decode("utf-8", errors="replace")[:500]
                last_error = OllamaError(
                    kind="http",
                    message=error_body or str(error.reason),
                    attempts=attempt,
                    status_code=error.code,
                )
            except (TimeoutError, socket.timeout) as error:
                last_error = OllamaError(
                    kind="timeout", message=str(error) or "Request timed out.", attempts=attempt
                )
            except URLError as error:
                reason = error.reason
                timed_out = isinstance(reason, (TimeoutError, socket.timeout))
                last_error = OllamaError(
                    kind="timeout" if timed_out else "connection",
                    message=str(reason),
                    attempts=attempt,
                )
            except OSError as error:
                last_error = OllamaError(
                    kind="connection", message=str(error), attempts=attempt
                )
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValueError) as error:
                last_error = OllamaError(
                    kind="invalid_response", message=str(error), attempts=attempt
                )

            if attempt > self.max_retries:
                raise last_error

        raise RuntimeError("Ollama retry loop exited unexpectedly.")

    @staticmethod
    def parse_ollama_ps_context(output: str, model: str) -> int | None:
        raw_lines = [line.rstrip() for line in output.splitlines() if line.strip()]
        lines = [
            re.split(r"\s{2,}", line.strip()) for line in raw_lines
        ]
        if not lines:
            return None
        header = lines[0]
        try:
            context_index = header.index("CONTEXT")
        except ValueError:
            return None
        for line_index, columns in enumerate(lines[1:], start=1):
            if columns and (
                columns[0] == model
                or columns[0].startswith(f"{model}:")
                or model.startswith(f"{columns[0]}:")
            ):
                if context_index >= len(columns):
                    fallback_columns = raw_lines[line_index].split()
                    if context_index >= len(fallback_columns):
                        return None
                    columns = fallback_columns
                try:
                    return int(columns[context_index].replace(",", ""))
                except ValueError:
                    return None
        return None

    def context_diagnostics(self, model: str) -> dict[str, object]:
        process_env_context = os.environ.get("OLLAMA_CONTEXT_LENGTH")
        diagnostics: dict[str, object] = {
            "process_ollama_context_length_env": process_env_context,
            "ollama_ps_context": None,
            "ollama_ps_error": None,
        }
        try:
            result = subprocess.run(
                ["ollama", "ps"],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
        except FileNotFoundError:
            diagnostics["ollama_ps_error"] = "ollama executable not found"
            return diagnostics
        except subprocess.TimeoutExpired:
            diagnostics["ollama_ps_error"] = "ollama ps timed out"
            return diagnostics
        except OSError as error:
            diagnostics["ollama_ps_error"] = str(error)
            return diagnostics

        if result.returncode != 0:
            diagnostics["ollama_ps_error"] = (
                result.stderr.strip() or f"ollama ps exited with {result.returncode}"
            )
            return diagnostics
        diagnostics["ollama_ps_context"] = self.parse_ollama_ps_context(
            result.stdout, model
        )
        return diagnostics


CLAIM_ROLE_TOOL: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "claim_role",
        "description": (
            "公開の昼議論で自分が主張する役職と、占い師COなら公開する占い結果を記録します。"
            "本人の実際の役職は公開しません。"
        ),
        "parameters": {
            "type": "object",
            "required": ["role"],
            "properties": {
                "role": {
                    "type": "string",
                    "enum": ["seer", "wolf", "madman", "villager"],
                },
                "target": {"type": "string"},
                "result": {
                    "type": "string",
                    "enum": ["wolf", "human"],
                },
                "reason": {"type": "string"},
                "vote_proposal": {"type": "string"},
            },
        },
    },
}
