import json
import socket
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
        self.timeout = timeout
        self.max_retries = max_retries
        self._transport = transport or _urlopen_transport

    def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        if not model.strip():
            raise ValueError("Model name cannot be empty.")
        body = json.dumps(
            {"model": model, "messages": messages, "stream": False}
        ).encode("utf-8")
        request = Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        for attempt in range(1, self.max_retries + 2):
            try:
                response_body = self._transport(request, self.timeout)
                response = json.loads(response_body)
                content = response["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("The response message content is not a string.")
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
