"""Local LLM via Ollama's HTTP API (stdlib only). Nothing leaves the machine."""
import json
import urllib.error
import urllib.request
from typing import Callable

DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_HOST = "http://localhost:11434"
INSTALL_HINT = ("Install ollama from https://ollama.com, start it, then run: ollama pull {model}\n"
                "kb eval needs it only to write test questions; search itself never uses an LLM.")

Generate = Callable[[str], dict]  # prompt -> parsed JSON reply ({} if the model returned junk)


class LLMUnavailable(RuntimeError):
    pass


def _request(url: str, payload: dict | None = None, timeout: float = 120) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def ollama(model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST, seed: int = 0) -> Generate:
    """Return a generate(prompt) function, after checking ollama is running and has the model."""
    try:
        tags = _request(f"{host}/api/tags", timeout=3)
    except (urllib.error.URLError, OSError) as e:
        raise LLMUnavailable(f"Can't reach ollama at {host} ({e}).\n" + INSTALL_HINT.format(model=model)) from e
    names = {m["name"] for m in tags.get("models", [])}
    if model not in names and f"{model}:latest" not in names:
        raise LLMUnavailable(f"ollama is running but doesn't have {model!r}.\n" + INSTALL_HINT.format(model=model))

    def generate(prompt: str) -> dict:
        reply = _request(f"{host}/api/generate", {
            "model": model, "prompt": prompt, "stream": False, "format": "json",
            "options": {"temperature": 0.2, "seed": seed},
        })
        try:
            parsed = json.loads(reply.get("response", ""))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    return generate
