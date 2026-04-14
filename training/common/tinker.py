"""
Shared Tinker utilities for RL and evaluation.
"""

from __future__ import annotations

from typing import Any

try:
    import tinker
    from tinker import types as tinker_types
except ImportError:
    tinker = None  # type: ignore[assignment]
    tinker_types = None  # type: ignore[assignment]


def build_tinker_service_client() -> Any:
    """Create a Tinker service client or raise a clear error."""
    if tinker is None:
        raise ImportError(
            "tinker is not installed. Install the training extras before running RL or eval: "
            'pip install -e ".[training]"'
        )

    service_ctor = getattr(tinker, "ServiceClient", None)
    if service_ctor is None:
        raise AttributeError("tinker.ServiceClient is unavailable in the installed SDK")

    try:
        return service_ctor()
    except TypeError:
        api_key = getattr(tinker, "api_key", None)
        if api_key is not None:
            return service_ctor(api_key=api_key)
        raise


def require_tinker_types() -> Any:
    if tinker_types is None:
        raise ImportError(
            "tinker is not installed. Install the training extras before running RL or eval: "
            'pip install -e ".[training]"'
        )
    return tinker_types


def render_prompt_to_model_input(messages: list[dict[str, str]], tokenizer: Any) -> Any:
    """Convert chat-format messages into a Tinker ModelInput."""
    types = require_tinker_types()
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role != "assistant":
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    parts.append("<|im_start|>assistant\n")
    prompt_text = "".join(parts)
    tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    return types.ModelInput.from_ints(tokens=tokens)


def clean_generated_code(text: str) -> str:
    """Trim common chat artifacts so Modal receives raw Python."""
    cleaned = text.strip()
    for sentinel in ("<|im_end|>", "<|endoftext|>"):
        if sentinel in cleaned:
            cleaned = cleaned.split(sentinel, 1)[0].rstrip()

    if "```python" in cleaned:
        start = cleaned.index("```python") + len("```python")
        end = cleaned.find("```", start)
        cleaned = cleaned[start:end if end != -1 else None].strip()
    elif cleaned.startswith("```"):
        start = cleaned.find("\n")
        if start != -1:
            cleaned = cleaned[start + 1 :]
        end = cleaned.find("```")
        cleaned = cleaned[:end if end != -1 else None].strip()

    return cleaned
