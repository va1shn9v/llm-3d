"""
LLM interaction via OpenRouter: query models with multi-view images,
extract generated code from responses.
"""

import base64
import re
import time
import logging
from typing import List, Optional, Tuple

from eval_pipeline.config import Config
from eval_pipeline.prompt import build_prompt

log = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_DELAY = 5  # seconds


# ══════════════════════════════════════════════════════════════════════════════
# PER-MODEL API PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

def get_model_params(model: str) -> dict:
    """Return model-specific API parameters for OpenRouter."""

    params = {
        "max_tokens": 8192,
        "temperature": 0.1,
        "top_p": 0.95,
        "frequency_penalty": 0.1,
        "plugins": [{"id": "response-healing"}],
    }

    thinking_models = {
        "qwen/qwen3-vl-8b-thinking",
        "qwen/qwen3-vl-30b-a3b-thinking",
    }

    reasoning_models = {
        "x-ai/grok-4.1-fast",
        "openai/gpt-5.2-codex",
    }

    anthropic_models = {
        "anthropic/claude-sonnet-4.6",
    }

    if model in thinking_models:
        params["max_tokens"] = 16384
        params["reasoning"] = {
            "effort": "medium",
            "exclude": False,
        }
    elif model in reasoning_models:
        params["max_tokens"] = 12288
        params["reasoning"] = {
            "effort": "low",
        }
    elif model in anthropic_models:
        params["max_tokens"] = 12288
        params["reasoning"] = {
            "effort": "low",
        }

    return params


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: QUERY LLMs VIA OPENROUTER
# ══════════════════════════════════════════════════════════════════════════════

def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_llm(
    model: str,
    image_paths: List[str],
    config: Config,
) -> Tuple[str, Optional[str], float]:
    """
    Send multi-view images to an LLM via OpenRouter and ask it to generate
    Blender Python code.

    Includes retry logic with exponential backoff for transient failures
    (rate limits, timeouts).

    Returns:
        (model_name, generated_code_or_None, response_time_seconds)
    """
    import requests

    prompt_text = build_prompt(config.num_views)

    # ── Build the message content with images ────────────────────────────
    content = []

    content.append({
        "type": "text",
        "text": prompt_text,
    })

    for i, img_path in enumerate(image_paths):
        b64 = encode_image_to_base64(img_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}",
            },
        })
        content.append({
            "type": "text",
            "text": f"[View {i+1} of {len(image_paths)}]",
        })

    # ── Build headers and payload ────────────────────────────────────────
    headers = {
        "Authorization": f"Bearer {config.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yourusername/image-to-3d-rlvr",
        "X-Title": "Image-to-3D Code Eval",
    }

    model_params = get_model_params(model)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        **model_params,
    }

    log.info(f"  Querying {model}...")
    t_start = time.time()

    # ── Retry loop ───────────────────────────────────────────────────────
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=300,
            )

            if response.status_code == 429:
                if attempt < MAX_RETRIES:
                    wait = RETRY_DELAY * (2 ** attempt)
                    log.warning(f"  Rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    elapsed = time.time() - t_start
                    log.warning(f"  {model} rate limited after {MAX_RETRIES + 1} attempts")
                    return model, None, elapsed

            if response.status_code != 200:
                elapsed = time.time() - t_start
                log.warning(
                    f"  {model} returned HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return model, None, elapsed

            # Success path
            elapsed = time.time() - t_start
            data = response.json()
            reply = data["choices"][0]["message"]["content"]

            code = extract_code_from_response(reply)

            if code:
                log.info(f"  {model} returned {len(code)} chars of code ({elapsed:.1f}s)")
            else:
                log.warning(f"  {model} returned no extractable code ({elapsed:.1f}s)")

            return model, code, elapsed

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                log.warning(f"  Timeout, retrying (attempt {attempt + 1})...")
                continue
            elapsed = time.time() - t_start
            log.warning(f"  {model} timed out after {elapsed:.1f}s")
            return model, None, elapsed
        except Exception as e:
            elapsed = time.time() - t_start
            log.warning(f"  {model} error: {e}")
            return model, None, elapsed

    elapsed = time.time() - t_start
    return model, None, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# CODE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_code_from_response(response_text: str) -> Optional[str]:
    """
    Extract Python code from an LLM response.
    Handles various formats: markdown fences, thinking model wrappers,
    raw code output, etc.
    """
    text = response_text.strip()

    # Strategy 1: ```python / ```Python / ```py fences — take the longest match
    pattern = r'```(?:[Pp]ython3?|[Pp]y)\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # Strategy 2: any ``` block containing 'import bpy'
    pattern = r'```\w*\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        if 'import bpy' in match or 'bpy.ops' in match:
            return match.strip()

    # Strategy 3: thinking model wrappers (<answer>, <output>, <code>, etc.)
    for tag in ['answer', 'output', 'code', 'result', 'solution']:
        tag_pattern = rf'<{tag}>(.*?)</{tag}>'
        tag_matches = re.findall(tag_pattern, text, re.DOTALL)
        for match in tag_matches:
            if 'import bpy' in match or 'bpy.ops' in match:
                cleaned = re.sub(r'```\w*\n?', '', match).strip()
                return cleaned

    # Strategy 4: raw code starting with 'import bpy'
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('import bpy'):
            code_lines = []
            for cl in lines[i:]:
                if cl.strip().startswith('```'):
                    break
                code_lines.append(cl)
            return '\n'.join(code_lines).strip()

    # Strategy 5: whole response looks like valid code
    if ('import bpy' in text or 'bpy.ops' in text) and 'GENERATION_COMPLETE' in text:
        cleaned = re.sub(r'^```\w*\n?', '', text)
        cleaned = re.sub(r'\n?```$', '', cleaned)
        return cleaned.strip()

    return None
