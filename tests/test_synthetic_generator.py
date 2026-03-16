import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.synthetic_generator import SyntheticGenerator, _resolve_teacher_api


def _make_generator(**overrides):
    gen = SyntheticGenerator.__new__(SyntheticGenerator)
    config = {
        "teacher_model": "gpt-5.3-codex",
        "teacher_api": "auto",
        "samples_per_caption": 2,
        "temperature": 0.7,
        "teacher_reasoning_effort": "high",
    }
    config.update(overrides)
    gen.gen_cfg = SimpleNamespace(**config)
    return gen


class _FakeChatCompletions:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.calls = 0

    async def create(self, **_kwargs):
        self.calls += 1
        if self._error:
            raise self._error
        return self._result


class _FakeResponses:
    def __init__(self, output_text, error_on_temperature=False):
        self._output_text = output_text
        self._error_on_temperature = error_on_temperature
        self.calls = 0
        self.request_kwargs = []

    async def create(self, **kwargs):
        self.calls += 1
        self.request_kwargs.append(kwargs.copy())
        if self._error_on_temperature and "temperature" in kwargs:
            raise Exception("Unsupported parameter: 'temperature' is not supported with this model.")
        return SimpleNamespace(output_text=self._output_text)


class _FakeClient:
    def __init__(
        self,
        *,
        chat_result=None,
        chat_error=None,
        response_text="",
        response_error_on_temperature=False,
    ):
        self.chat = SimpleNamespace(
            completions=_FakeChatCompletions(result=chat_result, error=chat_error),
        )
        self.responses = _FakeResponses(
            response_text,
            error_on_temperature=response_error_on_temperature,
        )


def test_resolve_teacher_api_auto_picks_expected_endpoint():
    assert _resolve_teacher_api(SimpleNamespace(teacher_model="gpt-5.3-codex", teacher_api="auto")) == "responses"
    assert _resolve_teacher_api(SimpleNamespace(teacher_model="gpt-4o-mini", teacher_api="auto")) == "chat"
    assert _resolve_teacher_api(SimpleNamespace(teacher_model="gpt-4o-mini", teacher_api="responses")) == "responses"


def test_codex_models_use_responses_api():
    gen = _make_generator()
    client = _FakeClient(response_text="import bpy\nprint('ok')\n")

    codes = asyncio.run(gen._call_teacher_batch(client, "test object"))

    assert codes == ["import bpy\nprint('ok')", "import bpy\nprint('ok')"]
    assert client.responses.calls == gen.gen_cfg.samples_per_caption
    assert client.chat.completions.calls == 0
    assert all("temperature" not in kwargs for kwargs in client.responses.request_kwargs)


def test_chat_endpoint_mismatch_falls_back_to_responses():
    gen = _make_generator(teacher_model="gpt-4o-mini", teacher_api="auto")
    client = _FakeClient(
        chat_error=Exception(
            "This is not a chat model and thus not supported in the v1/chat/completions endpoint.",
        ),
        response_text="import bpy\nprint('fallback')\n",
    )

    codes = asyncio.run(gen._call_teacher_batch(client, "fallback object"))

    assert codes == ["import bpy\nprint('fallback')", "import bpy\nprint('fallback')"]
    assert client.chat.completions.calls == 1
    assert client.responses.calls == gen.gen_cfg.samples_per_caption


def test_responses_temperature_error_retries_without_temperature():
    gen = _make_generator(teacher_model="gpt-4o-mini", teacher_api="responses")
    client = _FakeClient(
        response_text="import bpy\nprint('retry')\n",
        response_error_on_temperature=True,
    )

    codes = asyncio.run(gen._call_teacher_batch(client, "retry object"))

    assert codes == ["import bpy\nprint('retry')", "import bpy\nprint('retry')"]
    assert client.chat.completions.calls == 0
    assert client.responses.calls == gen.gen_cfg.samples_per_caption * 2
    assert "temperature" in client.responses.request_kwargs[0]
    assert "temperature" not in client.responses.request_kwargs[1]
