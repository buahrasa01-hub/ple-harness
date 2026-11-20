"""Offline smoke test for serve.py (no model quota, no network).

Run:  python3 test_serve_offline.py
Checks:
  1. The run-44702206886 pitfalls lines are present in SYSTEM.
  2. solve() still runs end to end with a fake model client that answers
     with a plain final message (exercises the loop + VERIFY_NUDGE path).
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("MODEL_NAME", "openai-compatible/nvidia/nemotron-3.5-super-vl:free")
# Dummy key only so openai.AsyncOpenAI() constructs at import; the client
# is replaced by a fake below and no network call is ever made.
os.environ.setdefault("OPENAI_API_KEY", "dummy-offline-test-key")

import serve
import taskbridge


def check_system():
    for needle in (
        "layer.dxf.name (not .name)",
        "entity.dxf.text (not .text)",
        "p['mark']",
    ):
        assert needle in serve.SYSTEM, f"SYSTEM missing: {needle}"
    print("SYSTEM pitfalls present: OK")


class FakeMessage:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None

    def model_dump(self, exclude_none=True):
        return {"role": "assistant", "content": self.content}


class FakeChoice:
    def __init__(self, message):
        self.message = message
        self.finish_reason = "stop"


class FakeReply:
    def __init__(self, message):
        self.choices = [FakeChoice(message)]
        self.usage = None


class FakeCompletions:
    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        # First call: deliverables exist (fail-open probe) so the loop sends
        # VERIFY_NUDGE; second call: finish.
        return FakeReply(FakeMessage("done: deliverables verified"))


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self):
        self.chat = FakeChat()


class FakeSession:
    workdir = "/tmp"
    instruction = "do a thing"  # names no deliverable paths
    final_message = ""

    def time_left(self):
        return None

    async def exec(self, command):
        raise taskbridge.TaskError("no container in offline test")

    async def log(self, text):
        pass

    async def usage(self, *args):
        pass


async def check_loop():
    serve.client = FakeClient()
    result = await serve.solve(FakeSession())
    assert result == "done: deliverables verified", f"unexpected result: {result!r}"
    assert serve.client.chat.completions.calls == 2, (
        f"expected 2 model calls (verify nudge + finish), "
        f"got {serve.client.chat.completions.calls}"
    )
    print("solve() loop with fake model (verify nudge + finish): OK")


if __name__ == "__main__":
    check_system()
    asyncio.run(check_loop())
    print("OFFLINE SMOKE TEST PASSED")
