"""Once models are cached, kb must not touch the network (the README promises offline use)."""
import os
import subprocess
import sys

PROBE = "import os, mcp_kb; print(os.environ.get('HF_HUB_OFFLINE'))"


def _probe(env_overrides: dict) -> str:
    env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    env.update(env_overrides)
    return subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True).stdout.strip()


def test_offline_when_models_are_cached():
    import mcp_kb

    assert mcp_kb.models_cached(), "test needs the models downloaded (run the other tests first)"
    assert _probe({}) == "1"


def test_online_on_first_run_when_nothing_is_cached(tmp_path):
    assert _probe({"HF_HUB_CACHE": str(tmp_path)}) == "None"  # must be allowed to download
