"""mcp_kb: local search over your docs for Claude.

Runs before any submodule import, so it can configure the Hugging Face and torch libraries
before they load: stay offline once the models are downloaded, and keep their logs quiet.
"""
import logging
import os
from pathlib import Path

# Models kb loads from the Hugging Face Hub (repo ids). Keep in sync with config.Settings defaults.
MODEL_REPOS = ("sentence-transformers/all-MiniLM-L6-v2", "cross-encoder/ms-marco-MiniLM-L-6-v2")


def _hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def models_cached() -> bool:
    """True if every model kb needs is already in the local Hugging Face cache."""
    cache = _hub_cache()
    return all(any((cache / f"models--{repo.replace('/', '--')}" / "snapshots").glob("*/config.json"))
               for repo in MODEL_REPOS)


# Offline once downloaded: no network calls (or "unauthenticated requests" warnings) on every start.
# The first run, before anything is cached, still downloads normally.
if models_cached():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
os.environ.setdefault("HF_HUB_VERBOSITY", "error")  # hides the "using cached commit hash" offline notice
