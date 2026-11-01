"""Model loaders, cached so each model loads once per process."""
from functools import lru_cache


@lru_cache(maxsize=None)
def get_tokenizer(model_name: str):
    """The embedding model's own tokenizer, so chunk budgets count real tokens."""
    from transformers import AutoTokenizer

    repo = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
    return AutoTokenizer.from_pretrained(repo)
