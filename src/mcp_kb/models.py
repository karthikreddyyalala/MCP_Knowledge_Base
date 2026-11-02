"""Model loaders, cached so each model loads once per process."""
from functools import lru_cache


@lru_cache(maxsize=None)
def get_tokenizer(model_name: str):
    """The embedding model's own tokenizer, so chunk budgets count real tokens."""
    from transformers import AutoTokenizer

    repo = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
    return AutoTokenizer.from_pretrained(repo)


@lru_cache(maxsize=None)
def get_reranker(model_name: str):
    """Cross-encoder that reads (query, chunk) together and scores how well they match."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)
