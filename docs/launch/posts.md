# Launch posts (drafts to edit and post yourself)

Post the blog first (dev.to, Medium, Hashnode or GitHub Pages), then link it from these.

## Hacker News

**Title** (Show HN, under 80 characters):

> Show HN: Local RAG for Claude, every component benchmarked with significance tests

**First comment** (HN expects the author to explain):

> I built an MCP server that lets Claude search my markdown notes locally (hybrid BM25 + embeddings, cross-encoder rerank, line citations). The part I'm happiest with is the measurement: every component is benchmarked on BEIR with paired significance tests, it's compared head-to-head with qmd and knowledge-rag on the same files, and `kb eval` has a local LLM write test questions from your own docs. I checked that evaluator against human-labelled BEIR and it ranked setups the same way on a held-out dataset. Some findings: my chunks were half invisible to the embedding model, reranking 20 candidates beat 50, and a "not found" threshold calibrated on one kind of text refused correct answers on another. Happy to answer questions; losses and limits are in the README.

## LinkedIn

> I rebuilt my local RAG project for Claude around one rule: measure everything, keep only what a significance test says is real.
>
> What I found:
> - My chunks were half invisible. The embedding model reads 256 tokens and silently drops the rest; I was feeding it ~520.
> - Hybrid search (keywords + embeddings) was the biggest win: +10% on SciFact and +18% on NFCorpus over where I started.
> - Reranking the top 20 candidates beat reranking 50, at half the latency.
> - Head-to-head on the same benchmark: tied with qmd on SciFact, behind it on NFCorpus (its LLM query rewriting helps there), ahead of knowledge-rag, and 67 ms per search vs 0.2 to 7 seconds.
> - A "not found" threshold tuned on one kind of text refused correct answers on another, so it now warns instead of hiding.
> - I built an evaluator that writes test questions from your own docs with a local LLM, then checked it against human-labelled data. A 3B model gave confidently wrong recommendations; a 7B model with a better prompt matched the human ranking exactly on a dataset it was never tuned on.
>
> Everything runs locally. Code, numbers and how to reproduce them: [repo link]
> Full write-up: [blog link]

## r/LocalLLaMA

**Title:**

> Benchmarked every piece of a local RAG stack on BEIR (hybrid, rerank, chunking) + a local-LLM evaluator validated against human labels

**Body:** the blog post's findings 1 to 6, with the tables, and a link to the repo. This community cares most about: the Ollama-based evaluator, 3B vs 7B results, and the latency numbers.
