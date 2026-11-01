# What actually improves local RAG? I measured every piece

I built a small MCP server that lets Claude search my markdown notes locally. Instead of stacking every trick I'd read about, I added one piece at a time, benchmarked each on public data, and kept only what a significance test said was real. Some results surprised me, and a few of my own assumptions turned out wrong.

Code, numbers and the scripts to reproduce them: https://github.com/karthikreddyyalala/MCP_Knowledge_Base

## The setup

Three [BEIR](https://github.com/beir-cellar/beir) datasets: SciFact (5k scientific abstracts), NFCorpus (3.6k nutrition and medical articles) and FiQA (58k finance forum answers), each with human-written test questions. Every document is written to a markdown file and goes through the same indexing and search code a user runs, so the numbers measure the product, not a separate script. The metric is nDCG@10: are the right documents at the very top?

Each change is compared with the version before it using a paired randomization test over per-question scores. If p is above 0.05, I call it noise, not a win.

The first check was boring and essential: my dense-only baseline matched published numbers for `all-MiniLM-L6-v2`, and my BM25 matched published BM25 within about a point. Until that's true, nothing after it means anything.

## Finding 1: my chunks were half invisible

The first version split documents into 400-word chunks. The embedding model reads 256 tokens and silently drops the rest, so roughly half of every chunk was never searchable. No error, no warning. The tokenizer even reports a max length of 512, which is how I missed it.

Fixing it (chunks sized in real tokens with the model's own tokenizer) barely moved BEIR scores, because BEIR documents are short. It matters for long documents like runbooks. Sometimes the right fix doesn't show up on the benchmark you have.

## Finding 2: hybrid search is the big win, except where it isn't

Adding BM25 keyword search and merging it with embeddings by rank (Reciprocal Rank Fusion) gave the largest jump:

| nDCG@10 | SciFact | NFCorpus | FiQA |
|---|---|---|---|
| dense only | 0.659 | 0.321 | 0.356 |
| BM25 only | 0.685 | 0.323 | 0.241 |
| hybrid (RRF) | **0.711** | **0.349** | 0.346 |

Significant on SciFact and NFCorpus (p < 0.001). On FiQA, BM25 is weak (paraphrased finance questions share few words with the answers), and equal-weight fusion dipped slightly. I didn't tune the fusion weights on the test set to make that go away; that would be overfitting the benchmark.

## Finding 3: rerank fewer candidates

A cross-encoder reranker reads the question and each candidate together. It fixed FiQA (+0.021 over hybrid, p = 0.003). The surprise: reranking the top **20** candidates tied or beat reranking 50 (SciFact +0.012, p = 0.018) at half the latency. More candidates gave the reranker more chances to promote something wrong.

Final default against where I started: SciFact 0.652 to 0.720 (+10%), NFCorpus 0.309 to 0.364 (+18%), FiQA level.

## Finding 4: competitive with the well-known tools, and much faster

I ran [qmd](https://github.com/tobi/qmd) and [knowledge-rag](https://github.com/lyonzin/knowledge-rag) on the same files and questions, each through its own MCP server with default settings:

| | SciFact | NFCorpus | median latency |
|---|---|---|---|
| qmd (LLM query expansion + LLM rerank) | 0.729 | **0.383** | 7.1 s |
| mine | 0.720 | 0.364 | 67 ms |
| knowledge-rag | 0.718 | 0.338 | 224 ms |

On SciFact the three are a statistical tie. On NFCorpus qmd wins (+0.018, p = 0.02): it rewrites each question with an LLM before searching, which helps on short medical queries, at about 85× the latency. Mine beats knowledge-rag there (+0.027, p < 0.001). Caveat: qmd's GPU backend failed to compile on my macOS version, so its latency is likely CPU-bound here and not representative.

## Finding 5: "I don't know" is harder than it looks

I wanted search to say "nothing relevant" instead of handing Claude unrelated text. I calibrated a threshold on the reranker's score, using cross-dataset questions as negatives (finance questions asked of a science corpus): answer 97% of real questions, refuse 70% of off-topic ones.

Two things happened. First, it caught a bug in my own test set: three "answerable" example questions weren't answered anywhere in the docs, and the old evaluation had been counting them as passes. Second, while recording a demo, it refused "undo a bad deployment" even though the top result was the right runbook section. Short, terse docs score low on a reranker trained on web passages. **A threshold calibrated on one kind of text doesn't transfer to another.** The default now flags low-confidence results instead of hiding them, and a hard refusal is opt-in, calibrated on your own docs.

## Finding 6: an evaluator for your own docs, checked against human labels

Public benchmarks don't tell you how search does on *your* files, which have no labelled questions. So `kb eval` has a local LLM (via Ollama) write questions from your own chunks, plus on-topic questions your docs don't answer, filters out questions that just copy the text, and scores each search setup.

Is that trustworthy? On BEIR I can check: rank the setups with synthetic questions and with the human ones, and compare the orderings (Spearman, 1.0 = identical):

| Question writer | SciFact (prompt tuned here) | NFCorpus (held out) |
|---|---|---|
| llama3.2:3b | 0.4, and it picked the wrong best setup | |
| qwen2.5:7b, first prompt | 0.8 | 0.8 |
| qwen2.5:7b, revised prompt | 0.8 | **1.0** |

The small model was confidently wrong: its questions reused the passage's wording, which made keyword search look best. A bigger model and a prompt asking for questions "the way a colleague who hasn't read it would ask" fixed the ranking on a dataset the prompt was never tuned on. Synthetic questions still favour keyword search a little; that's in the README as a known limit.

## What I'd tell someone building this

1. Reproduce a published baseline before trusting any number of your own.
2. Run a significance test. Half my "improvements" were noise.
3. Test the path users run. The published version of this project crashed on a fresh install, and no test noticed because none started the server.
4. Check that your evaluator agrees with humans before trusting what it recommends.

Everything is local: embeddings, BM25, reranker and the evaluation LLM. No API keys. Feedback and issues welcome.
