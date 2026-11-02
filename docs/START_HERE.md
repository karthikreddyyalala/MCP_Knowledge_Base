# Start here

The whole project in one page: what it is, what was done and why, where everything lives, and how to pick it back up. Read this first; the other docs go deeper.

| If you want to... | Read |
|---|---|
| Understand the project and get back on track | this page |
| Learn every concept, file and command | [HOW_IT_WORKS.md](HOW_IT_WORKS.md) |
| See what each phase built, measured and got wrong | [BUILD_LOG.md](BUILD_LOG.md) |
| See the original goals and rules | [the spec](superpowers/specs/2026-10-31-kb-measured-retrieval-design.md) |
| See raw numbers | [results/summary.md](../results/summary.md) |

---

## 1. What this is, in plain English

Claude can't see your private notes. This project is a small program that runs **on your laptop** and lets Claude **search your own markdown files** and answer with the exact file and line it used, without uploading your files anywhere.

Think of it as a librarian for your notes:
- **`kb index`** reads your notes and files every paragraph in two catalogues: one by *meaning*, one by *exact words*.
- When Claude asks a question, **`kb serve`** checks both catalogues, merges the best matches, has a careful second reader re-check the top 20, and hands Claude the best 5 with citations. If nothing scores as clearly relevant, it adds a **"Low confidence"** note so Claude doesn't treat weak matches as answers.
- **`kb eval`** is the part nobody else has: it writes test questions *from your own notes* with a local AI model and tells you how well search works on *your* files.

It connects to Claude through **MCP** (Model Context Protocol), the standard way to give Claude tools.

---

## 2. Why it was rebuilt

The original version (v1) worked, but a review found problems: about half of every paragraph was invisible to the search model, a fresh install crashed, and the headline benchmark number ("MRR 0.585, 3.5x BM25") wasn't measured fairly. Research also showed that plain "local search for Claude" already exists (tobi/qmd, knowledge-rag), so features alone wouldn't stand out.

**The chosen angle: measure everything honestly.** Every change is benchmarked on public datasets, every gain comes with a significance test, losses are reported, and the tool can measure itself on your own docs. That rigor is what recruiters and interviewers notice.

---

## 3. How it works (the short version)

```
YOUR NOTES (.md / .txt)
   │  kb index   (only files whose content changed)
   ▼
cut into chunks of <= 256 tokens, each labelled "file > heading"
   ├─► meaning catalogue  (embeddings in ChromaDB)
   └─► word catalogue     (BM25 in SQLite)

QUESTION from Claude
   │  kb serve
   ▼
top 50 by meaning + top 50 by words ─► merge by rank (RRF)
   ─► careful re-check of the top 20 (cross-encoder reranker)
   ─► best 5 chunks + [source: file.md:L8-L14]
   ─► best score too low?  add a "Low confidence" warning
       (hard "not found" only if you calibrate it on your docs with kb eval)
```

Every word in that diagram is explained in [HOW_IT_WORKS.md](HOW_IT_WORKS.md#2-concepts-in-the-order-you-meet-them).

---

## 4. What was done, phase by phase

| # | Phase | What it means | Key result |
|---|---|---|---|
| 0 | Cleanup | Real `kb` command, fixed the startup crash, removed junk | First search 2.3 s → 11 ms |
| 1 | Measure first | Fair benchmark on 3 public datasets (BEIR) | Baseline matches published numbers |
| 2 | Token-sized chunks | Chunks now fit what the model actually reads | A correctness fix; small effect on BEIR |
| 3 | Hybrid search | Added word search (BM25) and merged it with meaning search | SciFact +0.051, NFCorpus +0.028 (both real) |
| 4 | Reranker + "not found" | Careful re-check of the top 20; refuse when nothing fits | Fixed FiQA; caught a bug in v1's own test set |
| 5 | Incremental sync | Re-index only edited files; Claude picks up edits on restart | Second index run does no work |
| 6 | `kb eval` | Test search on your own docs with a local AI model | Validated against human labels (below) |
| 7 | Head-to-head | qmd and knowledge-rag on the same benchmark | Tie on SciFact; qmd ahead on NFCorpus; kb ahead of knowledge-rag; 67 ms vs 0.2 s vs 5 to 7 s |
| 8 | Ship | Packaging, offline fix, demo GIF, README, site, launch drafts | Built; waiting on your decisions |
| 9 | Query expansion | Optional LLM rewriting of questions (qmd's trick), `--expand` | No significant gain, ~25x slower: kept opt-in |

### Headline numbers (nDCG@10: 1.0 is perfect ranking)

| | SciFact | NFCorpus | FiQA |
|---|---|---|---|
| v1 (start) | 0.652 | 0.309 | 0.362 |
| **now** | **0.720** (+10%) | **0.364** (+18%) | **0.367** (level) |

### Is `kb eval` trustworthy?

`kb eval` writes its own test questions, so it's checked against datasets where humans wrote the questions: does it rank the four search setups in the same order humans' questions do? (Spearman 1.0 = identical order.)

| | SciFact (prompt tuned here) | NFCorpus (never tuned on) |
|---|---|---|
| small model, llama3.2:3b, prompt v1 | 0.4, picked the wrong winner | not run |
| qwen2.5:7b, prompt v1 | 0.8 | 0.8 |
| **qwen2.5:7b, prompt v2 (default)** | **0.8** | **1.0** |

Known limits, stated in the README: synthetic questions still slightly favour keyword search, and "not found" catches off-topic questions much better than near-misses.

---

## 5. The stories worth telling in an interview

1. **The published version crashed on a fresh install** and no test noticed, because no test started the server. Now a test talks to the server exactly like Claude does.
2. **Half of every chunk was invisible.** Chunks were 400 words; the model reads 256 tokens and silently drops the rest. The tokenizer even reports 512, which is the trap.
3. **Hybrid search, and why.** Meaning search finds paraphrases; word search finds exact names and codes. Merging by rank (RRF) beats either.
4. **The "not found" feature found a bug in the test set.** Three "answerable" example questions are answered nowhere in the docs; v1 had been counting them as passes.
5. **A demo caught a false refusal.** Recording the demo showed the "not found" threshold hiding a correct answer on short runbook text. Thresholds don't transfer between kinds of text, so the default became a warning.
6. **Correcting my own claims.** Two things I said turned out wrong when measured (model reloading; a BM25 example). Both are corrected in the build log.
7. **Validating the evaluator.** A small model gave confident but wrong recommendations; a 7B model with a better prompt matched human labels exactly on a dataset it was never tuned on.

---

## 6. Where everything lives

| What | Where |
|---|---|
| Code | `src/mcp_kb/` (product), `src/mcp_kb/evals/` (measurement) |
| Tests (65+) | `tests/` |
| Benchmark results (committed) | `results/` |
| Docs | `docs/` (this page, HOW_IT_WORKS, BUILD_LOG, spec, phase plans) |
| Branch | `v2-measured-retrieval` on GitHub (`main` is still v1) |
| Python environment | `.venv/` in the repo (uv, Python 3.12). The system Python's SciPy is broken on this Mac, so always use `.venv/bin/...` |
| Your default index | `~/.mcp-kb/chroma` |
| Benchmark downloads and indexes | `~/.mcp-kb/bench/` (safe to delete; rebuilt on demand) |
| Local AI models | Ollama (installed with Homebrew): `qwen2.5:7b`, `llama3.2:3b`. Remove with `brew uninstall ollama` |
| Comparison tools (qmd, knowledge-rag) | Installed in a temporary session folder. To reinstall, see below |

---

## 7. How to get back on track

```bash
cd /Users/jaipal/claude-code/mcp-knowledge-base
git switch v2-measured-retrieval && git pull
uv pip install -e ".[dev]"                   # if .venv is missing: uv venv --python 3.12 first
.venv/bin/python -m pytest tests -q          # everything should pass

# try it
.venv/bin/kb index ./example-docs
.venv/bin/kb search "how do I roll back a deployment"
.venv/bin/kb search "best pizza in naples"   # -> results with a "Low confidence" warning

# measure it on your own notes (needs Ollama running: `ollama serve`)
.venv/bin/kb eval ~/path/to/your/notes

# connect to Claude Code
claude mcp add kb -- $PWD/.venv/bin/kb serve --docs ~/path/to/your/notes
```

Reinstall the comparison tools (only needed to rerun phase 7):

```bash
mkdir -p ~/kb-tools/qmd && npm install --prefix ~/kb-tools/qmd @tobilu/qmd@2.8.3
uv venv --python 3.12 ~/kb-tools/krag && uv pip install --python ~/kb-tools/krag/bin/python knowledge-rag==4.11.1
.venv/bin/kb bench --dataset scifact --setup qmd --tool-bin ~/kb-tools/qmd/node_modules/.bin/qmd
.venv/bin/kb bench --dataset scifact --setup knowledge-rag --tool-bin ~/kb-tools/krag/bin/knowledge-rag
```

Then open [BUILD_LOG.md](BUILD_LOG.md#remaining-phases-from-the-spec) for the next unfinished phase.

---

## 8. What's left, and what needs you

| Item | Who |
|---|---|
| ~~Pick a PyPI name~~ | Done: `citekb` (commands `kb` and `citekb`) |
| ~~Choose a license~~ | Done: MIT |
| Create a PyPI account, add a trusted publisher for `release.yml` (steps at the top of that file), then `git tag v0.2.0 && git push --tags` | **You** |
| Look at the updated site (`docs/index.html`) in a browser | **You** |
| Redeploy the live site: it's a separate Vercel project (`mcp-kb-site`), not connected to GitHub, so merging does **not** update it. Run `npx vercel login`, then `cd docs && npx vercel --prod` and pick the `mcp-kb-site` project | **You** |
| ~~Fix the GitHub-connected Vercel project `mcp-base`~~ | Done: it treated the repo as a Python app (`PYTHON_ENTRYPOINT_NOT_FOUND`); `vercel.json` now serves `docs/` as static files and the preview builds. After merging, its production URL also shows the new page |
| Post the blog: drafts in `docs/launch/` (blog post, HN, LinkedIn, Reddit) | **You** |
| Update the portfolio card (link `MCP_Knowledge_Base`; replace the MS MARCO claim with the numbers above) | **You** (that site's code isn't in this repo) |
| Walk through the code with `/vibe-wise:learn` so you can explain every part | **You** |
| Merge `v2-measured-retrieval` into `main` when happy | **You** decide |
