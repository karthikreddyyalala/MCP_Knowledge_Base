"""Generate a test set from the user's own docs with a local LLM.

For each sampled chunk the LLM writes one question the chunk answers and one on-topic question
it does not. Three filters keep the set honest:
  - word overlap: answerable questions that mostly copy the chunk's words are dropped, because
    they would make keyword search look better than it is
  - answer check: the LLM must confirm the chunk really answers its question
  - elsewhere check: an "unanswerable" question is dropped if any of the top 3 search results
    answers it, because the docs answer it after all
"""
import dataclasses
import json
import random
import re
from pathlib import Path

from mcp_kb.config import Settings
from mcp_kb.llm import Generate
from mcp_kb.searcher import _get_collection, search

PROMPT_VERSION = 2  # 1: "use your own words"; 2: "phrase it like someone who has not read it"
MIN_WORDS = 12  # chunks shorter than this are mostly headings or fragments
MAX_OVERLAP = 0.75  # ponytail: fixed cut-off; tune if review shows good questions being dropped
STOPWORDS = set(
    "a an the is are was were be been being do does did how what when where which who whom why can could should "
    "would will shall may might must i we you he she it its they them of to in on for with at by from and or not "
    "as that this these those my our your their me us there here if then than so about into over any some all "
    "each every get got have has had".split())

ANSWERABLE_V1 = '''You are helping test a search engine over a team's internal documentation.
Read the passage and write ONE question a colleague might ask that this passage answers.
Rules:
- ask about a specific fact in the passage
- use your own words; do not copy phrases from the passage
- do not mention "the passage", "the document" or "the text"

Passage:
"""{text}"""

Reply with JSON only: {{"question": "..."}}'''

ANSWERABLE_V2 = '''You are helping test a search engine over a team's internal documentation.
Read the passage and write ONE question that this passage answers, phrased the way a colleague who
has NOT read it would ask: they know what they need, but not the passage's exact terms.
Rules:
- ask about one specific fact in the passage
- use everyday words and synonyms instead of the passage's technical terms, names and phrases
- keep it short (under 20 words)
- do not mention "the passage", "the document" or "the text"

Passage:
"""{text}"""

Reply with JSON only: {{"question": "..."}}'''

ANSWERABLE_PROMPTS = {1: ANSWERABLE_V1, 2: ANSWERABLE_V2}

UNANSWERABLE_PROMPT = '''You are helping test a search engine over a team's internal documentation.
Read the passage and write ONE question on the same topic that a colleague might plausibly ask,
but that this passage does NOT answer (for example about a related system, a different step, or a
detail that is missing).
Rules:
- keep it on the same topic as the passage
- do not mention "the passage", "the document" or "the text"

Passage:
"""{text}"""

Reply with JSON only: {{"question": "..."}}'''

JUDGE_PROMPT = '''Do any of these passages contain the answer to the question? Be strict: the answer
must actually be stated, not just related.

Question: {question}

Passages:
{passages}

Answer with JSON only: {{"answered": true}} or {{"answered": false}}'''


def content_words(text: str) -> set[str]:
    return {w.lower().rstrip("s") for w in re.findall(r"[A-Za-z0-9_]+", text)
            if w.lower() not in STOPWORDS and len(w) > 1}


def overlap(question: str, passage: str) -> float:
    """Share of the question's content words that also appear in the passage."""
    q = content_words(question)
    return len(q & content_words(passage)) / len(q) if q else 1.0


def _body(text: str) -> str:
    """Chunk text without the "path > heading" prefix, so the LLM sees only the content."""
    return text.split("\n\n", 1)[1] if "\n\n" in text else text


def _question(reply: dict) -> str | None:
    q = str(reply.get("question", "")).strip()
    return q if 3 <= len(q.split()) <= 40 else None


def answers(generate: Generate, question: str, passages: list[str]) -> bool:
    numbered = "\n\n".join(f"[{i}] {_body(p)}" for i, p in enumerate(passages, 1))
    return generate(JUDGE_PROMPT.format(question=question, passages=numbered)).get("answered") is True


def sample_chunks(settings: Settings, n: int, seed: int) -> list[dict]:
    col = _get_collection(settings)
    if col is None:
        return []
    data = col.get(include=["documents", "metadatas"])
    chunks = [{"id": i, "text": t, **m} for i, t, m in zip(data["ids"], data["documents"], data["metadatas"])
              if len(_body(t).split()) >= MIN_WORDS]
    chunks.sort(key=lambda c: c["id"])  # deterministic before sampling
    return random.Random(seed).sample(chunks, min(n, len(chunks)))


def generate_gold(settings: Settings, generate: Generate, n: int = 50, seed: int = 0,
                  log=lambda msg: None, prompt_version: int = PROMPT_VERSION) -> tuple[list[dict], dict]:
    """Return (gold items, filter stats). Items: {question, answerable, source, start_line, end_line}."""
    stats = dict.fromkeys(["chunks", "answerable_written", "dropped_overlap", "dropped_not_answered",
                           "unanswerable_written", "dropped_answered_elsewhere", "dropped_duplicate"], 0)
    retrieval = dataclasses.replace(settings, top_k=3, min_score=None)
    gold, seen = [], set()
    chunks = sample_chunks(settings, n, seed)
    stats["chunks"] = len(chunks)

    for i, c in enumerate(chunks, 1):
        log(f"[{i}/{len(chunks)}] {c['source']}:L{c['start_line']}")
        body = _body(c["text"])

        q = _question(generate(ANSWERABLE_PROMPTS[prompt_version].format(text=body)))
        if q:
            stats["answerable_written"] += 1
            if q.lower() in seen:
                stats["dropped_duplicate"] += 1
            elif overlap(q, body) > MAX_OVERLAP:
                stats["dropped_overlap"] += 1
            elif not answers(generate, q, [c["text"]]):
                stats["dropped_not_answered"] += 1
            else:
                seen.add(q.lower())
                gold.append({"question": q, "answerable": True, "source": c["source"],
                             "start_line": c["start_line"], "end_line": c["end_line"]})

        u = _question(generate(UNANSWERABLE_PROMPT.format(text=body)))
        if u:
            stats["unanswerable_written"] += 1
            if u.lower() in seen:
                stats["dropped_duplicate"] += 1
            elif answers(generate, u, [r["text"] for r in search(u, retrieval)]):
                stats["dropped_answered_elsewhere"] += 1
            else:
                seen.add(u.lower())
                gold.append({"question": u, "answerable": False, "source": None,
                             "start_line": None, "end_line": None})
    return gold, stats


def save_gold(path: Path, gold: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(g) + "\n" for g in gold))


def load_gold(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
