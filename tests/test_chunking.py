import pytest

from mcp_kb.chunking import chunk_markdown
from mcp_kb.models import get_tokenizer


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer("all-MiniLM-L6-v2")


def _long_section(n_words=2000):
    words = [f"w{i}" if i % 7 else "rollback.sh" for i in range(n_words)]
    return "\n".join(" ".join(words[i:i + 12]) for i in range(0, n_words, 12))


def test_every_chunk_fits_the_model_window(tok):
    chunks = chunk_markdown("# Big\n\n" + _long_section(), "big.md", tok)
    assert len(chunks) > 5
    for c in chunks:
        assert len(tok(c["text"])["input_ids"]) <= 256  # includes [CLS]/[SEP] and the prefix


def test_line_ranges_point_at_the_text(tok):
    lines = ["# Doc", ""] + [f"filler line {i} " * 5 for i in range(60)]
    lines[40] = "the zebracorn appears here"
    chunks = chunk_markdown("\n".join(lines), "doc.md", tok)
    hits = [c for c in chunks if "zebracorn" in c["text"]]
    assert hits
    assert all(c["start_line"] <= 41 <= c["end_line"] for c in hits)  # 1-based line 41


def test_breadcrumb_prefix(tok):
    text = "# Runbook\n\nIntro.\n\n## Deploy\n\nRun rollback.sh if errors spike.\n"
    chunks = chunk_markdown(text, "ops/runbook.md", tok)
    deploy = [c for c in chunks if "rollback.sh" in c["text"]][0]
    assert deploy["text"].startswith("ops/runbook > Runbook > Deploy\n\n")
    assert deploy["start_line"] == deploy["end_line"] == 7


def test_sibling_heading_replaces_previous(tok):
    text = "# A\n\n## B\n\nbee\n\n## C\n\nsea\n"
    sea = [c for c in chunk_markdown(text, "x.md", tok) if "sea" in c["text"]][0]
    assert sea["text"].startswith("x > A > C\n\n")


def test_long_sections_overlap(tok):
    chunks = chunk_markdown(_long_section(), "big.md", tok)
    for a, b in zip(chunks, chunks[1:]):
        first_word_of_b = b["text"].split("\n\n", 1)[1].split()[0]
        assert first_word_of_b in a["text"]


def test_chunks_never_split_a_word(tok):
    chunks = chunk_markdown(_long_section(), "big.md", tok)
    body_words = set(_long_section().split())
    for c in chunks:
        body = c["text"].split("\n\n", 1)[1]
        assert set(body.split()) <= body_words


def test_empty_and_headingless(tok):
    assert chunk_markdown("", "e.md", tok) == []
    assert chunk_markdown("# Only a heading\n", "e.md", tok) == []
    c = chunk_markdown("just some words", "notes/todo.md", tok)
    assert c[0]["text"] == "notes/todo\n\njust some words"


def test_prefix_can_be_disabled(tok):
    c = chunk_markdown("# T\n\nhello world", "a.md", tok, context_prefix=False)
    assert c[0]["text"] == "hello world"
