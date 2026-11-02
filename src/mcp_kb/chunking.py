"""Heading-aware, token-budgeted markdown chunking.

Each chunk fits the embedding model's window (max_tokens, counting [CLS]/[SEP] and the
breadcrumb prefix), never splits a word, and keeps exact source line numbers.
"""
import bisect
import re
from pathlib import PurePosixPath

HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
SPECIAL_TOKENS = 2  # [CLS] and [SEP]


def _sections(text: str):
    """Yield (heading trail, [(line_no, line)]) for each run of lines between headings."""
    trail: list[tuple[int, str]] = []  # (level, title) stack, e.g. [(1, "Runbook"), (2, "Deploy")]
    body: list[tuple[int, str]] = []
    for line_no, line in enumerate(text.splitlines(keepends=True), 1):
        m = HEADING.match(line.rstrip("\n"))
        if m:
            if body:
                yield [t for _, t in trail], body
            level = len(m.group(1))
            trail = [h for h in trail if h[0] < level] + [(level, m.group(2))]
            body = []
        else:
            body.append((line_no, line))
    if body:
        yield [t for _, t in trail], body


def chunk_markdown(text: str, source: str, tokenizer, max_tokens: int = 256,
                   overlap: int = 32, context_prefix: bool = True) -> list[dict]:
    """Split markdown into chunks: [{"text", "start_line", "end_line"}]."""
    doc_label = str(PurePosixPath(source).with_suffix(""))
    chunks = []

    for trail, lines in _sections(text):
        body = "".join(line for _, line in lines)
        if not body.strip():
            continue
        line_starts, pos = [], 0  # char offset where each body line starts
        for _, line in lines:
            line_starts.append(pos)
            pos += len(line)
        line_of = lambda char: lines[bisect.bisect_right(line_starts, char) - 1][0]  # noqa: E731

        prefix = " > ".join([doc_label, *trail]) + "\n\n" if context_prefix else ""
        # ponytail: a huge heading trail could eat the budget; floor at 64 tokens of body
        budget = max(max_tokens - SPECIAL_TOKENS - len(tokenizer.tokenize(prefix)), 64)

        enc = tokenizer(body, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        n = len(offsets)
        # word_start[i]: token i begins a new word (whitespace before it), so we may cut there
        word_start = [i == 0 or offsets[i - 1][1] < offsets[i][0] for i in range(n)] + [True]

        start = 0
        while start < n:
            end = min(start + budget, n)
            cut = end
            while cut > start and not word_start[cut]:
                cut -= 1
            end = cut if cut > start else end  # one giant "word" longer than the budget: hard cut

            c0, c1 = offsets[start][0], offsets[end - 1][1]
            chunks.append({"text": prefix + body[c0:c1], "start_line": line_of(c0), "end_line": line_of(c1 - 1)})
            if end == n:
                break

            nxt = end - overlap
            while nxt < end and not word_start[nxt]:
                nxt += 1
            start = nxt if nxt > start else end

    return chunks
