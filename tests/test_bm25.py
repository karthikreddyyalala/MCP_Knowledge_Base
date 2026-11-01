from mcp_kb import bm25


def _rows():
    return [
        {"id": "c1", "text": "Deploy fails with ERR_4021 when the token expired.", "source": "errors.md", "start_line": 3, "end_line": 4},
        {"id": "c2", "text": "Run rollback.sh to restore the previous release.", "source": "runbook.md", "start_line": 8, "end_line": 9},
        {"id": "c3", "text": "Onboarding: install Python and clone the repository.", "source": "onboarding.md", "start_line": 1, "end_line": 2},
    ]


def test_finds_exact_identifier(tmp_path):
    bm25.write(tmp_path, _rows())
    hits = bm25.query(tmp_path, "what is ERR_4021", 5)
    assert hits[0]["id"] == "c1"
    assert hits[0]["source"] == "errors.md" and hits[0]["start_line"] == 3


def test_stemming_matches_word_forms(tmp_path):
    bm25.write(tmp_path, _rows())
    assert bm25.query(tmp_path, "restoring releases", 5)[0]["id"] == "c2"


def test_punctuation_cannot_break_the_query(tmp_path):
    bm25.write(tmp_path, _rows())
    hits = bm25.query(tmp_path, 'rollback.sh C++ (urgent) "quoted" AND OR NOT*', 5)
    assert hits and hits[0]["id"] == "c2"


def test_no_words_or_no_index_returns_empty(tmp_path):
    assert bm25.query(tmp_path, "anything", 5) == []
    bm25.write(tmp_path, _rows())
    assert bm25.query(tmp_path, "!!! ???", 5) == []


def test_write_replaces_previous_index(tmp_path):
    bm25.write(tmp_path, _rows())
    bm25.write(tmp_path, _rows()[:1])
    assert bm25.query(tmp_path, "rollback", 5) == []
