"""BEIR runner on a tiny fake dataset in BEIR's on-disk format (no network)."""
import json

from mcp_kb.evals.beir import doc_ids_from_chunks, load, run, write_corpus


def _fake_dataset(root):
    ds = root / "fake"
    (ds / "qrels").mkdir(parents=True)
    corpus = [
        {"_id": "d1", "title": "Volcanoes", "text": "Magma rises through the crust and erupts as lava from volcanoes."},
        {"_id": "d2", "title": "Photosynthesis", "text": "Plants convert sunlight, water and carbon dioxide into glucose."},
        {"_id": "d3", "title": "Tides", "text": "The gravitational pull of the moon causes ocean tides."},
    ]
    queries = [
        {"_id": "q1", "text": "how do plants make sugar from sunlight"},
        {"_id": "q2", "text": "what causes ocean tides"},
        {"_id": "q3", "text": "a query with no test judgments"},
    ]
    (ds / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in corpus))
    (ds / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries))
    (ds / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td2\t1\nq2\td3\t2\n")
    return ds


def test_load_keeps_only_judged_queries(tmp_path):
    corpus, queries, qrels = load(_fake_dataset(tmp_path))
    assert set(corpus) == {"d1", "d2", "d3"}
    assert set(queries) == {"q1", "q2"}
    assert qrels == {"q1": {"d2": 1}, "q2": {"d3": 2}}


def test_write_corpus_one_markdown_file_per_doc(tmp_path):
    corpus, _, _ = load(_fake_dataset(tmp_path))
    out = tmp_path / "docs"
    write_corpus(corpus, out)
    assert sorted(p.name for p in out.iterdir()) == ["d1.md", "d2.md", "d3.md"]
    assert (out / "d2.md").read_text().startswith("# Photosynthesis")


def test_doc_ids_from_chunks_dedupes_in_rank_order():
    hits = [{"source": "d2.md"}, {"source": "d1.md"}, {"source": "d2.md"}, {"source": "d3.md"}]
    assert doc_ids_from_chunks(hits) == ["d2", "d1", "d3"]


def test_run_end_to_end(tmp_path):
    corpus, queries, qrels = load(_fake_dataset(tmp_path))
    result = run("fake", corpus, queries, qrels, tmp_path / "work", setup="dense")
    m = result["metrics"]
    assert {"ndcg@10", "recall@10", "mrr@10", "p50_ms", "p95_ms", "n_queries"} <= set(m)
    assert m["n_queries"] == 2
    assert result["run"]["q2"][0] == "d3"
    assert m["mrr@10"] == 1.0


def test_setups_with_different_chunking_get_separate_indexes(tmp_path):
    corpus, queries, qrels = load(_fake_dataset(tmp_path))
    run("fake", corpus, queries, qrels, tmp_path / "work", setup="dense")
    run("fake", corpus, queries, qrels, tmp_path / "work", setup="dense-no-prefix")
    assert len(list((tmp_path / "work" / "fake").glob("chroma-*"))) == 2
