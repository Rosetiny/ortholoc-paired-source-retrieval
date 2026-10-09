import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import ortholoc_psr as psr
import prepare_dataset

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "private"
AVAILABLE = [c for c in psr.CONDITIONS if (psr.MANIFEST_DIR / f"{c}.csv").exists()]


def all_at_rank(rank, n=8):
    """n queries, each with its positive outscored by exactly rank-1 other gallery items."""
    sim = np.full((n, n), 0.1)
    for i in range(n):
        sim[i, i] = 0.5
        sim[i, [(i + d) % n for d in range(1, rank)]] = 0.9
    q, g = [f"q{i}" for i in range(n)], [f"g{i}" for i in range(n)]
    return psr.evaluate_similarity(sim, q, g, dict(zip(q, g)))


def toy(n=12, seed=0):
    rng = np.random.default_rng(seed)
    q = [f"q{i:02d}" for i in range(n)]
    g = [f"g{i:02d}" for i in range(n)]
    return rng.normal(size=(n, n)), q, g, dict(zip(q, g))


# ---------------------------------------------------------------- metric

@pytest.mark.parametrize("rank, ap", [(1, 100.0), (2, 25.0), (3, 100.0 / 6), (5, 10.0)])
def test_ap_of_single_indexed_positive(rank, ap):
    result = all_at_rank(rank)
    assert set(result["ranks"].values()) == {rank}
    assert result["AP"] == pytest.approx(ap)
    assert result["R@1"] == (100.0 if rank == 1 else 0.0)


@pytest.mark.parametrize("rank", range(1, 40))
def test_ap_equals_trapezoidal_precision_recall_area(rank):
    # Trapezoid under the precision-recall curve with one positive at `rank`:
    # recall jumps 0 -> 1, precision goes from the value just before the hit to 1/rank.
    row = rank - 1
    precision_before = 1.0 if row == 0 else 0.0 / row
    trapezoid = (precision_before + 1.0 / rank) / 2.0
    assert all_at_rank(rank, n=40)["AP"] == pytest.approx(100.0 * trapezoid)


def test_metrics_are_averaged_over_queries():
    sim = np.array([[9, 1, 1, 1, 1], [1, 9, 1, 1, 1], [1, 9, 5, 1, 1], [9, 9, 9, 9, 5]], dtype=float)
    q, g = ["a", "b", "c", "d"], ["A", "B", "C", "D", "E"]
    with pytest.raises(psr.ProtocolError):
        psr.evaluate_similarity(sim, q, g, {"a": "A", "b": "B", "c": "C", "d": "E"})  # D unused
    sim, g = sim[:, [0, 1, 2, 4]], ["A", "B", "C", "E"]
    result = psr.evaluate_similarity(sim, q, g, {"a": "A", "b": "B", "c": "C", "d": "E"})
    assert list(result["ranks"].values()) == [1, 1, 2, 4]
    assert result["hits_at_1"] == 2 and result["R@1"] == 50.0
    assert result["AP"] == pytest.approx(100.0 * (1 + 1 + 0.25 + 0.125) / 4)


def test_result_does_not_depend_on_query_or_gallery_order():
    sim, q, g, pos = toy()
    base = psr.evaluate_similarity(sim, q, g, pos)
    rng = np.random.default_rng(1)
    pq, pg = rng.permutation(len(q)), rng.permutation(len(g))
    shuffled = psr.evaluate_similarity(sim[pq][:, pg], [q[i] for i in pq], [g[j] for j in pg], pos)
    assert shuffled["ranks"] == base["ranks"]
    assert shuffled["AP"] == pytest.approx(base["AP"]) and shuffled["R@1"] == base["R@1"]


def test_tied_scores_rank_the_positive_last_among_ties():
    sim = np.eye(4)
    sim[0] = [0.7, 0.7, 0.7, 0.1]
    q, g = ["q0", "q1", "q2", "q3"], ["g0", "g1", "g2", "g3"]
    for order in ([0, 1, 2, 3], [3, 2, 1, 0], [1, 0, 3, 2]):
        result = psr.evaluate_similarity(sim[:, order], q, [g[j] for j in order], dict(zip(q, g)))
        assert result["ranks"] == {"q0": 3, "q1": 1, "q2": 1, "q3": 1}


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_scores_are_rejected(bad):
    sim, q, g, pos = toy()
    sim[3, 4] = bad
    with pytest.raises(psr.ProtocolError, match="finite"):
        psr.evaluate_similarity(sim, q, g, pos)


def test_shape_mismatch_is_rejected():
    sim, q, g, pos = toy()
    with pytest.raises(psr.ProtocolError, match="shape"):
        psr.evaluate_similarity(sim[:, :-1], q, g, pos)


def test_duplicate_ids_are_rejected():
    sim, q, g, pos = toy()
    with pytest.raises(psr.ProtocolError, match="duplicate"):
        psr.evaluate_similarity(sim, q, g[:-1] + [g[0]], pos)


def test_missing_positive_is_rejected():
    sim, q, g, pos = toy()
    with pytest.raises(psr.ProtocolError, match="missing"):
        psr.evaluate_similarity(sim[:, :-1], q, g[:-1], pos)


def test_embeddings_are_scored_by_cosine_similarity():
    rng = np.random.default_rng(2)
    _, q, g, pos = toy()
    qe, ge = rng.normal(size=(12, 16)), rng.normal(size=(12, 16)) * 7.0
    cosine = (qe / np.linalg.norm(qe, axis=1, keepdims=True)) @ (ge / np.linalg.norm(ge, axis=1, keepdims=True)).T
    assert psr.evaluate_embeddings(qe, ge, q, g, pos)["ranks"] == psr.evaluate_similarity(cosine, q, g, pos)["ranks"]
    ge[0] = 0.0
    with pytest.raises(psr.ProtocolError, match="zero"):
        psr.evaluate_embeddings(qe, ge, q, g, pos)


# ---------------------------------------------------------------- manifests

def test_all_three_manifests_exist():
    assert set(AVAILABLE) == set(psr.CONDITIONS)


@pytest.mark.parametrize("condition", sorted(psr.CONDITIONS))
def test_every_condition_can_be_evaluated_end_to_end(condition, tmp_path):
    truth = psr.positives(psr.load_condition(condition))
    queries, gallery = list(truth), list(truth.values())
    scores = tmp_path / "scores.npz"
    np.savez(scores, query_ids=np.array(queries), gallery_ids=np.array(gallery[::-1]),
             similarity=np.eye(len(queries))[:, ::-1])
    result = psr.evaluate_file(condition, scores)
    assert result["hits_at_1"] == result["num_queries"] == psr.CONDITIONS[condition]["size"]
    assert result["R@1"] == result["AP"] == 100.0


@pytest.mark.parametrize("condition", AVAILABLE)
def test_released_manifest_is_valid_and_matches_recorded_sha256(condition):
    rows = psr.load_condition(condition)
    spec = psr.CONDITIONS[condition]
    assert len(rows) == len(psr.positives(rows)) == spec["size"]
    assert {psr.parse_id(r["positive_gallery_id"])[1] for r in rows} == {spec["source"]}


def corrupt(condition, edit):
    rows = [dict(r) for r in psr.load_condition(condition)]
    edit(rows)
    return psr.validate_rows(condition, rows)


def test_manifest_corruptions_are_detected():
    assert corrupt("outplace_r", lambda rows: None) == []
    assert corrupt("outplace_r", lambda rows: rows.pop())
    assert corrupt("outplace_r", lambda rows: rows.__setitem__(1, dict(rows[0])))
    assert corrupt("outplace_r", lambda rows: rows.reverse())
    assert corrupt("outplace_r", lambda rows: rows[5].update(query_npz="/data/" + rows[5]["query_npz"]))
    assert corrupt("outplace_r", lambda rows: rows[5].update(positive_gallery_id=rows[6]["positive_gallery_id"]))

    def swap_source(rows):
        other = rows[7]["query_id"].replace("_R", "_xDOP")
        rows[7].update(positive_gallery_id=other, gallery_npz=f"test_outPlace/{other}.npz")
    assert any("reference source" in e for e in corrupt("outplace_r", swap_source))


def test_outplace_conditions_share_the_same_queries():
    r, x = psr.load_condition("outplace_r"), psr.load_condition("outplace_xdop")
    assert psr.validate_pairing(r, x) == []
    assert [row["canonical_id"] for row in r] == [row["canonical_id"] for row in x]
    assert not set(psr.positives(r).values()) & set(psr.positives(x).values())
    x[3]["canonical_id"] = "L08_9999"
    assert psr.validate_pairing(r, x)


def test_gallery_from_the_other_reference_source_is_rejected():
    r, x = psr.load_condition("outplace_r"), psr.load_condition("outplace_xdop")
    q = [row["query_id"] for row in r]
    g = [row["positive_gallery_id"] for row in x]
    with pytest.raises(psr.ProtocolError, match="reference source"):
        psr.evaluate_similarity(np.eye(len(q)), q, g, psr.positives(r))


def test_validate_cli_exits_nonzero_on_a_bad_manifest(tmp_path):
    for name in ("outplace_r.csv", "outplace_xdop.csv", "SHA256SUMS"):
        (tmp_path / name).write_bytes((psr.MANIFEST_DIR / name).read_bytes())
    cmd = [sys.executable, str(ROOT / "validate_manifest.py"), "--manifest-dir", str(tmp_path),
           "--condition", "outplace_r", "--condition", "outplace_xdop"]
    assert subprocess.run(cmd, capture_output=True).returncode == 0
    with open(tmp_path / "outplace_r.csv", "ab") as f:
        f.write(b"x\n")
    assert subprocess.run(cmd, capture_output=True).returncode != 0


# ---------------------------------------------------------------- data preparation

def test_prepare_writes_q95_jpeg_queries_and_lossless_png_references(tmp_path):
    rng = np.random.default_rng(3)
    rows = psr.make_rows("outplace_r", ["L08_R0001", "L08_R0000"])
    raw = tmp_path / "raw" / "test_outPlace"
    raw.mkdir(parents=True)
    arrays = {}
    for row in rows:
        arrays[row["query_id"]] = (rng.integers(0, 256, (48, 64, 3), dtype=np.uint8),
                                   rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
        np.savez(raw / f"{row['query_id']}.npz", image_query=arrays[row["query_id"]][0],
                 image_dop=arrays[row["query_id"]][1])
    before = {p.name: p.read_bytes() for p in raw.iterdir()}

    out = tmp_path / "out"
    prepare_dataset.prepare(tmp_path / "raw", rows, out)

    for sid, (query, dop) in arrays.items():
        reference = io.BytesIO()
        Image.fromarray(query).save(reference, "JPEG", quality=95)
        assert (out / "query" / f"{sid}.jpg").read_bytes() == reference.getvalue()
        assert np.array_equal(np.asarray(Image.open(out / "gallery" / f"{sid}.png")), dop)
    assert (out / "index.csv").read_text().splitlines()[1].startswith("L08_R0000,query/L08_R0000.jpg")
    assert {p.name: p.read_bytes() for p in raw.iterdir()} == before

    np.savez(raw / "L08_R0000.npz", image_query=arrays["L08_R0000"][0].astype(np.float32),
             image_dop=arrays["L08_R0000"][1])
    with pytest.raises(psr.ProtocolError, match="uint8"):
        prepare_dataset.prepare(tmp_path / "raw", rows, tmp_path / "out2")


# ---------------------------------------------------------------- frozen results (private, never committed)

def frozen_cases():
    expected_file = PRIVATE / "expected_seed1.json"
    if not expected_file.exists():
        return []
    expected = json.loads(expected_file.read_text(encoding="utf-8"))
    return [(c, m, v) for c, models in expected.items() for m, v in models.items()
            if (PRIVATE / "frozen_scores" / f"{m}__{c}.npz").exists()]


@pytest.mark.skipif(not frozen_cases(), reason="private frozen scores are not present")
@pytest.mark.parametrize("condition, model, expected", frozen_cases(), ids=lambda v: v if isinstance(v, str) else "")
def test_frozen_similarity_reproduces_recorded_metrics(condition, model, expected):
    path = PRIVATE / "frozen_scores" / f"{model}__{condition}.npz"
    result = psr.evaluate_file(condition, path)
    assert result["hits_at_1"] == expected["hits"]
    assert result["num_queries"] == expected["queries"]
    assert result["R@1"] == pytest.approx(expected["r1"], abs=1e-4)
    assert result["AP"] == pytest.approx(expected["ap"], abs=1e-4)
    assert f"{result['R@1']:.2f}" == f"{expected['paper_r1']:.2f}"
    assert f"{result['AP']:.2f}" == f"{expected['paper_ap']:.2f}"
    with np.load(path) as z:
        sim = z["similarity"]
    positive = sim[np.arange(len(sim)), np.arange(len(sim))]
    assert ((sim == positive[:, None]).sum(axis=1) == 1).all(), "a tie would make the frozen ranking ambiguous"

    out = subprocess.run([sys.executable, str(ROOT / "evaluate.py"), "--condition", condition, "--input", str(path)],
                         capture_output=True, text=True)
    assert out.returncode == 0 and f"{expected['paper_r1']:.2f}" in out.stdout
