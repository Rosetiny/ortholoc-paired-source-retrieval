"""OrthoLoC-PSR: protocol constants, manifests and the retrieval metric."""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import numpy as np

PROTOCOL_VERSION = "1.0"
MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"
QUERY_ARRAY = "image_query"
GALLERY_ARRAY = "image_dop"

CONDITIONS = {
    "inplace_r": {"split": "test_inPlace", "source": "R", "size": 142},
    "outplace_r": {"split": "test_outPlace", "source": "R", "size": 496},
    "outplace_xdop": {"split": "test_outPlace", "source": "xDOP", "size": 496},
}
COLUMNS = ["condition", "split", "reference_source", "location_id", "canonical_id",
           "query_id", "query_npz", "positive_gallery_id", "gallery_npz"]

_SAMPLE_ID = re.compile(r"(L\d+)_(R|xDOPDSM|xDOP)(\d+)")


class ProtocolError(ValueError):
    """The input does not conform to the benchmark protocol."""


# ---------------------------------------------------------------- sample ids and manifests

def parse_id(sample_id):
    """'L08_xDOP0133' -> ('L08', 'xDOP', '0133')."""
    match = _SAMPLE_ID.fullmatch(str(sample_id))
    if not match:
        raise ProtocolError(f"not an OrthoLoC sample id: {sample_id!r}")
    return match.groups()


def canonical_id(sample_id):
    """Id of the UAV capture, shared by every reference-source variant of a sample."""
    location, _, number = parse_id(sample_id)
    return f"{location}_{number}"


def make_rows(condition, sample_ids):
    spec = CONDITIONS[condition]
    rows = []
    for sample_id in sorted(str(s) for s in sample_ids):
        npz = f"{spec['split']}/{sample_id}.npz"
        rows.append({
            "condition": condition, "split": spec["split"], "reference_source": spec["source"],
            "location_id": parse_id(sample_id)[0], "canonical_id": canonical_id(sample_id),
            "query_id": sample_id, "query_npz": npz,
            "positive_gallery_id": sample_id, "gallery_npz": npz,
        })
    return rows


def write_manifest(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise ProtocolError(f"{path}: unexpected columns {reader.fieldnames}")
        rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ProtocolError(f"{path}: a row does not have exactly {len(COLUMNS)} fields")
    return rows


def validate_rows(condition, rows):
    """Return a list of protocol violations (empty when the manifest is valid)."""
    spec = CONDITIONS[condition]
    errors = []
    if len(rows) != spec["size"]:
        errors.append(f"expected {spec['size']} rows, found {len(rows)}")
    for i, row in enumerate(rows):
        where = f"row {i + 1}"
        for column, value in (("condition", condition), ("split", spec["split"]),
                              ("reference_source", spec["source"])):
            if row[column] != value:
                errors.append(f"{where}: {column} is {row[column]!r}, expected {value!r}")
        try:
            location, query_source, _ = parse_id(row["query_id"])
            gallery_source = parse_id(row["positive_gallery_id"])[1]
        except ProtocolError as error:
            errors.append(f"{where}: {error}")
            continue
        if query_source != spec["source"]:
            errors.append(f"{where}: query {row['query_id']} is from reference source {query_source}, "
                          f"expected {spec['source']}")
        if gallery_source != spec["source"]:
            errors.append(f"{where}: gallery item {row['positive_gallery_id']} is from reference source "
                          f"{gallery_source}, expected {spec['source']}")
        if row["positive_gallery_id"] != row["query_id"]:
            errors.append(f"{where}: the indexed positive must be the reference of the query's own sample")
        if row["location_id"] != location or row["canonical_id"] != canonical_id(row["query_id"]):
            errors.append(f"{where}: location_id/canonical_id do not match query_id")
        for column, sample_id in (("query_npz", row["query_id"]), ("gallery_npz", row["positive_gallery_id"])):
            if row[column] != f"{spec['split']}/{sample_id}.npz":
                errors.append(f"{where}: {column} must be the relative path {spec['split']}/{sample_id}.npz")
    for column in ("query_id", "positive_gallery_id"):
        values = [row[column] for row in rows]
        if len({v.lower() for v in values}) != len(values):
            errors.append(f"{column} values are not unique (compared case-insensitively)")
    queries = [row["query_id"] for row in rows]
    if queries != sorted(queries):
        errors.append("rows are not sorted by query_id")
    return errors


def validate_pairing(rows_r, rows_xdop):
    """OutPlace-R and OutPlace-xDOP must pose the same UAV queries in the same order."""
    a = [row["canonical_id"] for row in rows_r]
    b = [row["canonical_id"] for row in rows_xdop]
    if a == b:
        return []
    return [f"outplace_r and outplace_xdop do not pose the same canonical queries in the same order "
            f"({len(set(a) ^ set(b))} ids are in only one of them)"]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recorded_sha256(manifest_dir=MANIFEST_DIR):
    sums = Path(manifest_dir) / "SHA256SUMS"
    if not sums.exists():
        return {}
    return {name: digest for digest, name in (line.split() for line in sums.read_text().splitlines() if line.strip())}


def load_condition(condition, manifest_dir=MANIFEST_DIR):
    """Rows of a condition's manifest; raises unless it is intact and valid."""
    if condition not in CONDITIONS:
        raise ProtocolError(f"unknown condition {condition!r}; choose from {sorted(CONDITIONS)}")
    path = Path(manifest_dir) / f"{condition}.csv"
    if not path.exists():
        raise ProtocolError(f"manifest for {condition} is not available: {path}")
    expected = recorded_sha256(manifest_dir).get(path.name)
    if expected != sha256_file(path):
        raise ProtocolError(f"{path.name}: SHA-256 does not match SHA256SUMS; the manifest was modified")
    rows = read_manifest(path)
    errors = validate_rows(condition, rows)
    if errors:
        raise ProtocolError(f"{path.name}: " + "; ".join(errors[:5]))
    return rows


def positives(rows):
    """query_id -> id of its single indexed positive. The values are the whole gallery."""
    return {row["query_id"]: row["positive_gallery_id"] for row in rows}


# ---------------------------------------------------------------- metric

def _id_list(ids, name):
    ids = [str(i) for i in np.asarray(ids).ravel()]
    if len(set(ids)) != len(ids):
        raise ProtocolError(f"duplicate {name} ids")
    return ids


def _source(sample_id):
    try:
        return parse_id(sample_id)[1]
    except ProtocolError:
        return None


def evaluate_similarity(similarity, query_ids, gallery_ids, positives):
    """R@1 and AP (both in percent) of a Q x G similarity matrix, higher = more similar.

    The rank of a query is the number of gallery items scoring at least as high as its
    indexed positive, so a tie never counts in the query's favour.
    AP is the trapezoidal area under the precision-recall curve, which for a single
    positive at rank k is 1 when k = 1 and 1 / (2k) otherwise.
    """
    queries, gallery = _id_list(query_ids, "query"), _id_list(gallery_ids, "gallery")
    scores = np.asarray(similarity)
    if scores.dtype.kind not in "fiu":
        raise ProtocolError(f"similarity must be real-valued, got dtype {scores.dtype}")
    if scores.shape != (len(queries), len(gallery)):
        raise ProtocolError(f"similarity has shape {scores.shape}, expected "
                            f"({len(queries)} queries, {len(gallery)} gallery items)")
    scores = scores.astype(np.float64)
    if not np.isfinite(scores).all():
        raise ProtocolError("similarity must be finite (found NaN or Inf)")

    if set(queries) != set(positives):
        raise ProtocolError(f"query ids do not match the condition: {len(set(positives) - set(queries))} missing, "
                            f"{len(set(queries) - set(positives))} unexpected")
    expected = set(positives.values())
    problems = []
    if expected - set(gallery):
        problems.append(f"gallery is missing {len(expected - set(gallery))} indexed positive(s)")
    extra = set(gallery) - expected
    if extra:
        foreign = {_source(i) for i in extra} - {_source(i) for i in expected} - {None}
        problems.append(f"gallery holds {len(extra)} item(s) outside this condition"
                        + (f", from another reference source ({', '.join(sorted(foreign))})" if foreign else ""))
    if problems:
        raise ProtocolError("; ".join(problems))

    column = {gallery_id: j for j, gallery_id in enumerate(gallery)}
    positive = scores[np.arange(len(queries)), [column[positives[q]] for q in queries]]
    ranks = (scores >= positive[:, None]).sum(axis=1)
    ap = np.where(ranks == 1, 1.0, 1.0 / (2.0 * ranks))
    hits = int((ranks == 1).sum())
    return {
        "num_queries": len(queries), "num_gallery": len(gallery), "hits_at_1": hits,
        "R@1": 100.0 * hits / len(queries), "AP": 100.0 * float(ap.mean()),
        "ranks": {q: int(r) for q, r in zip(queries, ranks)},
    }


def evaluate_embeddings(query_embeddings, gallery_embeddings, query_ids, gallery_ids, positives):
    """Score by cosine similarity of the given embeddings, then evaluate."""
    normalised = []
    for name, embeddings in (("query", query_embeddings), ("gallery", gallery_embeddings)):
        e = np.asarray(embeddings)
        if e.dtype.kind not in "fiu" or e.ndim != 2:
            raise ProtocolError(f"{name} embeddings must be a real-valued N x D array")
        e = e.astype(np.float64)
        if not np.isfinite(e).all():
            raise ProtocolError(f"{name} embeddings must be finite (found NaN or Inf)")
        norm = np.linalg.norm(e, axis=1, keepdims=True)
        if (norm == 0).any():
            raise ProtocolError(f"{name} embeddings contain a zero vector")
        normalised.append(e / norm)
    if normalised[0].shape[1] != normalised[1].shape[1]:
        raise ProtocolError("query and gallery embeddings have different dimensions")
    return evaluate_similarity(normalised[0] @ normalised[1].T, query_ids, gallery_ids, positives)


def evaluate_file(condition, path, manifest_dir=MANIFEST_DIR):
    """Evaluate an .npz holding query_ids, gallery_ids and either `similarity` or
    `query_embeddings` + `gallery_embeddings`."""
    truth = positives(load_condition(condition, manifest_dir))
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        if not {"query_ids", "gallery_ids"} <= keys:
            raise ProtocolError("the input needs string arrays `query_ids` and `gallery_ids`")
        has_similarity = "similarity" in keys
        has_embeddings = {"query_embeddings", "gallery_embeddings"} <= keys
        if has_similarity == has_embeddings:
            raise ProtocolError("provide either `similarity` or `query_embeddings` + `gallery_embeddings`")
        if has_similarity:
            return evaluate_similarity(data["similarity"], data["query_ids"], data["gallery_ids"], truth)
        return evaluate_embeddings(data["query_embeddings"], data["gallery_embeddings"],
                                   data["query_ids"], data["gallery_ids"], truth)
