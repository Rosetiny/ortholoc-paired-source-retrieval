# OrthoLoC-PSR: Paired-Source UAV-to-DOP Retrieval Benchmark

A fixed image-retrieval protocol derived from the public
[OrthoLoC](https://deepscenario.github.io/OrthoLoC/) dataset. The same UAV queries are
retrieved against orthophoto galleries built from two different reference sources, so
the effect of the reference source on retrieval can be measured on identical queries.

This repository contains the protocol (manifests), an evaluator and a data preparation
tool. **It contains no imagery.** The UAV images, the orthophotos (DOP / xDOP), the
ground truth and the train/val/test split are the work of the OrthoLoC authors and must
be obtained from them; OrthoLoC is distributed under CC BY-NC-SA 4.0. What is defined
here is only how the existing paired samples are arranged into retrieval galleries.

## Conditions

| Condition | Split | Queries | Gallery | Reference source |
|---|---|---:|---:|---|
| `inplace_r` | `test_inPlace` | 142 | 142 | R-DOP (orthophoto from the same reconstruction) |
| `outplace_r` | `test_outPlace` | 496 | 496 | R-DOP |
| `outplace_xdop` | `test_outPlace` | 496 | 496 | xDOP (external orthophoto) |

- A query is the UAV image (`image_query`) of an OrthoLoC sample; a gallery item is the
  orthophoto (`image_dop`) of a sample.
- Each query has exactly one **indexed positive**: the orthophoto stored in its own
  sample. Other gallery items that happen to overlap the query are not counted as correct.
- A gallery only holds references of one source. `outplace_r` and `outplace_xdop` pose
  the same 496 UAV captures (matched by `canonical_id`) against two separate galleries.

`manifests/<condition>.csv` lists every query, its gallery item and the official `.npz`
each one comes from. `manifests/SHA256SUMS` pins the manifests.

## Usage

```bash
pip install -r requirements.txt
python validate_manifest.py
```

**1. Get the data.** Download `test_inPlace/` and `test_outPlace/` of the official
dataset (see the [OrthoLoC repository](https://github.com/deepscenario/OrthoLoC)).

**2. Prepare the inputs.**

```bash
python prepare_dataset.py --dataset-root /path/to/OrthoLoC/full --condition outplace_r --condition outplace_xdop --output-dir ./psr_data
```

This writes `psr_data/<condition>/query/<id>.jpg`, `gallery/<id>.png` and `index.csv`.
Queries are stored as JPEG (quality 95, 4:2:0) and references as lossless PNG, at native
resolution; the official files are only read. In the reference evaluation these files
were read with OpenCV, converted to RGB, resized to 384x384 (`INTER_LINEAR_EXACT`) and
normalised with the ImageNet mean and standard deviation.

**3. Score with your model** and save one `.npz` per condition containing string arrays
`query_ids` and `gallery_ids` plus either

- `similarity`: a Q x G matrix, higher meaning more similar, or
- `query_embeddings` (Q x D) and `gallery_embeddings` (G x D), which are compared by
  cosine similarity.

Rows and columns may be in any order; they are matched to the manifest by id.

**4. Evaluate.**

```bash
python evaluate.py --condition outplace_r --input my_outplace_r.npz --output result.json
```

## Metrics

For each query, its **rank** is the number of gallery items whose score is at least as
high as the score of its indexed positive. A tie therefore never helps the query, and
the result does not depend on the order of the gallery.

- **R@1**: percentage of queries with rank 1.
- **AP**: mean over queries of the trapezoidal area under the precision-recall curve.
  With a single positive at rank *k* this equals 1 for *k* = 1 and 1 / (2*k*) for
  *k* >= 2 (rank 2 gives 0.25, rank 5 gives 0.10). This is not the reciprocal rank 1/*k*.

The evaluator refuses inputs with NaN/Inf scores, duplicate ids, a missing positive, or
gallery items that do not belong to the condition (for example xDOP references in an
R-DOP condition).

## Checks behind the protocol

- The manifests were generated from, and re-checked against, the file lists produced by
  the data loader of the reference evaluation.
- For all 496 OutPlace queries, the UAV image stored in the R-DOP sample and the one
  stored in the xDOP sample are the same array, so `outplace_r` and `outplace_xdop`
  differ only in the gallery.
- `prepare_dataset.py` was compared with an existing unpacked copy of the evaluation
  inputs for all 1,134 samples: every query file is byte-identical and every reference
  is pixel-identical.

## Limitations

- Only the indexed pair counts as a match, although neighbouring samples overlap.
- `test_outPlace` covers three locations (L08: 423 queries, L50: 41, L51: 32).
- Both reference sources occur in the OrthoLoC training split, so `outplace_xdop`
  measures a change of reference source, not an unseen one.

## License

The MIT License in `LICENSE` applies only to the original code of OrthoLoC-PSR: the
evaluator (`evaluate.py`, `ortholoc_psr.py`), the data preparation script
(`prepare_dataset.py`), the manifest validator (`validate_manifest.py`) and the tests
(`tests/`).

The query–gallery manifests (`manifests/*.csv`) are **not** covered by the MIT License.
They are derived from OrthoLoC and contain OrthoLoC sample identifiers, file names and
the query–reference correspondences of OrthoLoC's paired samples. Any use of them is
subject to the license and attribution requirements of OrthoLoC (CC BY-NC-SA 4.0) and of
the upstream data on which OrthoLoC is built.

This repository does not redistribute any OrthoLoC imagery. The UAV images, the
orthophotos, the ground truth and the splits must be obtained from the OrthoLoC authors.

## Citation

OrthoLoC-PSR is built entirely on OrthoLoC
([project page](https://deepscenario.github.io/OrthoLoC/),
[repository](https://github.com/deepscenario/OrthoLoC)). If you use it, cite the
OrthoLoC paper:

```bibtex
@inproceedings{dhaouadi2025ortholoc,
  title        = {OrthoLoC: {UAV} 6-DoF Localization and Calibration Using Orthographic Geodata},
  author       = {Oussema Dhaouadi and Riccardo Marin and Johannes Michael Meier and Jacques Kaiser and Daniel Cremers},
  booktitle    = {The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year         = {2025}
}
```
