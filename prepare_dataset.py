"""Extract the query and gallery images of OrthoLoC-PSR from the official OrthoLoC .npz samples.

The official files are only read. Queries are written as JPEG (quality 95, 4:2:0) and
references as PNG, which is the form the benchmark inputs were evaluated in.
"""
import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import ortholoc_psr as psr


def _rgb_uint8(array, what):
    array = np.asarray(array)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise psr.ProtocolError(f"{what}: expected an H x W x 3 uint8 image, got {array.dtype} {array.shape}")
    return array


def encode_query(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, "JPEG", quality=95, subsampling="4:2:0",
                                optimize=False, progressive=False)
    return buffer.getvalue()


def encode_gallery(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, "PNG")
    return buffer.getvalue()


def prepare(dataset_root, rows, out_dir):
    """Write query/<id>.jpg, gallery/<id>.png and index.csv for the manifest rows."""
    dataset_root, out_dir = Path(dataset_root).resolve(), Path(out_dir).resolve()
    if out_dir == dataset_root or dataset_root in out_dir.parents:
        raise psr.ProtocolError("the output directory must be outside the dataset root")
    missing = [p for row in rows for p in {row["query_npz"], row["gallery_npz"]}
               if not (dataset_root / p).is_file()]
    if missing:
        raise psr.ProtocolError(f"{len(missing)} sample(s) not found under {dataset_root}, e.g. {missing[0]}")

    (out_dir / "query").mkdir(parents=True, exist_ok=True)
    (out_dir / "gallery").mkdir(parents=True, exist_ok=True)
    index = []
    for row in rows:
        with np.load(dataset_root / row["query_npz"], allow_pickle=False) as sample:
            query = _rgb_uint8(sample[psr.QUERY_ARRAY], f"{row['query_npz']}[{psr.QUERY_ARRAY}]")
        with np.load(dataset_root / row["gallery_npz"], allow_pickle=False) as sample:
            gallery = _rgb_uint8(sample[psr.GALLERY_ARRAY], f"{row['gallery_npz']}[{psr.GALLERY_ARRAY}]")
        query_image = f"query/{row['query_id']}.jpg"
        gallery_image = f"gallery/{row['positive_gallery_id']}.png"
        (out_dir / query_image).write_bytes(encode_query(query))
        (out_dir / gallery_image).write_bytes(encode_gallery(gallery))
        index.append([row["query_id"], query_image, row["positive_gallery_id"], gallery_image])
    with open(out_dir / "index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["query_id", "query_image", "positive_gallery_id", "gallery_image"])
        writer.writerows(index)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True,
                        help="directory holding the official test_inPlace/ and test_outPlace/ .npz folders")
    parser.add_argument("--condition", action="append", required=True, choices=sorted(psr.CONDITIONS),
                        help="condition to prepare (repeatable)")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        for condition in args.condition:
            rows = psr.load_condition(condition)
            prepare(args.dataset_root, rows, Path(args.output_dir) / condition)
            print(f"{condition}: wrote {len(rows)} queries and {len(rows)} gallery images")
    except psr.ProtocolError as error:
        sys.exit(f"ERROR: {error}")


if __name__ == "__main__":
    main()
