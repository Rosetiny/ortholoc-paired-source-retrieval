"""Evaluate retrieval scores on one OrthoLoC-PSR condition."""
import argparse
import json
import sys

import ortholoc_psr as psr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=sorted(psr.CONDITIONS))
    parser.add_argument("--input", required=True,
                        help=".npz with query_ids, gallery_ids and either similarity (Q x G) "
                             "or query_embeddings + gallery_embeddings")
    parser.add_argument("--output", help="write metrics and per-query ranks to this JSON file")
    args = parser.parse_args()

    try:
        result = psr.evaluate_file(args.condition, args.input)
    except psr.ProtocolError as error:
        sys.exit(f"ERROR: {error}")

    print(f"OrthoLoC-PSR v{psr.PROTOCOL_VERSION}  {args.condition}")
    print(f"queries / gallery : {result['num_queries']} / {result['num_gallery']}")
    print(f"R@1               : {result['R@1']:.2f}  ({result['hits_at_1']}/{result['num_queries']})")
    print(f"AP                : {result['AP']:.2f}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"protocol_version": psr.PROTOCOL_VERSION, "condition": args.condition, **result}, f, indent=2)


if __name__ == "__main__":
    main()
