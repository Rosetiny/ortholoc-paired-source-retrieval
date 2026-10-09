"""Check that the benchmark manifests are intact. Exits non-zero on any failure."""
import argparse
import sys

import ortholoc_psr as psr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", choices=sorted(psr.CONDITIONS),
                        help="condition to check (repeatable); default: all three")
    parser.add_argument("--manifest-dir", default=psr.MANIFEST_DIR)
    args = parser.parse_args()

    loaded, failed = {}, False
    for condition in args.condition or sorted(psr.CONDITIONS):
        try:
            loaded[condition] = psr.load_condition(condition, args.manifest_dir)
            print(f"OK    {condition}: {len(loaded[condition])} queries, {len(loaded[condition])} gallery items")
        except psr.ProtocolError as error:
            failed = True
            print(f"FAIL  {condition}: {error}")
    if {"outplace_r", "outplace_xdop"} <= set(loaded):
        errors = psr.validate_pairing(loaded["outplace_r"], loaded["outplace_xdop"])
        failed = failed or bool(errors)
        print("FAIL  pairing: " + errors[0] if errors else "OK    outplace_r / outplace_xdop pose the same queries")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
