import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from analyze_shapley_permutation import main


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analyze_shapley_permutation_text",
        description="Per-question top-1/2/5 + exact-ranking agreement across text-chunk "
                    "Shapley permutations (run_shapley_text.py --permute output).")
    p.add_argument("--input", required=True, help="A run_shapley_text.py --permute output JSON.")
    p.add_argument("--output", default=None, help="Base path for output CSV, JSON summary, and plots.")
    return p


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
