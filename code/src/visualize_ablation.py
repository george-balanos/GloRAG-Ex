"""Aggregate ablation results and produce comparison plots + CSV.

Walks the ablation root produced by run_ablation.sh, parses each cell-dir name
to extract phase / method / hyperparameter, computes per-cell aggregate metrics,
and writes:
  - <out-dir>/ablation_summary.csv      one row per cell
  - <out-dir>/deletions.png             no-PSP vs PSP grouped bars
  - <out-dir>/additions_success_rate.png   3-panel (adm) line plot
  - <out-dir>/additions_mean_cost.png       3-panel line plot
  - <out-dir>/additions_llm_calls.png       3-panel line plot
"""

import argparse
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.cfe_evaluation.evaluate import load_results
from src.dataset_setup import DATASETS

_RE_FT_NO_PSP = re.compile(r"^ft_delete_no_psp$")
_RE_FT_PSP    = re.compile(r"^ft_delete_psp_k(?P<k>\d+)$")
_RE_TF_NONE   = re.compile(r"^tf_add_adm(?P<adm>[123])_none$")
_RE_TF_TIER   = re.compile(r"^tf_add_adm(?P<adm>[123])_tier_w(?P<w>[\d.]+)$")
_RE_TF_BLEND  = re.compile(r"^tf_add_adm(?P<adm>[123])_blend_a(?P<a>[\d.]+)$")


def parse_cell(name: str) -> dict | None:
    """Return a dict describing the cell, or None if unrecognised.

    Keys:
      phase         : 'ft' | 'tf'
      method        : 'no_psp' | 'psp' | 'none' | 'tier' | 'blend'
      adm           : int | None
      hyperparam    : 'k' | 'w' | 'a' | None
      hyperparam_value : float | int | None
    """
    if _RE_FT_NO_PSP.match(name):
        return {"phase": "ft", "method": "no_psp", "adm": None,
                "hyperparam": None, "hyperparam_value": None}
    m = _RE_FT_PSP.match(name)
    if m:
        return {"phase": "ft", "method": "psp", "adm": None,
                "hyperparam": "k", "hyperparam_value": int(m["k"])}
    m = _RE_TF_NONE.match(name)
    if m:
        return {"phase": "tf", "method": "none", "adm": int(m["adm"]),
                "hyperparam": None, "hyperparam_value": None}
    m = _RE_TF_TIER.match(name)
    if m:
        return {"phase": "tf", "method": "tier", "adm": int(m["adm"]),
                "hyperparam": "w", "hyperparam_value": float(m["w"])}
    m = _RE_TF_BLEND.match(name)
    if m:
        return {"phase": "tf", "method": "blend", "adm": int(m["adm"]),
                "hyperparam": "a", "hyperparam_value": float(m["a"])}
    return None


# ─── Discovery & per-cell aggregation ────────────────────────────────────────

def find_cells(root: Path, dataset: str) -> list[tuple[str, Path]]:
    """Yield (cell_name, leaf_dir) for every cell that has at least one JSON.

    Layout produced by run_ablation.sh:
      <root>/<cell>/<dataset>/{delete_ops_ft, add_ops_tf}/counterfactual_*.json
    """
    out: list[tuple[str, Path]] = []
    if not root.exists():
        return out
    for cell_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        ds_dir = cell_dir / dataset
        if not ds_dir.is_dir():
            continue
        for leaf in ds_dir.iterdir():
            if leaf.is_dir() and any(leaf.glob("*.json")):
                out.append((cell_dir.name, leaf))
    return out


def dedupe_latest(results: list[dict]) -> list[dict]:
    """If multiple JSONs share a question, keep the one with the latest timestamp."""
    by_q: dict[str, dict] = {}
    for r in results:
        q = r.get("question", "")
        ts = r.get("timestamp", "")
        if q not in by_q or ts > by_q[q].get("timestamp", ""):
            by_q[q] = r
    return list(by_q.values())


def _mean(xs):
    xs = list(xs)
    return statistics.mean(xs) if xs else float("nan")


def _median(xs):
    xs = list(xs)
    return statistics.median(xs) if xs else float("nan")


def aggregate(results: list[dict]) -> dict:
    found = [r for r in results if r.get("found")]
    n = len(results)
    return {
        "n_questions":      n,
        "n_found":          len(found),
        "success_rate":     len(found) / n if n else float("nan"),
        "mean_cost":        _mean(r["cost"] for r in found),
        "median_cost":      _median(r["cost"] for r in found),
        "mean_num_ops":     _mean(r["num_operations"] for r in found),
        "mean_llm_calls":   _mean(r["llm_calls"] for r in results),
        "median_llm_calls": _median(r["llm_calls"] for r in results),
    }


# ─── Plots ───────────────────────────────────────────────────────────────────

_METRICS = [
    ("success_rate",   "Success rate"),
    ("mean_cost",      "Mean cost (found)"),
    ("mean_llm_calls", "Mean LLM calls"),
]


def plot_deletions(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["phase"] == "ft"].copy()
    if sub.empty:
        print("[plot_deletions] no ft rows; skipping.")
        return

    method_order = ["no_psp", "psp"]
    sub = sub[sub["method"].isin(method_order)]
    sub["method"] = pd.Categorical(sub["method"], categories=method_order, ordered=True)
    sub = sub.sort_values("method")

    fig, axes = plt.subplots(1, len(_METRICS), figsize=(4.0 * len(_METRICS), 4.0))
    if len(_METRICS) == 1:
        axes = [axes]

    for ax, (metric, title) in zip(axes, _METRICS):
        bars = ax.bar(
            sub["method"].astype(str),
            sub[metric],
            color=["#4C72B0", "#DD8452"],
            edgecolor="black",
        )
        ax.set_title(title)
        ax.set_xlabel("")
        for bar, val in zip(bars, sub[metric]):
            if pd.notna(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Deletions (T→F): no-PSP vs PSP", fontweight="bold")
    fig.tight_layout()
    path = out / "deletions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def plot_additions(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["phase"] == "tf"].copy()
    if sub.empty:
        print("[plot_additions] no tf rows; skipping.")
        return

    adm_values = sorted(int(a) for a in sub["adm"].dropna().unique())
    if not adm_values:
        print("[plot_additions] no adm values; skipping.")
        return

    for metric, title in _METRICS:
        fig, axes = plt.subplots(
            1, len(adm_values),
            figsize=(4.5 * len(adm_values), 4.0),
            sharey=True,
        )
        if len(adm_values) == 1:
            axes = [axes]

        for ax, adm in zip(axes, adm_values):
            adm_rows = sub[sub["adm"] == adm]

            tier = adm_rows[adm_rows["method"] == "tier"].sort_values("hyperparam_value")
            blend = adm_rows[adm_rows["method"] == "blend"].sort_values("hyperparam_value")
            none_row = adm_rows[adm_rows["method"] == "none"]

            if not tier.empty:
                ax.plot(tier["hyperparam_value"], tier[metric],
                        marker="o", label="tier (width)", color="#4C72B0")
            if not blend.empty:
                ax.plot(blend["hyperparam_value"], blend[metric],
                        marker="s", label="blend (alpha)", color="#DD8452")
            if not none_row.empty and pd.notna(none_row[metric].iloc[0]):
                ax.axhline(none_row[metric].iloc[0],
                           color="gray", linestyle="--", label="none (baseline)")

            ax.set_title(f"adm = {adm}")
            ax.set_xlabel("heuristic strength (tier_width / alpha)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[0].set_ylabel(title)
        axes[0].legend(loc="best", frameon=False, fontsize=9)
        fig.suptitle(f"Additions (F→T) — {title}", fontweight="bold")
        fig.tight_layout()
        path = out / f"additions_{metric}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="visualize_ablation",
        description="Aggregate ablation result JSONs into a CSV + comparison plots.",
    )
    p.add_argument("--ablation-root", default="src/counterfactuals/results/ablation",
                   help="Root containing one subdirectory per ablation cell.")
    p.add_argument("--dataset", choices=DATASETS, default="synthetic",
                   help="Dataset name (matches the subdir inside each cell).")
    p.add_argument("--out-dir", default="src/counterfactuals/results/ablation/_viz",
                   help="Directory for CSV + PNG outputs.")
    p.add_argument("--csv", default=None,
                   help="Optional CSV path (default: <out-dir>/ablation_summary.csv).")
    return p


def main(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for cell_name, leaf in find_cells(Path(args.ablation_root), args.dataset):
        parsed = parse_cell(cell_name)
        if parsed is None:
            print(f"[skip] unrecognized cell: {cell_name}")
            continue
        results = dedupe_latest(load_results(str(leaf)))
        rows.append({"cell": cell_name, **parsed, **aggregate(results)})

    if not rows:
        raise SystemExit(
            f"No ablation results found under {args.ablation_root} for dataset={args.dataset}."
        )

    df = pd.DataFrame(rows)
    csv_path = Path(args.csv) if args.csv else out / "ablation_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} cell rows → {csv_path}")

    plot_deletions(df, out)
    plot_additions(df, out)
    print(f"Done. Outputs under {out}/")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
