#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_METRICS = DATA_DIR / "addresses" / "tdccp_address_metrics.csv"
OUT_DIR = ROOT / "outputs" / "figures"


# ----------------------------- helpers ---------------------------------
def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"from_address", "net_ui", "peak_balance_ui", "direct_txn_count"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"[error] metrics csv missing columns: {sorted(missing)}")
    return df


def fmt_y_plain(v, pos):
    try:
        iv = int(v)
        if abs(v - iv) < 1e-9:
            return f"{iv:,}"
        return f"{v:,.2f}"
    except Exception:
        return str(v)


def fmt_x_decades(v, pos):
    if v <= 0:
        return ""
    p = int(round(np.log10(v)))
    val = 10 ** p
    if abs(v - val) > 1e-9:
        return ""
    if val < 1_000:
        return f"{int(val)}"
    if val < 1_000_000:
        return f"{int(val/1_000)}k"
    if val < 1_000_000_000:
        return f"{int(val/1_000_000)}M"
    return f"{int(val/1_000_000_000)}B"


def compute_sizes(n: pd.Series) -> pd.Series:
    n = pd.to_numeric(n, errors="coerce").fillna(0).clip(lower=1)
    return np.sqrt(n) * 40.0


def bucket_definitions() -> List[Tuple[str, int, int, str]]:
    return [
        ("1", 1, 1, "#1f77b4"),
        ("2–10", 2, 10, "#ff7f0e"),
        ("11–50", 11, 50, "#2ca02c"),
        ("51–100", 51, 100, "#d62728"),
        ("101–500", 101, 500, "#9467bd"),
        ("501–1000", 501, 1000, "#8c564b"),
        ("1000+", 1001, np.inf, "#17becf"),
    ]


def assign_buckets(tx_counts: pd.Series) -> Tuple[pd.Series, pd.Series]:
    buckets = bucket_definitions()
    labels: List[str] = []
    colors: List[str] = []
    for count in tx_counts.to_numpy():
        label = "1000+"
        color = "#17becf"
        for name, start, end, hex_color in buckets:
            if start <= count <= end:
                label = name
                color = hex_color
                break
        labels.append(label)
        colors.append(color)
    return pd.Series(labels, index=tx_counts.index), pd.Series(colors, index=tx_counts.index)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.copy()
    sub = sub[sub["peak_balance_ui"] > 0]
    sub = sub[sub["direct_txn_count"].fillna(0) > 0]
    if "percent_intermediary" in sub.columns:
        sub = sub[sub["percent_intermediary"].fillna(0) == 0]
    if sub.empty:
        raise SystemExit("[error] no rows to plot after filtering (peak>0, direct>0, !intermediary).")
    return sub


# ----------------------------- plotting --------------------------------
def plot_bubbles(
    df: pd.DataFrame,
    window_label: str | None,
    outfile: Path,
    figsize: Tuple[float, float],
    dpi: int,
) -> None:
    sub = prepare_dataframe(df)
    sizes = compute_sizes(sub["direct_txn_count"])
    label_groups, colors = assign_buckets(sub["direct_txn_count"].astype(int))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("white")

    ax.scatter(
        sub["peak_balance_ui"],
        sub["net_ui"],
        s=sizes.to_numpy(),
        color=colors.to_numpy(),
        alpha=0.6,
        edgecolors="black",
        linewidths=0.5,
    )

    ax.set_xscale("log", base=10)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_x_decades))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=tuple(np.arange(2, 10) * 0.1)))
    ax.xaxis.set_minor_formatter(NullFormatter())

    max_x = sub["peak_balance_ui"].max()
    right = 10 ** np.ceil(np.log10(max(1.0, max_x)))
    ax.set_xlim(left=1.0, right=right)

    ax.set_yscale("symlog", base=10, linthresh=1.0, linscale=1.0)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_y_plain))
    ax.axhline(0, color="#606060", linewidth=1.0, alpha=0.8, zorder=0)

    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.grid(True, which="minor", axis="x", linestyle=":", alpha=0.2)

    ax.set_xlabel("Peak Balance (TDCCP)", fontsize=16)
    ax.set_ylabel("Net Volume (TDCCP)", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)

    legend_handles: List[Line2D] = []
    legend_labels: List[str] = []
    bucket_colors: Dict[str, str] = {name: color for name, _, _, color in bucket_definitions()}
    for label in [lab for lab in bucket_colors if (label_groups == lab).any()]:
        color = bucket_colors[label]
        proxy = Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=10,
        )
        legend_handles.append(proxy)
        legend_labels.append(label)

    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            title="Direct swaps (color buckets)",
            loc="upper left",
            frameon=True,
            framealpha=0.9,
            fontsize=12,
            title_fontsize=13,
        )

    title = "Address Bubbles — TDCCP"
    if window_label:
        title += f" — {window_label}"
    ax.set_title(title, pad=16, fontsize=20)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.6)
    fig.savefig(outfile, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"[done] {outfile}")


def parse_figsize(s: str) -> Tuple[float, float]:
    try:
        w_str, h_str = (s.split(",", 1) if "," in s else s.split("x", 1))
        return float(w_str), float(h_str)
    except Exception as exc:
        raise SystemExit("[error] --figsize must be 'width,height' (e.g. 24,12)") from exc


def main():
    ap = argparse.ArgumentParser(description="Bubble chart of addresses by TDCCP net vs peak balance.")
    ap.add_argument("--metrics", default=str(DEFAULT_METRICS), help="Path to metrics CSV")
    ap.add_argument("--figsize", type=str, default="24,12", help="Figure size in inches (width,height)")
    ap.add_argument("--dpi", type=int, default=400, help="Figure DPI (default: 400)")
    ap.add_argument("--outfile", type=str, help="Output path (PNG)")
    ap.add_argument("--window-label", type=str, help="Optional label for chart title & filename")
    ap.add_argument("--top-labels", type=int, default=0, help="(ignored) kept for pipeline compatibility")
    ap.add_argument("--min-peak", type=float, default=0.0, help="(ignored) maintained for compatibility")
    ap.add_argument("--min-direct", type=int, default=0, help="(ignored) maintained for compatibility")
    args = ap.parse_args()

    metrics = Path(args.metrics)
    if not metrics.exists():
        sys.exit(f"[error] metrics not found: {metrics}")

    df = load_metrics(metrics)
    figsize = parse_figsize(args.figsize)
    dpi = int(args.dpi)

    if args.outfile:
        outfile = Path(args.outfile)
    else:
        tag = args.window_label or "all"
        outfile = OUT_DIR / f"Address_Bubbles_addresses_{tag}.png"

    plot_bubbles(df, args.window_label, outfile, figsize, dpi)


if __name__ == "__main__":
    main()

