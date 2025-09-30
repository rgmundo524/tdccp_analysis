#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import FuncFormatter, LogLocator

# ---------- project defaults ----------
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ROOT / "data" / "addresses" / "tdccp_address_metrics.csv"
OUT_DIR = ROOT / "outputs" / "figures"
# --------------------------------------


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"from_address", "net_ui", "peak_balance_ui", "direct_txn_count"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"[error] metrics csv missing columns: {sorted(missing)}")
    return df


def bucket_color(tx_count: int) -> str:
    """Map tx count into a distinct categorical color group."""
    if tx_count == 1: return "tab:blue"
    if 2 <= tx_count <= 10: return "tab:orange"
    if 11 <= tx_count <= 50: return "tab:green"
    if 51 <= tx_count <= 100: return "tab:red"
    if 101 <= tx_count <= 500: return "tab:purple"
    if 501 <= tx_count <= 1000: return "tab:brown"
    return "tab:pink"  # 1000+


def bucket_label(tx_count: int) -> str:
    if tx_count == 1: return "1"
    if 2 <= tx_count <= 10: return "2–10"
    if 11 <= tx_count <= 50: return "11–50"
    if 51 <= tx_count <= 100: return "51–100"
    if 101 <= tx_count <= 500: return "101–500"
    if 501 <= tx_count <= 1000: return "501–1000"
    return "1000+"


def plot_bubbles(
    df: pd.DataFrame,
    top_labels: int,
    window_label: str | None,
    outfile: Path,
    figsize: tuple[float, float],
    dpi: int,
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Assign color groups and sizes
    df["color"] = df["direct_txn_count"].apply(bucket_color)
    df["label_group"] = df["direct_txn_count"].apply(bucket_label)
    # Bubble size ~ sqrt(tx_count) so that area ~ tx_count
    df["size"] = np.sqrt(df["direct_txn_count"].clip(lower=1)) * 40

    ax.scatter(
        df["peak_balance_ui"],
        df["net_ui"],
        s=df["size"],
        c=df["color"],
        alpha=0.6,
        edgecolor="k",
        linewidth=0.5,
    )

    # # Label top N by |net|
    # if top_labels > 0:
    #     top = df.reindex(df["net_ui"].abs().sort_values(ascending=False).index).head(top_labels)
    #     for _, row in top.iterrows():
    #         ax.text(row["peak_balance_ui"], row["net_ui"], row["from_address"][:6], fontsize=8)

    # ----- X axis: clean log ticks -----
    ax.set_xscale("log")
    xmax = max(1.0, float(df["peak_balance_ui"].max()))
    ax.set_xlim(left=1, right=xmax * 1.2)

    def fmt_x(v, pos):
        if v >= 1e9: return f"{int(v/1e9)}B"
        if v >= 1e6: return f"{int(v/1e6)}M"
        if v >= 1e3: return f"{int(v/1e3)}k"
        return str(int(v))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_x))
    
    # 10% minor ticks between decades: 2…9
    # ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.linspace(1.111, 9, 8)/10, numticks=10))
    # ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
    # ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
    # ax.xaxis.set_minor_locator(MultipleLocator( (ax.get_xticks()[1] - ax.get_xticks()[0]) / 9 ))

    ax.set_xlabel("Peak Balance (TDCCP)")
    ax.grid(True, which="major", linestyle="--", alpha=0.6)
    # ax.grid(True, which="minor", linestyle=":",alpha=0.3)

    # ----- Y axis: plain-number symlog -----
    ax.set_yscale("symlog", linthresh=1)

    def fmt_y(v, pos):
        return f"{v:,.0f}"
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_y))
    ax.set_ylabel("Net Volume (TDCCP)")

    # ----- Legend: color = txn count bucket (sizes remain continuous) -----
    handles = []
    labels = []
    for grp in ["1", "2–10", "11–50", "51–100", "101–500", "501–1000", "1000+"]:
        sub = df[df["label_group"] == grp]
        if sub.empty:
            continue
        h = ax.scatter([], [], c=sub["color"].iloc[0], s=100, alpha=0.6, edgecolor="k")
        handles.append(h)
        labels.append(grp)
    if handles:
        ax.legend(handles, labels, title="Direct swaps (color buckets)", loc="upper left")

    # Title & save
    title = "Address Bubbles — TDCCP"
    if window_label:
        title += f" — {window_label}"
    ax.set_title(title)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {outfile}")


def parse_figsize(s: str) -> tuple[float, float]:
    try:
        w, h = (x.strip() for x in s.split(","))
        return float(w), float(h)
    except Exception:
        raise SystemExit("[error] --figsize must be in the form 'W,H' (e.g. 18,12)")


def main():
    ap = argparse.ArgumentParser(description="Bubble chart of addresses by TDCCP net vs peak balance.")
    ap.add_argument("--metrics", default=str(DEFAULT_METRICS), help="Path to metrics CSV")
    ap.add_argument("--top-labels", type=int, default=20, help="Annotate top-N by |net| (default=20)")
    ap.add_argument("--min-peak", type=float, default=0.0, help="(reserved) min peak filter")
    ap.add_argument("--min-direct", type=int, default=0, help="(reserved) min direct txns filter")
    ap.add_argument("--figsize", type=str, default="40,20", help="Width,height in inches (default=30,20)")
    ap.add_argument("--dpi", type=int, default=300, help="Figure DPI (default=300)")
    ap.add_argument("--outfile", type=str, help="Output path (PNG)")
    ap.add_argument("--window-label", type=str, help="Optional label for chart title & filename")
    args = ap.parse_args()

    metrics = Path(args.metrics)
    if not metrics.exists():
        sys.exit(f"[error] metrics not found: {metrics}")

    df = load_metrics(metrics)

    # Parse size & dpi
    figsize = parse_figsize(args.figsize)
    dpi = int(args.dpi)

    # Output file
    if args.outfile:
        outfile = Path(args.outfile)
    else:
        tag = args.window_label or "all"
        outfile = OUT_DIR / f"Address_Bubbles_addresses_{tag}.png"

    plot_bubbles(df, args.top_labels, args.window_label, outfile, figsize, dpi)


if __name__ == "__main__":
    main()

