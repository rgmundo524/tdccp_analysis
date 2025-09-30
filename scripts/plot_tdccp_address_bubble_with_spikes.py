#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

# ---------- defaults ----------
BASE = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = BASE / "data" / "addresses" / "tdccp_address_metrics.csv"
DEFAULT_SETTINGS = BASE / "settings.csv"
OUT_DIR = BASE / "outputs" / "figures"
# ------------------------------

# --------- formatters / ticks ---------
def eng_formatter_with_B(x: float, _pos=None) -> str:
    """English engineering formatter using k, M, B (instead of G), T."""
    if x == 0:
        return "0"
    sign = "-" if x < 0 else ""
    a = abs(x)
    if a < 1:
        # show small numbers with up to 3 decimals
        return f"{sign}{a:.3g}"
    if a < 1e3:
        return f"{sign}{a:.0f}"
    if a < 1e6:
        return f"{sign}{a/1e3:.0f}k"
    if a < 1e9:
        return f"{sign}{a/1e6:.0f}M"
    if a < 1e12:
        return f"{sign}{a/1e9:.0f}B"  # key difference vs 'G'
    return f"{sign}{a/1e12:.0f}T"

ENG_FMT = FuncFormatter(eng_formatter_with_B)

def set_log_x_with_decades_and_deciles(ax: plt.Axes, data_min: float, data_max: float):
    """Log X axis with major ticks at decades, minor ticks at 20..90% between decades."""
    # clamp lower bound to 1 if all positive and we want a clean scale
    lo = max(1.0, float(np.nanmin([data_min, 1.0])))
    hi = float(max(data_max, lo * 10))

    ax.set_xscale("log")
    # major decades
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
    ax.xaxis.set_major_formatter(ENG_FMT)

    # minor: 2..9 * each decade (i.e., 20%, 30%, .. 90%)
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    # no special minor formatter
    ax.set_xlim(lo, hi)

def set_symlog_y(ax: plt.Axes, data: pd.Series, linthresh: float = 1.0):
    """Symmetric log; show actual numbers with EngFormatter."""
    ax.set_yscale("symlog", linthresh=linthresh, linscale=1.0, base=10.0)
    ax.yaxis.set_major_formatter(ENG_FMT)
    # Let Matplotlib decide limits but ensure we include extremes:
    ymin = np.nanmin([data.min(), -linthresh*10])
    ymax = np.nanmax([data.max(), linthresh*10])
    if np.isfinite(ymin) and np.isfinite(ymax) and ymin < ymax:
        ax.set_ylim(ymin, ymax)

# --------- settings.csv helpers ---------
def read_address_labels(settings_csv: Path) -> Dict[str, str]:
    """
    Read settings.csv (Category, Key, Value, ...) and return
    { from_address -> label } for rows where Category == 'address' (case-insensitive).
    """
    if not settings_csv.exists():
        return {}
    # Read with csv module (tolerant of comments/extra cols)
    labels: Dict[str, str] = {}
    with settings_csv.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # Default column indexes
        cat_i, key_i, val_i = 0, 1, 2
        if header:
            # try to find by name
            cols = [c.strip().lower() for c in header]
            if "category" in cols: cat_i = cols.index("category")
            if "key"      in cols: key_i = cols.index("key")
            if "value"    in cols: val_i = cols.index("value")
        for row in reader:
            if not row or len(row) <= max(cat_i, key_i, val_i):
                continue
            cat = (row[cat_i] or "").strip().lower()
            key = (row[key_i] or "").strip()
            val = (row[val_i] or "").strip()
            if cat == "address" and key:
                labels[key] = val or "Labeled"
    return labels

def read_spike_addresses(spike_csv: Path) -> Set[str]:
    """
    Read spike addresses CSV (must contain 'from_address'). Returns a set of addresses.
    """
    df = pd.read_csv(spike_csv)
    if "from_address" not in df.columns:
        raise SystemExit(f"[error] spike addresses file missing 'from_address' column: {spike_csv}")
    return set(df["from_address"].astype(str))

# --------- color palette helpers ---------
def make_color_cycle(n: int) -> List[str]:
    """
    Return an array of n distinct colors (cycling if needed) with a readable palette.
    """
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]
    if n <= len(base):
        return base[:n]
    out = []
    i = 0
    while len(out) < n:
        out.append(base[i % len(base)])
        i += 1
    return out

# --------- plotting ---------
def plot_bubbles_by_label(
    metrics_csv: Path,
    settings_csv: Path,
    outfile: Path,
    top_labels: int = 20,
    figsize: Tuple[float, float] = (20.0, 12.0),
    dpi: int = 180,
    window_label: Optional[str] = None,
    spike_addresses_csv: Optional[Path] = None,
):
    """
    Plot address bubbles using label groups from settings.csv[address] and (optionally)
    add a 'Spike' group from spike_addresses_csv.
    """
    df = pd.read_csv(metrics_csv)
    need = {"from_address", "peak_balance_ui", "net_ui", "direct_txn_count"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"[error] metrics csv missing columns: {sorted(missing)}")

    # Coerce types
    df["from_address"] = df["from_address"].astype(str)
    df["peak_balance_ui"] = pd.to_numeric(df["peak_balance_ui"], errors="coerce")
    df["net_ui"] = pd.to_numeric(df["net_ui"], errors="coerce")
    df["direct_txn_count"] = pd.to_numeric(df["direct_txn_count"], errors="coerce").fillna(0).astype(int)

    # Positive peak filter (so log scale is valid)
    df = df[df["peak_balance_ui"] > 0].copy()
    if df.empty:
        raise SystemExit("[info] No rows with peak_balance_ui > 0 to plot.")

    # Labels from settings
    addr_labels = read_address_labels(settings_csv)
    df["label"] = df["from_address"].map(addr_labels).fillna("Other")

    # If spike file present, overwrite label to "Spike" for those addresses
    if spike_addresses_csv:
        spike_set = read_spike_addresses(spike_addresses_csv)
        df.loc[df["from_address"].isin(spike_set), "label"] = "Spike"

    # Colors for unique labels (excluding "Other" which will be light gray at the end)
    labels_order = [l for l in sorted(df["label"].unique()) if l != "Other"]
    colors = make_color_cycle(len(labels_order))
    label_to_color = {lbl: col for lbl, col in zip(labels_order, colors)}
    # Grey for unlabeled
    label_to_color["Other"] = "#c9c9c9"

    # Bubble size (dynamic by direct swaps)
    dcnt = df["direct_txn_count"].clip(lower=0)
    sizes = 24.0 + 18.0 * np.sqrt(dcnt)  # continuous size

    # Figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot by label group to ensure legend is grouped
    for lbl in labels_order + (["Other"] if "Other" in df["label"].unique() else []):
        sub = df[df["label"] == lbl]
        if sub.empty:
            continue
        ax.scatter(
            sub["peak_balance_ui"].values,
            sub["net_ui"].values,
            s=sizes[sub.index],
            c=label_to_color[lbl],
            alpha=0.7,
            edgecolor="none",
            label=f"{lbl} ({len(sub)})",
            zorder=3 if lbl != "Other" else 2
        )

    # Axes
    set_log_x_with_decades_and_deciles(ax, df["peak_balance_ui"].min(), df["peak_balance_ui"].max())
    set_symlog_y(ax, df["net_ui"], linthresh=1.0)

    # Gridlines (major + minor on x; major only on y to avoid clutter)
    ax.grid(True, which="major", linestyle="--", alpha=0.25)
    ax.grid(True, which="minor", axis="x", linestyle=":", alpha=0.15)

    ax.set_xlabel("Peak TDCCP balance (units, log scale)")
    ax.set_ylabel("Net TDCCP volume (buy − sell, symmetric log)")

    # Title
    ttl = "TDCCP Address Bubbles by Label"
    if window_label:
        ttl += f" — {window_label}"
    ax.set_title(ttl, pad=10)

    # # Label top N addresses by |net_ui|
    # if top_labels and top_labels > 0:
    #     top = df.reindex(df["net_ui"].abs().sort_values(ascending=False).head(top_labels).index)
    #     for _, r in top.iterrows():
    #         ax.annotate(
    #             r["from_address"][:8],
    #             (r["peak_balance_ui"], r["net_ui"]),
    #             textcoords="offset points",
    #             xytext=(6, 4),
    #             ha="left",
    #             fontsize=8,
    #             alpha=0.85,
    #             zorder=5,
    #         )

    # Legend (labels)
    leg = ax.legend(loc="best", frameon=False, title="Label (count)")
    if leg and leg.get_title():
        leg.get_title().set_fontsize(10)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {outfile}")

# --------- CLI ---------
def main():
    ap = argparse.ArgumentParser(
        description="Plot TDCCP address bubble chart colored by settings.csv address labels; optionally highlight a spike addresses group."
    )
    ap.add_argument("--metrics", type=Path, default=DEFAULT_METRICS, help="Path to tdccp_address_metrics.csv")
    ap.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS, help="Path to settings.csv")
    ap.add_argument("--spike-addresses", type=Path, help="Path to spike_addresses_*.csv (optional). If provided, a 'Spike' group is added.")
    ap.add_argument("--top-labels", type=int, default=20, help="Annotate top-N addresses by |net_ui|")
    ap.add_argument("--figsize", default="40x20", help="Figure size as WxH in inches (default: 20x12)")
    ap.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    ap.add_argument("--outfile", type=Path, default=None, help="Output PNG path (default auto)")
    ap.add_argument("--window-label", default=None, help="Text to append in the title/filename (e.g., 20250301-20250503)")
    args = ap.parse_args()

    if not args.metrics.exists():
        raise SystemExit(f"[error] metrics file not found: {args.metrics}")
    if not args.settings.exists():
        raise SystemExit(f"[error] settings.csv not found: {args.settings}")

    # parse figsize
    try:
        w, h = (float(p) for p in args.figsize.lower().split("x"))
        figsize = (w, h)
    except Exception:
        figsize = (20.0, 12.0)

    # outfile
    if args.outfile is None:
        tag = args.window_label or "window"
        args.outfile = OUT_DIR / f"Address_Bubbles_by_label_{tag}.png"

    plot_bubbles_by_label(
        metrics_csv=args.metrics,
        settings_csv=args.settings,
        outfile=args.outfile,
        top_labels=args.top_labels,
        figsize=figsize,
        dpi=args.dpi,
        window_label=args.window_label,
        spike_addresses_csv=args.spike_addresses,
    )

if __name__ == "__main__":
    main()

