#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import colorsys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_METRICS = DATA_DIR / "addresses" / "tdccp_address_metrics.csv"
SETTINGS = ROOT / "settings.csv"
OUT_DIR = ROOT / "outputs" / "figures"


# ----------------------------- helpers ---------------------------------
def read_settings_labels(settings_csv: Path) -> Tuple[Dict[str, str], List[str]]:
    """
    Read address labels from settings.csv into {from_address: label}.
    Expects rows where Category (first column) == 'address', Key is the
    address, and Value is the label.
    """
    if not settings_csv.exists():
        return {}, []

    addr_map: Dict[str, str] = {}
    ordered_labels: List[str] = []
    seen_labels: set[str] = set()
    key_idx, val_idx, cat_idx = 1, 2, 0

    with settings_csv.open("r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if header:
            # detect columns by name if present
            name_to_idx = { (c or "").strip().lower(): i for i, c in enumerate(header) }
            if "category" in name_to_idx: cat_idx = name_to_idx["category"]
            if "key" in name_to_idx:      key_idx = name_to_idx["key"]
            if "value" in name_to_idx:    val_idx = name_to_idx["value"]

        for row in rdr:
            if not row or len(row) <= max(cat_idx, key_idx, val_idx):
                continue
            cat = (row[cat_idx] or "").strip().lower()
            if cat != "address":
                continue
            addr = (row[key_idx] or "").strip()
            label = (row[val_idx] or "").strip()
            if addr and label:
                addr_map[addr] = label
                if label not in seen_labels:
                    ordered_labels.append(label)
                    seen_labels.add(label)

    return addr_map, ordered_labels


def human_range_label(start: Optional[str], end: Optional[str]) -> Optional[str]:
    if not start or not end:
        return None
    s = pd.to_datetime(start, utc=True, errors="coerce")
    e = pd.to_datetime(end,   utc=True, errors="coerce")
    if pd.isna(s) or pd.isna(e):
        return None
    return f"{s.strftime('%Y%m%d')}-{e.strftime('%Y%m%d')}"


def settings_window() -> Tuple[Optional[str], Optional[str]]:
    start = None
    end = None
    if SETTINGS.exists():
        with SETTINGS.open("r", encoding="utf-8") as f:
            rdr = csv.reader(f)
            header = next(rdr, None)
            key_idx, val_idx = 1, 2
            if header:
                hmap = { (c or "").strip().lower(): i for i, c in enumerate(header) }
                if "key" in hmap:   key_idx = hmap["key"]
                if "value" in hmap: val_idx = hmap["value"]
            for row in rdr:
                if not row or len(row) <= max(key_idx, val_idx):
                    continue
                k = (row[key_idx] or "").strip().upper()
                v = (row[val_idx] or "").strip()
                if k == "START":
                    start = v
                elif k == "END":
                    end = v
    return start, end


def fmt_y_plain(v, pos):
    # plain integers with thousands separators on a symlog axis
    try:
        iv = int(v)
        if abs(v - iv) < 1e-9:
            return f"{iv:,}"
        return f"{v:,.2f}"
    except Exception:
        return str(v)


def fmt_x_decades(v, pos):
    # base-10 ticks: 1, 10, 100, 1k, 10k, 100k, 1M, ...
    if v <= 0:
        return ""
    p = int(round(np.log10(v)))
    val = 10 ** p
    if abs(v - val) > 1e-9:
        return ""  # only label the decades; minors are unlabeled
    if val < 1_000:
        return f"{int(val)}"
    elif val < 1_000_000:
        return f"{int(val/1_000)}k"
    elif val < 1_000_000_000:
        return f"{int(val/1_000_000)}M"
    else:
        return f"{int(val/1_000_000_000)}B"


def compute_sizes(n: pd.Series) -> pd.Series:
    """Match the base address bubble sizing (area ~ direct swap count)."""

    # The base plot (`plot_tdccp_address_bubble.py`) uses sqrt(count) * 40 for
    # the scatter ``s`` value which keeps single-transaction addresses visible
    # while giving multi-hundred swap actors a noticeable—but not overpowering—
    # footprint.  Mirroring that here keeps the two plots visually aligned.
    n = pd.to_numeric(n, errors="coerce").fillna(0).clip(lower=1)
    return np.sqrt(n) * 40.0


def slugify_label(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", label.strip()).strip("_")
    return slug or "group"


def build_vibrant_palette(count: int) -> List[str]:
    """Return a list of high-contrast, saturated colors for the given count."""

    if count <= 0:
        return []

    palette: List[str] = []
    tableau = list(mcolors.TABLEAU_COLORS.values())
    palette.extend(tableau[: min(count, len(tableau))])

    if len(palette) == count:
        return palette

    remaining = count - len(palette)

    # Spread additional colors around the HSV wheel using the golden ratio to
    # avoid repeating hues while keeping saturation/value high for vibrancy.
    golden_ratio = 0.61803398875
    hue = 0.0
    for idx in range(remaining):
        hue = (hue + golden_ratio) % 1.0
        saturation = 0.78 if idx % 3 == 0 else 0.88
        value = 0.9 if idx % 2 == 0 else 0.82
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        palette.append(mcolors.to_hex((r, g, b)))

    return palette


# ----------------------------- plotting --------------------------------
def render_bubble_plot(
    sub: pd.DataFrame,
    sizes: pd.Series,
    label_series: pd.Series,
    display_labels: List[str],
    color_map: Dict[str, str],
    window_label: Optional[str],
    dpi: int,
    figsize: Tuple[float, float],
    highlight_label: Optional[str] = None,
    outfile: Optional[Path] = None,
) -> Path:
    highlight_colors: Dict[str, str] = {}
    if highlight_label:
        for lab in display_labels:
            if lab == highlight_label:
                if highlight_label == "Other":
                    highlight_colors[lab] = "#8C8C8C"
                else:
                    highlight_colors[lab] = color_map.get(lab, "#1f77b4")
            elif lab == "Other":
                highlight_colors[lab] = "#D3D3D3"
            else:
                highlight_colors[lab] = "#E0E0E0"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("white")

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

    legend_handles: List = []
    legend_labels: List[str] = []
    for lab in display_labels:
        mask = (label_series == lab)
        color = highlight_colors.get(lab, color_map.get(lab, "#D3D3D3"))
        if mask.any():
            scatter = ax.scatter(
                sub.loc[mask, "peak_balance_ui"],
                sub.loc[mask, "net_ui"],
                s=sizes.loc[mask].to_numpy(),
                color=color,
                alpha=0.6,
                edgecolors="black",
                linewidths=0.5,
                label=lab,
            )
            legend_handles.append(scatter)
            legend_labels.append(lab)
        else:
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
            legend_labels.append(lab)

    title = "Address Bubbles — TDCCP"
    if window_label:
        title += f" — {window_label}"
    if highlight_label:
        title += f" — Highlight: {highlight_label}"
    ax.set_title(title, pad=16, fontsize=20)
    ax.set_xlabel("Peak Balance (TDCCP)", fontsize=16)
    ax.set_ylabel("Net Volume (TDCCP)", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)

    ax.legend(
        legend_handles,
        legend_labels,
        title="Address labels (settings.csv)",
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        borderpad=0.6,
        fontsize=12,
        title_fontsize=13,
        scatterpoints=1,
    )

    if outfile:
        out = outfile
    else:
        suffix = f"_{window_label}" if window_label else "_"
        if highlight_label:
            slug = slugify_label(highlight_label)
            out = OUT_DIR / f"Address_Bubbles_byLabel{suffix}_highlight_{slug}.png"
        else:
            out = OUT_DIR / f"Address_Bubbles_byLabel{suffix}.png"

    fig.tight_layout(pad=0.6)
    fig.savefig(out, bbox_inches="tight", dpi=dpi)

    plt.close(fig)
    print(f"[done] {out}")
    return out


def plot_bubbles_by_label(
    df: pd.DataFrame,
    addr_labels: Dict[str, str],
    label_order: List[str],
    window_label: Optional[str],
    outfile: Optional[Path] = None,
    dpi: int = 400,
    figsize: Tuple[float, float] = (24.0, 12.0),

) -> List[Path]:
    """
    Render address bubble plots with colors driven by settings labels.
    Produces the combined view plus per-label highlight variants.
    ``label_order`` should be the ordered unique label list pulled from
    settings.csv so that colors/legend entries remain stable as categories
    expand.
    Returns the list of generated figure paths.
    """

    # Expected columns from metrics:
    # from_address, net_ui, peak_balance_ui, direct_txn_count, intermediary_txn_count, percent_intermediary
    need = ["from_address", "net_ui", "peak_balance_ui", "direct_txn_count"]
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"[error] metrics csv missing column: {c}")

    # Filter to mimic base chart:
    #   - require a positive peak balance (x>0 on log scale)
    #   - require at least one direct swap
    #   - exclude intermediary-heavy addresses if the column is present
    sub = df.copy()
    sub = sub[sub["peak_balance_ui"] > 0]
    sub = sub[sub["direct_txn_count"].fillna(0) > 0]
    if "percent_intermediary" in sub.columns:
        sub = sub[(sub["percent_intermediary"].fillna(0) == 0)]

    if sub.empty:
        raise SystemExit("[error] no rows to plot after filtering (peak>0, direct>0, !intermediary).")

    # Sizes (continuous by direct swaps)
    sizes = compute_sizes(sub["direct_txn_count"])

    # Labels from settings
    label_series = sub["from_address"].map(addr_labels).fillna("Other")

    # Determine display order: 'Other' (if present in data), then each
    # settings-defined label in file order, then any additional labels that
    # slipped through (unlikely but keeps things robust).
    display_labels: List[str] = []
    if (label_series == "Other").any():
        display_labels.append("Other")
    for lab in label_order:
        if lab not in display_labels:
            display_labels.append(lab)
    for lab in label_series.unique():
        if lab not in display_labels:
            display_labels.append(lab)

    # Build a dynamic palette: 'Other' stays light gray, address-label colors
    # draw from a vibrant generator that starts with Tableau's saturated hues
    # and then rotates evenly around the color wheel.  This keeps each label
    # visually distinct even as new groups are added to settings.csv.
    color_map: Dict[str, str] = {"Other": "#D3D3D3"}
    non_other = [lab for lab in display_labels if lab != "Other"]
    if non_other:
        vibrant_colors = build_vibrant_palette(len(non_other))
        color_idx = 0
        for lab in label_order:
            if lab == "Other" or lab not in display_labels:
                continue
            color_map[lab] = vibrant_colors[color_idx % len(vibrant_colors)]
            color_idx += 1
        for lab in display_labels:
            if lab == "Other" or lab in color_map:
                continue
            color_map[lab] = vibrant_colors[color_idx % len(vibrant_colors)]
            color_idx += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs: List[Path] = []
    combined_out = render_bubble_plot(
        sub,
        sizes,
        label_series,
        display_labels,
        color_map,
        window_label,
        dpi,
        figsize,
        highlight_label=None,
        outfile=outfile,
    )
    outputs.append(combined_out)

    for lab in display_labels:
        mask = label_series == lab
        if not mask.any():
            continue
        highlight_out = render_bubble_plot(
            sub,
            sizes,
            label_series,
            display_labels,
            color_map,
            window_label,
            dpi,
            figsize,
            highlight_label=lab,
            outfile=None,
        )
        outputs.append(highlight_out)

    return outputs


# ------------------------------ CLI ------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Plot address bubbles colored by settings.csv address labels (everything else identical to the base chart)."
    )
    ap.add_argument("--metrics", default=str(DEFAULT_METRICS), help="Path to tdccp_address_metrics.csv")
    ap.add_argument("--settings", default=str(SETTINGS), help="Path to settings.csv")
    ap.add_argument("--window-label", default=None, help="Text to append in the title (e.g., 20250301-20250505)")
    ap.add_argument("--outfile", default=None, help="Output PNG path (defaults to outputs/figures/Address_Bubbles_byLabel_*.png)")
    ap.add_argument("--dpi", type=int, default=400, help="Figure DPI (default: 400)")
    ap.add_argument("--figsize", default="24,12", help="Figure size in inches (width,height)")
    # accepted but ignored—keeps pipeline compatibility without altering output
    ap.add_argument("--top-labels", type=int, default=0, help="(ignored) kept only for pipeline compatibility")

    args = ap.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise SystemExit(f"[error] metrics csv not found: {metrics_path}")

    # load metrics
    df = pd.read_csv(metrics_path)

    # load labels
    labels_map, label_order = read_settings_labels(Path(args.settings))

    # window label
    win_lbl = args.window_label
    if not win_lbl:
        s, e = settings_window()
        win_lbl = human_range_label(s, e) or None

    # parse figsize
    try:
        w_str, h_str = (args.figsize.split(",", 1) if "," in args.figsize else args.figsize.split("x", 1))
        figsize = (float(w_str), float(h_str))
    except Exception:
        figsize = (24.0, 12.0)

    outfile = Path(args.outfile) if args.outfile else None
    plot_bubbles_by_label(
        df,
        labels_map,
        label_order,
        win_lbl,
        outfile=outfile,
        dpi=args.dpi,
        figsize=figsize,
    )


if __name__ == "__main__":
    main()

