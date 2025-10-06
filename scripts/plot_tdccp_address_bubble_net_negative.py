#!/usr/bin/env python3
from __future__ import annotations

"""Plot TDCCP address bubbles focusing on large net-negative wallets.

This helper mirrors the base labelled address bubble chart but automatically
collects addresses whose TDCCP sell volume exceeds their buy volume by a
configurable threshold (default: 10k TDCCP).  The qualifying addresses are
assigned a synthetic label so they render together while the underlying bubble
sizing/axes remain identical to the other charts in this repository.
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set

import numpy as np
import pandas as pd

# The labelled bubble plot hosts the rendering logic we want to reuse.  Import
# it dynamically so running this script directly still finds the module.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_tdccp_address_bubble_by_label import (  # noqa: E402
    human_range_label,
    plot_bubbles_by_label,
    settings_window,
)

ROOT = SCRIPT_DIR.parent


DEFAULT_METRICS = Path(
    "/home/moondough/Projects/tdccp_analysis/data/addresses/tdccp_address_metrics.csv"
)
DEFAULT_NEGATIVE = ROOT / "outputs" / "analysis" / "tdccp_negative_net_addresses.csv"
OUT_DIR = ROOT / "outputs" / "figures"
SETTINGS = ROOT / "settings.csv"


def normalize_addresses(addrs: Iterable[object]) -> Set[str]:
    """Return a set of trimmed string addresses, ignoring blanks/NaN."""

    result: Set[str] = set()
    for addr in addrs:
        if isinstance(addr, float) and np.isnan(addr):
            continue
        addr_str = str(addr).strip()
        if addr_str:
            result.add(addr_str)
    return result


def resolve_address_column(columns: Sequence[str]) -> Optional[str]:
    """Return the column name that most likely contains wallet addresses."""

    normalized = [(col, (col or "").strip().lower()) for col in columns if col]
    preferred = [
        "from_address",
        "address",
        "addr",
        "wallet_address",
        "wallet",
        "spike_address",
        "highlight_address",
        "account_address",
        "account",
    ]
    for target in preferred:
        for original, lowered in normalized:
            if lowered == target:
                return original

    for original, lowered in normalized:
        if "address" in lowered and not lowered.startswith("to_"):
            return original
    return None


def addresses_from_negative_csv(path: Path, threshold: float) -> Set[str]:
    """Load addresses whose buy-sell delta is ≤ -threshold from CSV."""

    if not path.exists():
        raise SystemExit(f"[error] negative net csv not found: {path}")

    df = pd.read_csv(path)
    addr_column = resolve_address_column(df.columns)
    if not addr_column:
        raise SystemExit(
            "[error] could not locate an address column in "
            f"{path.name}; available columns: {', '.join(df.columns)}"
        )

    buy = df.get("buy_tdccp")
    sell = df.get("sell_tdccp")
    if buy is not None and sell is not None:
        buy_series = pd.to_numeric(buy, errors="coerce").fillna(0.0)
        sell_series = pd.to_numeric(sell, errors="coerce").fillna(0.0)
        delta = buy_series - sell_series
    elif "net_ui" in df.columns:
        delta = pd.to_numeric(df["net_ui"], errors="coerce")
    elif "buy_minus_sell_tdccp" in df.columns:
        delta = pd.to_numeric(df["buy_minus_sell_tdccp"], errors="coerce")
    else:
        raise SystemExit(
            "[error] negative net csv missing buy/sell or net columns; "
            "expected buy_tdccp & sell_tdccp or net_ui/buy_minus_sell_tdccp"
        )

    magnitude = float(abs(threshold))
    mask = delta <= -magnitude
    if not mask.any():
        return set()

    addrs = df.loc[mask, addr_column].dropna().astype(str).str.strip()
    return normalize_addresses(addrs)


def addresses_from_metrics(df: pd.DataFrame, threshold: float) -> Set[str]:
    """Return addresses whose buy minus sell TDCCP ≤ -threshold."""

    if "buy_tdccp" not in df.columns or "sell_tdccp" not in df.columns:
        return set()

    buy = pd.to_numeric(df["buy_tdccp"], errors="coerce").fillna(0.0)
    sell = pd.to_numeric(df["sell_tdccp"], errors="coerce").fillna(0.0)
    delta = buy - sell
    magnitude = float(abs(threshold))
    mask = delta <= -magnitude
    if not mask.any():
        return set()
    return normalize_addresses(df.loc[mask, "from_address"].astype(str))


def parse_figsize(spec: str) -> tuple[float, float]:
    """Parse a width,height string into a matplotlib-friendly tuple."""

    try:
        if "," in spec:
            width, height = spec.split(",", 1)
        elif "x" in spec.lower():
            width, height = spec.lower().split("x", 1)
        else:
            width, height = spec.split(" ", 1)
        return float(width), float(height)
    except Exception:
        return 24.0, 12.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a TDCCP address bubble chart that spotlights wallets whose "
            "net flow is ≤ -THRESHOLD TDCCP (sell heavy)."
        )
    )
    parser.add_argument(
        "--metrics",
        default=str(DEFAULT_METRICS),
        help="Path to tdccp_address_metrics.csv",
    )
    parser.add_argument(
        "--negative-csv",
        default=str(DEFAULT_NEGATIVE),
        help="Fallback CSV with buy-minus-sell deltas (default: outputs/analysis/tdccp_negative_net_addresses.csv)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10_000.0,
        help=(
            "Minimum net-negative magnitude (buy minus sell ≤ -THRESHOLD) required "
            "to include an address (default: 10000)."
        ),
    )
    parser.add_argument(
        "--window-label",
        default=None,
        help="Optional explicit window label for the chart title.",
    )
    parser.add_argument(
        "--outfile",
        default=None,
        help="Optional output path for the combined chart PNG.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="Figure DPI (default: 400).",
    )
    parser.add_argument(
        "--figsize",
        default="24,12",
        help="Figure size in inches as WIDTH,HEIGHT (default: 24,12).",
    )
    parser.add_argument(
        "--include-others",
        action="store_true",
        help="Keep non-qualifying addresses in the dataset (they remain labelled as 'Other').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise SystemExit(f"[error] metrics csv not found: {metrics_path}")

    metrics = pd.read_csv(metrics_path)
    if "from_address" not in metrics.columns:
        raise SystemExit("[error] metrics csv missing required 'from_address' column")

    metrics["from_address"] = metrics["from_address"].astype(str).str.strip()

    threshold = float(abs(args.threshold))
    if threshold == 0:
        raise SystemExit("[error] threshold must be non-zero")

    if threshold.is_integer():
        threshold_display = f"{int(threshold):,}"
        threshold_slug = str(int(threshold))
    else:
        threshold_display = f"{threshold:,.2f}".rstrip("0").rstrip(".")
        threshold_slug = str(threshold).replace(".", "_")

    qualifying = addresses_from_metrics(metrics, threshold)
    if not qualifying:
        fallback_path = Path(args.negative_csv)
        qualifying = addresses_from_negative_csv(fallback_path, threshold)

    if not qualifying:
        raise SystemExit(
            "[error] no addresses met the sell-minus-buy threshold; "
            "ensure metrics include buy/sell columns or adjust --threshold"
        )

    available = normalize_addresses(metrics["from_address"].tolist())
    qualifying &= available
    if not qualifying:
        raise SystemExit(
            "[error] qualifying addresses were not present in metrics after filtering"
        )

    if not args.include_others:
        metrics = metrics[metrics["from_address"].isin(qualifying)].copy()
        if metrics.empty:
            raise SystemExit(
                "[error] no rows remain after limiting to qualifying addresses"
            )

    label_title = f"Net ≤ -{threshold_display} TDCCP"
    label_map = {addr: label_title for addr in qualifying}
    label_order = [label_title]

    if args.window_label:
        window_label = args.window_label
    else:
        start, end = settings_window()
        window_label = human_range_label(start, end)

    figsize = parse_figsize(args.figsize)

    if args.outfile:
        outfile = Path(args.outfile)
    else:
        suffix = f"_{window_label}" if window_label else ""
        outfile = OUT_DIR / f"Address_Bubbles_netNegative_{threshold_slug}{suffix}.png"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_bubbles_by_label(
        metrics,
        label_map,
        label_order,
        window_label,
        outfile=outfile,
        dpi=args.dpi,
        figsize=figsize,
        highlight_addrs=qualifying,
    )


if __name__ == "__main__":
    main()
