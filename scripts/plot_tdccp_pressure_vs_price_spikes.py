#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

from plot_tdccp_pressure_vs_price import (
    default_window_from_settings,
    tdccp_mint_from_settings,
    detect_time_column,
    ensure_ts,
    derive_tdccp_delta_ui,
    human_ts_range_label,
    format_ytick_plain,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG_DIR = ROOT / "outputs" / "figures"


def _load_price_history(price_path: Path, debug: bool = False) -> Optional[pd.DataFrame]:
    if not price_path.exists():
        if debug:
            print(f"[price] missing: {price_path}")
        return None
    df = pd.read_csv(price_path)
    if "ts" not in df.columns:
        if debug:
            print(f"[price] 'ts' column missing in {price_path}")
        return None
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    px_col = None
    for cand in ("price_usd", "close", "price"):
        if cand in df.columns:
            px_col = cand
            break
    if not px_col:
        if debug:
            print(f"[price] no price column found in {price_path}")
        return None
    price = df.dropna(subset=["ts", px_col]).sort_values("ts").rename(columns={px_col: "price_usd"})
    if debug:
        print(f"[price] loaded {len(price)} rows from {price_path}")
    return price[["ts", "price_usd"]]


def _bucket_to_timedelta(bucket: str) -> pd.Timedelta:
    try:
        return pd.to_timedelta(bucket)
    except ValueError as exc:
        raise SystemExit(f"[error] unsupported bucket '{bucket}' — use pandas-compatible offsets") from exc


def _load_flows(
    swaps_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    symbol: str,
    debug: bool = False,
) -> pd.DataFrame:
    if not swaps_path.exists():
        sys.exit(f"[error] swaps csv not found: {swaps_path}")
    df = pd.read_csv(swaps_path, low_memory=False)
    if df.empty:
        sys.exit(f"[error] swaps csv is empty: {swaps_path}")

    tcol = detect_time_column(df)
    df["ts"] = ensure_ts(df, tcol)
    window = df[(df["ts"] >= start) & (df["ts"] < end)].copy()
    if debug:
        print(f"[swaps] {len(df)} total rows; window {start}→{end} → {len(window)} rows")

    tdccp_mint = tdccp_mint_from_settings()
    if not tdccp_mint:
        sys.exit("[error] MINT not found in settings.csv (Key= MINT). Needed to identify TDCCP swaps.")

    if "net_tdccp" in window.columns:
        delta = pd.to_numeric(window["net_tdccp"], errors="coerce")
        if debug:
            usable = delta.notna().sum()
            print(f"[flows] using 'net_tdccp' (usable rows: {usable}/{len(delta)})")
    else:
        if debug:
            print("[flows] 'net_tdccp' missing—deriving TDCCP delta from token columns.")
        delta = derive_tdccp_delta_ui(window, tdccp_mint, symbol, debug=debug)

    if "intermediary_label" in window.columns:
        inter = window["intermediary_label"].astype(str).str.lower()
        is_intermed = inter.str.contains("intermed")
    else:
        is_intermed = pd.Series(False, index=window.index)

    flows = window.assign(delta=delta, is_intermed=is_intermed)[["ts", "delta", "is_intermed"]]
    flows = flows.dropna(subset=["delta"])
    if debug:
        print(f"[flows] sample:\n{flows.head(5)}")
    return flows


def _collect_spike_windows(
    metrics_path: Path, min_delta_pct: float, debug: bool = False
) -> list[pd.Timestamp]:
    if not metrics_path.exists():
        sys.exit(f"[error] metrics csv not found: {metrics_path}")
    df = pd.read_csv(metrics_path)
    if "bucket" not in df.columns or "delta_direct_pct" not in df.columns:
        sys.exit(
            "[error] metrics csv missing required columns 'bucket' and 'delta_direct_pct'"
        )
    df["bucket"] = pd.to_datetime(df["bucket"], utc=True, errors="coerce")
    df = df.dropna(subset=["bucket"])
    thresh = abs(min_delta_pct)
    spikes = df.loc[df["delta_direct_pct"] <= -thresh, "bucket"].sort_values()
    if debug:
        print(
            f"[spikes] {len(spikes)} sell-heavy buckets with delta_direct_pct ≤ -{thresh}"
        )
    return list(spikes)


def _plot_bucket_with_spikes(
    bucket: str,
    flows: pd.DataFrame,
    price: Optional[pd.DataFrame],
    label: str,
    outfile_dir: Path,
    spike_starts: Iterable[pd.Timestamp],
    debug: bool = False,
) -> Path:
    if flows.empty:
        if debug:
            print(f"[plot] {bucket}: no flows available → skipping")
        return Path()

    w = flows.set_index("ts")
    buy = w.loc[w["delta"] > 0, "delta"].resample(bucket).sum().fillna(0.0)
    sell = (-w.loc[w["delta"] < 0, "delta"]).resample(bucket).sum().fillna(0.0)
    inter_buy = (
        w.loc[(w["delta"] > 0) & (w["is_intermed"]), "delta"]
        .resample(bucket).sum().fillna(0.0)
    )

    if buy.empty and sell.empty and inter_buy.empty:
        if debug:
            print(f"[plot] {bucket}: no resampled volumes → skipping")
        return Path()

    p = None
    if price is not None and not price.empty:
        p = price.set_index("ts").resample(bucket)["price_usd"].last().dropna()

    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
    ax.plot(buy.index, buy.values, linewidth=1.5, label="Buy volume (TDCCP)", color="green")
    ax.plot(sell.index, sell.values, linewidth=1.5, label="Sell volume (TDCCP)", color="red")
    ax.plot(
        inter_buy.index,
        inter_buy.values,
        linewidth=1.5,
        linestyle="--",
        label="Intermediary buy volume",
        color="blue",
    )

    ax.set_title(f"TDCCP Volume vs Price • Spikes Highlight • {bucket} • {label}")
    ax.set_ylabel("Volume (TDCCP units)")
    ax.yaxis.set_major_formatter(FuncFormatter(format_ytick_plain))
    ax.grid(True, axis="both", linestyle="--", alpha=0.5)

    ax2 = None
    if p is not None and not p.empty:
        ax2 = ax.twinx()
        ax2.plot(p.index, p.values, linewidth=1.5, alpha=1, label="Price (USD)", color="black")
        ax2.set_ylabel("Price (USD)")
        ax2.yaxis.set_major_formatter(FuncFormatter(format_ytick_plain))
        ax2.grid(False)

    spike_starts = list(spike_starts)
    if spike_starts:
        width = _bucket_to_timedelta(bucket)
        ax.relim()
        ax.autoscale_view()
        ymin, ymax = ax.get_ylim()
        if ymin == ymax:
            pad = abs(ymin) if ymin != 0 else 1.0
            ymin -= pad
            ymax += pad
            ax.set_ylim(ymin, ymax)
        for start in spike_starts:
            if pd.isna(start):
                continue
            end = start + width
            span = ax.axvspan(
                start,
                end,
                ymin=0.0,
                ymax=1.0,
                facecolor="none",
                edgecolor="red",
                linewidth=2.0,
            )
            span.set_zorder(ax.lines[0].get_zorder() + 1 if ax.lines else 3)

    if ax2 is not None:
        lines, labels = [], []
        for a in (ax, ax2):
            lns, lbs = a.get_legend_handles_labels()
            lines.extend(lns)
            labels.extend(lbs)
        ax.legend(lines, labels, loc="upper left")
    else:
        ax.legend(loc="upper left")

    fig.autofmt_xdate()
    outfile_dir.mkdir(parents=True, exist_ok=True)
    out = outfile_dir / f"VolumeLines_Price_{bucket}_{label}_spikes.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    if debug:
        print(f"[plot] wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Plot TDCCP volume vs price per bucket with spike windows highlighted in red."
        )
    )
    ap.add_argument("--bucket", required=True, help="Resample bucket size (e.g. 1d,12h,3h,1h,30min)")
    ap.add_argument("--metrics", required=True, help="Path to spike metrics CSV for the bucket")
    ap.add_argument(
        "--min-delta-pct",
        type=float,
        required=True,
        help=(
            "Threshold applied to delta_direct_pct (only buckets with sell-side "
            "imbalance ≤ -threshold are highlighted)"
        ),
    )
    ap.add_argument("--start", help="UTC start YYYY-mm-dd (defaults to settings START)")
    ap.add_argument("--end", help="UTC end YYYY-mm-dd (defaults to settings END)")
    ap.add_argument("--symbol", default="TDCCP", help="Symbol fallback when mint columns show symbols")
    ap.add_argument("--swaps", default=str(DATA / "swaps.csv"), help="Path to swaps.csv")
    ap.add_argument(
        "--price",
        default=str(DATA / "tdccp_price_history.csv"),
        help="Path to TDCCP price history CSV (optional)",
    )
    ap.add_argument(
        "--outfile-dir",
        default=str(FIG_DIR),
        help=f"Output directory for figures (default: {FIG_DIR})",
    )
    ap.add_argument("--debug", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    s_def, e_def = default_window_from_settings()
    start_str = args.start or s_def
    end_str = args.end or e_def
    if not start_str or not end_str:
        sys.exit("[error] --start/--end not provided and START/END missing in settings.csv")

    start = pd.to_datetime(start_str, utc=True, errors="coerce")
    end = pd.to_datetime(end_str, utc=True, errors="coerce")
    if pd.isna(start) or pd.isna(end) or end <= start:
        sys.exit("[error] invalid start/end; ensure ISO YYYY-MM-DD and end > start")

    label = human_ts_range_label(start, end)
    flows = _load_flows(Path(args.swaps), start, end, args.symbol, debug=args.debug)

    price_df = _load_price_history(Path(args.price), debug=args.debug)
    if price_df is not None:
        price_df = price_df[(price_df["ts"] >= start) & (price_df["ts"] < end)]

    spikes = _collect_spike_windows(Path(args.metrics), args.min_delta_pct, debug=args.debug)

    out = _plot_bucket_with_spikes(
        args.bucket,
        flows,
        price_df,
        label,
        Path(args.outfile_dir),
        spikes,
        debug=args.debug,
    )
    if not out or not out.exists():
        sys.exit(
            "[error] spike-highlight figure was not produced.\n"
            "Possible causes: no flows detected, unsupported bucket, or empty metrics."
        )

    print(f"[done] wrote spike-highlight figure: {out}")


if __name__ == "__main__":
    main()
