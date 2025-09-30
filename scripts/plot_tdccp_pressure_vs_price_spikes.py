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


def _infer_mode_tag(metrics_path: Path, bucket: str) -> Optional[str]:
    """Best-effort extraction of the mode tag from the metrics filename."""
    stem = metrics_path.stem
    prefix = f"spike_buckets_{bucket}_"
    if stem.startswith(prefix):
        remainder = stem[len(prefix) :]
        head, sep, _ = remainder.rpartition("_")
        if sep:
            return head or None
    return None


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
    metrics_path: Path,
    bucket: str,
    *,
    min_delta_pct: float | None = None,
    top_sell_count: int = 0,
    debug: bool = False,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not metrics_path.exists():
        sys.exit(f"[error] metrics csv not found: {metrics_path}")
    df = pd.read_csv(metrics_path)
    if "bucket" not in df.columns or "delta_direct_pct" not in df.columns:
        sys.exit(
            "[error] metrics csv missing required columns 'bucket' and 'delta_direct_pct'"
        )
    df["bucket"] = pd.to_datetime(df["bucket"], utc=True, errors="coerce")
    df = df.dropna(subset=["bucket"]).sort_values("bucket")

    if top_sell_count > 0:
        if "delta_direct" not in df.columns:
            sys.exit(
                "[error] metrics csv missing 'delta_direct' required for --top-sell-count mode"
            )
        sellers = df[df["delta_direct"] < 0].copy()
        sellers = sellers.sort_values(
            ["delta_direct", "sell_direct", "bucket"], ascending=[True, False, True]
        )
        selected = sellers.head(top_sell_count)
        if debug:
            print(
                f"[spikes] using top {len(selected)} sell-heavy buckets by delta_direct (most negative)"
            )
    else:
        thresh = abs(min_delta_pct or 0.0)
        selected = df[df["delta_direct_pct"] <= -thresh]
        if debug:
            print(
                f"[spikes] {len(selected)} sell-heavy buckets with delta_direct_pct ≤ -{thresh}"
            )

    starts = selected["bucket"].dropna().drop_duplicates().tolist()
    starts.sort()
    width = _bucket_to_timedelta(bucket)
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for ts in starts:
        if pd.isna(ts):
            continue
        if not merged:
            merged.append((ts, ts + width))
            continue
        last_start, last_end = merged[-1]
        if ts <= last_end:
            merged[-1] = (last_start, max(last_end, ts + width))
        else:
            merged.append((ts, ts + width))
    return merged


def _plot_bucket_with_spikes(
    bucket: str,
    flows: pd.DataFrame,
    price: Optional[pd.DataFrame],
    label: str,
    mode_tag: Optional[str],
    outfile_dir: Path,
    spike_windows: Iterable[tuple[pd.Timestamp, pd.Timestamp]],
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

    title_bits = ["TDCCP Volume vs Price", "Spikes Highlight", bucket]
    if mode_tag:
        title_bits.append(mode_tag)
    title_bits.append(label)
    ax.set_title(" • ".join(title_bits))
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

    spike_windows = list(spike_windows)
    if spike_windows:
        ax.relim()
        ax.autoscale_view()
        ymin, ymax = ax.get_ylim()
        if ymin == ymax:
            pad = abs(ymin) if ymin != 0 else 1.0
            ymin -= pad
            ymax += pad
            ax.set_ylim(ymin, ymax)
        for start, end in spike_windows:
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
    filename_parts = ["VolumeLines_Price", bucket]
    if mode_tag:
        filename_parts.append(mode_tag)
    filename_parts.append(label)
    out = outfile_dir / ("_".join(filename_parts) + "_spikes.png")
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
        "--mode-tag",
        help=(
            "Optional label describing the spike-selection mode. If omitted, the script attempts "
            "to infer it from the metrics filename."
        ),
    )
    ap.add_argument(
        "--min-delta-pct",
        type=float,
        default=25.0,
        help=(
            "Threshold applied to delta_direct_pct (only buckets with sell-side "
            "imbalance ≤ -threshold are highlighted). Ignored when --top-sell-count > 0."
        ),
    )
    ap.add_argument(
        "--top-sell-count",
        type=int,
        default=0,
        help=(
            "When > 0, highlight the top N sell-heavy buckets by most-negative delta_direct "
            "instead of using --min-delta-pct."
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

    if args.top_sell_count <= 0 and args.min_delta_pct is None:
        sys.exit(
            "[error] provide --min-delta-pct or a positive --top-sell-count to choose spike windows"
        )

    spikes = _collect_spike_windows(
        Path(args.metrics),
        args.bucket,
        min_delta_pct=args.min_delta_pct,
        top_sell_count=args.top_sell_count,
        debug=args.debug,
    )

    mode_tag = args.mode_tag or _infer_mode_tag(Path(args.metrics), args.bucket)
    if args.debug and mode_tag:
        print(f"[spikes] mode tag: {mode_tag}")

    out = _plot_bucket_with_spikes(
        bucket=args.bucket,
        flows=flows,
        price=price_df,
        label=label,
        mode_tag=mode_tag,
        outfile_dir=Path(args.outfile_dir),
        spike_windows=spikes,
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
