#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
PY = sys.executable
SWAPS_CSV = BASE / "data" / "swaps.csv"
OUT_DIR   = BASE / "outputs" / "analysis"

PRESSURE_SPIKES = SCRIPTS_DIR / "plot_tdccp_pressure_vs_price_spikes.py"
TDCCP_SYMBOL = "TDCCP"
TDCCP_MINT   = "Hg8bKz4mvs8KNj9zew1cEF9tDw1x2GViB4RFZjVEmfrD"

def _match_tdccp(s: pd.Series) -> pd.Series:
    sc = s.astype(str).str.strip()
    return sc.str.casefold().eq(TDCCP_SYMBOL.casefold()) | sc.eq(TDCCP_MINT)

def _parse_date(d: str) -> pd.Timestamp:
    return pd.to_datetime(d, utc=True, errors="raise")

def bucketize(ts: pd.Series, bucket: str) -> pd.Series:
    return ts.dt.floor(bucket)

def load_swaps(start: str|None, end: str|None) -> pd.DataFrame:
    if not SWAPS_CSV.exists():
        raise SystemExit("data/swaps.csv not found. Run the core pipeline first.")
    df = pd.read_csv(SWAPS_CSV)
    need = ["time_iso","router_token1","router_token2","amount_token_1","amount_token_2"]
    miss = [c for c in need if c not in df.columns]
    if miss: raise SystemExit(f"swaps.csv missing columns: {miss}")
    df["ts"] = pd.to_datetime(df["time_iso"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).copy()
    if start: df = df[df["ts"] >= _parse_date(start)]
    if end:   df = df[df["ts"]  < _parse_date(end)]
    # keep only swaps where TDCCP appears on either side
    involved = _match_tdccp(df["router_token1"]) | _match_tdccp(df["router_token2"])
    df = df.loc[involved].copy()
    if df.empty: raise SystemExit("No TDCCP-involving swaps in the selected time window.")
    return df


def _sanitize_float_for_tag(value: float) -> str:
    """Return a filesystem-friendly representation of a float."""
    try:
        dec = Decimal(str(value)).normalize()
    except InvalidOperation:
        dec = Decimal(value)
    as_str = format(dec, "f").rstrip("0").rstrip(".")
    if not as_str:
        as_str = "0"
    return as_str.replace("-", "neg").replace(".", "p")

def compute_volumes(df: pd.DataFrame, bucket: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (vol_direct, vol_all) indexed by bucket with buy/sell."""
    tmp = df.copy()
    tmp["bucket"] = bucketize(tmp["ts"], bucket)

    has_label = "intermediary_label" in tmp.columns
    is_inter  = tmp["intermediary_label"].eq("intermediary") if has_label else pd.Series(False, index=tmp.index)
    is_direct = ~is_inter

    buy_mask  = _match_tdccp(tmp["router_token2"])
    sell_mask = _match_tdccp(tmp["router_token1"])

    # ALL rows
    buy_all  = tmp.loc[buy_mask].groupby("bucket")["amount_token_2"].sum(min_count=1)
    sell_all = tmp.loc[sell_mask].groupby("bucket")["amount_token_1"].sum(min_count=1)

    # DIRECT ONLY (intermediaries excluded) — this is what we use for spike logic
    buy_dir  = tmp.loc[buy_mask & is_direct].groupby("bucket")["amount_token_2"].sum(min_count=1)
    sell_dir = tmp.loc[sell_mask & is_direct].groupby("bucket")["amount_token_1"].sum(min_count=1)

    idx = pd.Index(sorted(set(buy_all.index) | set(sell_all.index) | set(buy_dir.index) | set(sell_dir.index)))
    vol_all = pd.DataFrame(index=idx)
    vol_all["buy_all"]  = buy_all.reindex(idx, fill_value=0.0)
    vol_all["sell_all"] = sell_all.reindex(idx, fill_value=0.0)

    vol_dir = pd.DataFrame(index=idx)
    vol_dir["buy_direct"]  = buy_dir.reindex(idx, fill_value=0.0)
    vol_dir["sell_direct"] = sell_dir.reindex(idx, fill_value=0.0)

    return vol_dir, vol_all

def compute_metrics(vol_dir: pd.DataFrame, vol_all: pd.DataFrame, routing_thresh: float) -> pd.DataFrame:
    out = vol_dir.join(vol_all, how="outer").fillna(0.0)
    # totals
    out["total_direct"] = out["buy_direct"] + out["sell_direct"]
    out["delta_direct"] = out["buy_direct"] - out["sell_direct"]
    out["delta_direct_abs"] = out["delta_direct"].abs()
    out["delta_direct_pct"] = np.where(
        out["total_direct"] > 0,
        100.0 * out["delta_direct"] / out["total_direct"],
        0.0
    )

    out["total_all"] = out["buy_all"] + out["sell_all"]
    # routing contribution share = (all - direct)/all
    routing_num = (out["total_all"] - out["total_direct"]).clip(lower=0.0)
    out["routing_share"] = np.where(out["total_all"] > 0, routing_num / out["total_all"], 0.0)
    out["routing_heavy_bucket"] = out["routing_share"] > routing_thresh
    return out

def top_addresses_for_buckets(df: pd.DataFrame, bucket: str, selected_buckets: list[pd.Timestamp], top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (addresses CSV, swaps CSV) for selected buckets (direct rows only)."""
    tmp = df.copy()
    tmp["bucket"] = bucketize(tmp["ts"], bucket)
    has_label = "intermediary_label" in tmp.columns
    is_inter  = tmp["intermediary_label"].eq("intermediary") if has_label else pd.Series(False, index=tmp.index)
    tmp = tmp.loc[~is_inter].copy()  # direct only

    # Within selected buckets, aggregate per from_address
    focus = tmp[tmp["bucket"].isin(selected_buckets)].copy()
    if "from_address" not in focus.columns:
        focus["from_address"] = "(unknown)"

    # address-level direct volumes
    buy_mask  = _match_tdccp(focus["router_token2"])
    sell_mask = _match_tdccp(focus["router_token1"])

    agg = []
    if not focus.empty:
        g = focus.groupby("from_address", dropna=False)
        agg_df = g.agg(
            buy_direct=("amount_token_2", lambda s: s[buy_mask.loc[s.index]].sum() if len(s) else 0.0),
            sell_direct=("amount_token_1", lambda s: s[sell_mask.loc[s.index]].sum() if len(s) else 0.0),
            txn_count=("trans_id", "nunique"),
        ).reset_index()
        agg_df["net_direct"] = agg_df["buy_direct"] - agg_df["sell_direct"]
        agg = agg_df
    addr_df = pd.DataFrame(agg) if isinstance(agg, list) else agg
    addr_top = addr_df.sort_values("net_direct", ascending=False).head(top_n) if not addr_df.empty else addr_df

    # raw swaps in selected buckets
    swaps_out = focus.sort_values(["bucket","ts"])
    return addr_top, swaps_out

def main():
    ap = argparse.ArgumentParser(description="TDCCP spike analysis (direct-only by default).")
    ap.add_argument("--buckets", default="1d,12h,6h,3h,1h",
                    help="Comma-separated list of buckets to analyze (e.g. 1d,12h,6h,3h,1h,30min,10min)")
    ap.add_argument("--start", help="UTC start (YYYY-mm-dd)")
    ap.add_argument("--end",   help="UTC end (YYYY-mm-dd)")
    ap.add_argument(
        "--min-delta-pct",
        type=float,
        default=25.0,
        help=(
            "Minimum sell-side delta_direct_pct magnitude to mark a spike "
            "(filters buckets where delta_direct_pct ≤ -threshold; default 25). "
            "Ignored when --top-sell-count is positive."
        ),
    )
    ap.add_argument(
        "--top-sell-count",
        type=int,
        default=0,
        help=(
            "When > 0, ignore --min-delta-pct and instead select the top N sell-heavy "
            "buckets by most-negative delta_direct for highlighting and downstream exports."
        ),
    )
    ap.add_argument("--top-n", type=int, default=50, help="Top N addresses to include (default 50)")
    ap.add_argument("--routing-thresh", type=float, default=0.25,
                    help="Flag bucket as routing_heavy_bucket if (all-direct)/all > thresh (default 0.25)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_swaps(args.start, args.end)
    if args.debug:
        print(f"[swaps] rows={len(df)} window={df['ts'].min()} → {df['ts'].max()}")

    buckets_in = []
    for b in [x.strip().lower() for x in args.buckets.split(",") if x.strip()]:
        if b in {"1d","12h","6h","3h","1h","30min","10min","30t","10t"}:
            if b == "30t": b = "30min"
            if b == "10t": b = "10min"
            buckets_in.append(b)
    if not buckets_in:
        raise SystemExit("No valid buckets provided.")

    # date tag for filenames
    tag_start = (df["ts"].min().strftime("%Y%m%d"))
    tag_end   = (df["ts"].max().strftime("%Y%m%d"))
    daterange = f"{tag_start}-{tag_end}"

    for bucket in buckets_in:
        vol_dir, vol_all = compute_volumes(df, bucket)
        metrics = compute_metrics(vol_dir, vol_all, args.routing_thresh)

        # pick spike buckets either by threshold or top-N sell deltas
        if args.top_sell_count > 0:
            sellers = metrics[metrics["delta_direct"] < 0].copy()
            sellers = (
                sellers.reset_index(drop=False)
                .rename(columns={"index": "bucket"})
                .sort_values(
                    ["delta_direct", "sell_direct", "bucket"],
                    ascending=[True, False, True],
                )
            )
            sel = sellers.head(args.top_sell_count)
            mode_tag = f"topsell{args.top_sell_count}"
        else:
            sell_thresh = abs(args.min_delta_pct)
            sel = metrics[metrics["delta_direct_pct"] <= -sell_thresh].copy()
            mode_tag = f"mindelta{_sanitize_float_for_tag(sell_thresh)}"
        if args.top_sell_count > 0:
            sel_buckets = list(sel["bucket"])
        else:
            sel_buckets = list(sel.index)
        metrics_out = metrics.assign(selected_spike=metrics.index.isin(sel_buckets))

        # write bucket metrics (includes direct & all + routing_heavy_bucket)
        buckets_path = OUT_DIR / f"spike_buckets_{bucket}_{mode_tag}_{daterange}.csv"
        metrics_out.reset_index(drop=False).rename(columns={"index":"bucket"}).to_csv(
            buckets_path, index=False
        )

        # write bucket metrics (includes direct & all + routing_heavy_bucket)
        buckets_path = OUT_DIR / f"spike_buckets_{bucket}_{daterange}.csv"
        metrics_out.reset_index(drop=False).rename(columns={"index":"bucket"}).to_csv(
            buckets_path, index=False
        )
        # addresses & raw swaps for the selected buckets
        addr_top, swaps_out = top_addresses_for_buckets(df, bucket, sel_buckets, args.top_n)
        addr_path  = OUT_DIR / f"spike_addresses_{bucket}_{mode_tag}_{daterange}.csv"
        swaps_path = OUT_DIR / f"spike_swaps_{bucket}_{mode_tag}_{daterange}.csv"
        addr_top.to_csv(addr_path, index=False)
        swaps_out.to_csv(swaps_path, index=False)

        if args.debug:
            mode = (
                f"top {min(len(sel), args.top_sell_count)} sell" if args.top_sell_count > 0
                else f"≤ -{abs(args.min_delta_pct)} pct"
            )
            print(
                f"[{bucket}] buckets={len(metrics)} spikes={len(sel_buckets)} (mode={mode}) "
                f"addr_top={len(addr_top)} swaps={len(swaps_out)}"
            )
            print(f"[write] {buckets_path.name}, {addr_path.name}, {swaps_path.name}")

        if PRESSURE_SPIKES.exists():
            plot_cmd = [
                PY, str(PRESSURE_SPIKES),
                "--bucket", bucket,
                "--metrics", str(buckets_path),
                "--mode-tag", mode_tag,
            ]
            if args.top_sell_count > 0:
                plot_cmd += ["--top-sell-count", str(args.top_sell_count)]
            else:
                plot_cmd += ["--min-delta-pct", str(args.min_delta_pct)]
            if args.start:
                plot_cmd += ["--start", args.start]
            if args.end:
                plot_cmd += ["--end", args.end]
            if args.debug:
                plot_cmd.append("--debug")
            try:
                subprocess.run(plot_cmd, check=True)
            except subprocess.CalledProcessError as exc:
                raise SystemExit(
                    f"[error] plot_tdccp_pressure_vs_price_spikes failed (bucket={bucket}) → rc={exc.returncode}"
                )
        else:
            if args.debug:
                print(f"[warn] spike highlight script missing: {PRESSURE_SPIKES}")

if __name__ == "__main__":
    main()

