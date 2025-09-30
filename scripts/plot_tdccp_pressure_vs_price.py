#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "figures"
SETTINGS = ROOT / "settings.csv"

# --------------------------------------------------------------------------------------
# Settings helpers
# --------------------------------------------------------------------------------------

def read_settings_value(key_name: str) -> Optional[str]:
    """Read a single Value from settings.csv by Key (case-insensitive)."""
    if not SETTINGS.exists():
        return None
    key_idx, val_idx = 1, 2
    try:
        with SETTINGS.open("r", encoding="utf-8") as f:
            rdr = csv.reader(f)
            header = next(rdr, None)
            if header:
                for i, col in enumerate(header):
                    lc = (col or "").strip().lower()
                    if lc == "key":
                        key_idx = i
                    elif lc == "value":
                        val_idx = i
            target = (key_name or "").strip().upper()
            for row in rdr:
                if not row or len(row) <= max(key_idx, val_idx):
                    continue
                if (row[key_idx] or "").strip().upper() == target:
                    return (row[val_idx] or "").strip()
    except Exception:
        return None
    return None


def default_window_from_settings() -> Tuple[Optional[str], Optional[str]]:
    return read_settings_value("START"), read_settings_value("END")


def tdccp_mint_from_settings() -> Optional[str]:
    return read_settings_value("MINT")


# --------------------------------------------------------------------------------------
# Column detection utilities
# --------------------------------------------------------------------------------------

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_time_column(df: pd.DataFrame) -> str:
    candidates = [
        "time_iso", "timestamp", "ts", "block_time_iso", "block_time", "datetime", "time",
    ]
    col = pick_col(df, candidates)
    if not col:
        raise SystemExit(
            f"[error] none of the time columns found: {', '.join(candidates)}"
        )
    return col


def ensure_ts(df: pd.DataFrame, time_col: str) -> pd.Series:
    """Return a UTC datetime64[ns] series from df[time_col]."""
    s = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    return s


def str2num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def detect_inout_schema(df: pd.DataFrame) -> Optional[Dict[str, str]]:
    """
    Detect an in/out schema: token/amount columns for in/out legs.
    Returns mapping with keys: in_mint, out_mint, in_ui, out_ui (best-effort).
    """
    in_mint = pick_col(df, [
        "amount_in_mint", "in_mint", "token_in_mint", "in_token_mint", "mint_in", "tokenInMint"
    ])
    out_mint = pick_col(df, [
        "amount_out_mint", "out_mint", "token_out_mint", "out_token_mint", "mint_out", "tokenOutMint"
    ])
    if not in_mint or not out_mint:
        return None

    in_ui = pick_col(df, ["amount_in_ui", "in_amount_ui", "ui_in_amount"])
    out_ui = pick_col(df, ["amount_out_ui", "out_amount_ui", "ui_out_amount"])

    if in_ui is None:
        in_ui = pick_col(df, ["amount_in", "in_amount", "amountIn"])
    if out_ui is None:
        out_ui = pick_col(df, ["amount_out", "out_amount", "amountOut"])

    if not in_ui or not out_ui:
        return None

    return {"in_mint": in_mint, "out_mint": out_mint, "in_ui": in_ui, "out_ui": out_ui}


def detect_pair_schema(df: pd.DataFrame) -> Optional[Dict[str, str]]:
    """
    Detect a token1/token2 style schema, with either signed deltas or amounts.
    Returns mapping with keys: t1, t2, d1_ui, d2_ui (preferred), else a1_ui, a2_ui.
    """
    t1 = pick_col(df, ["token1_mint", "mint1", "router_token1", "token1", "tokenA", "a_mint"])
    t2 = pick_col(df, ["token2_mint", "mint2", "router_token2", "token2", "tokenB", "b_mint"])
    if not t1 or not t2:
        return None

    # Prefer signed deltas (UI)
    d1 = pick_col(df, ["token1_delta_ui", "delta1_ui", "ui_delta_token1", "token1_ui_delta"])
    d2 = pick_col(df, ["token2_delta_ui", "delta2_ui", "ui_delta_token2", "token2_ui_delta"])
    if d1 and d2:
        return {"t1": t1, "t2": t2, "d1_ui": d1, "d2_ui": d2}

    # Fall back to unsigned amounts (UI)
    a1 = pick_col(df, ["token1_amount_ui", "amount1_ui", "ui_token1", "token1_ui"])
    a2 = pick_col(df, ["token2_amount_ui", "amount2_ui", "ui_token2", "token2_ui"])
    if a1 and a2:
        return {"t1": t1, "t2": t2, "a1_ui": a1, "a2_ui": a2}

    # Fall back to raw amounts (last resort)
    a1r = pick_col(df, ["token1_amount", "amount1", "raw_token1"])
    a2r = pick_col(df, ["token2_amount", "amount2", "raw_token2"])
    if a1r and a2r:
        return {"t1": t1, "t2": t2, "a1_ui": a1r, "a2_ui": a2r}

    return None


# --------------------------------------------------------------------------------------
# Delta derivation (fallback only if 'net_tdccp' missing)
# --------------------------------------------------------------------------------------

def match_token_series(series: pd.Series, target_mint: str, symbol: str = "TDCCP") -> pd.Series:
    s = series.astype(str)
    return (s == target_mint) | (s.str.upper() == symbol.upper())


def derive_tdccp_delta_ui(
    df: pd.DataFrame,
    tdccp_mint: str,
    tdccp_symbol: str = "TDCCP",
    debug: bool = False,
) -> pd.Series:
    """
    Returns a Series: +UI for TDCCP bought, -UI for TDCCP sold.
    Rows where sign cannot be determined will return NaN (and be ignored).
    """
    # in/out first
    io = detect_inout_schema(df)
    if io:
        if debug:
            print(f"[detect] using in/out schema: {io}")
        in_is_tdccp = match_token_series(df[io["in_mint"]], tdccp_mint, tdccp_symbol)
        out_is_tdccp = match_token_series(df[io["out_mint"]], tdccp_mint, tdccp_symbol)

        in_amt = str2num(df[io["in_ui"]])
        out_amt = str2num(df[io["out_ui"]])

        delta = pd.Series(np.nan, index=df.index, dtype=float)
        delta[out_is_tdccp] = out_amt[out_is_tdccp]     # acquired TDCCP → +out
        delta[in_is_tdccp] = -in_amt[in_is_tdccp]       # gave TDCCP     → -in
        return delta

    # token1/token2 schema
    pair = detect_pair_schema(df)
    if pair:
        if debug:
            print(f"[detect] using pair schema: {pair}")
        t1_is = match_token_series(df[pair["t1"]], tdccp_mint, tdccp_symbol)
        t2_is = match_token_series(df[pair["t2"]], tdccp_mint, tdccp_symbol)

        if "d1_ui" in pair and "d2_ui" in pair:
            d1 = str2num(df[pair["d1_ui"]])
            d2 = str2num(df[pair["d2_ui"]])
            delta = pd.Series(np.nan, index=df.index, dtype=float)
            delta[t1_is] = d1[t1_is]
            delta[t2_is] = d2[t2_is]
            return delta

        if debug:
            print("[warn] only unsigned amounts; cannot infer buy vs sell; skipping those rows.")
        return pd.Series(np.nan, index=df.index, dtype=float)

    if debug:
        print("[warn] no recognized schema; cannot compute TDCCP delta. All rows will be ignored.")
    return pd.Series(np.nan, index=df.index, dtype=float)


# --------------------------------------------------------------------------------------
# Price loading
# --------------------------------------------------------------------------------------

def load_price_history(debug: bool = False) -> Optional[pd.DataFrame]:
    """
    Load data/tdccp_price_history.csv if present.
    Expected columns: ts (datetime), price_usd or close or price.
    """
    price_file = DATA / "tdccp_price_history.csv"
    if not price_file.exists():
        if debug:
            print(f"[price] missing: {price_file}")
        return None
    ph = pd.read_csv(price_file)
    if "ts" not in ph.columns:
        if debug:
            print(f"[price] 'ts' column missing in {price_file}")
        return None
    ph["ts"] = pd.to_datetime(ph["ts"], utc=True, errors="coerce")
    px_col = pick_col(ph, ["price_usd", "close", "price"])
    if not px_col:
        if debug:
            print(f"[price] price column missing in {price_file}")
        return None
    ph = ph.dropna(subset=["ts", px_col]).sort_values("ts").rename(columns={px_col: "price_usd"})
    if debug:
        print(f"[price] loaded {len(ph)} rows from {price_file}")
    return ph[["ts", "price_usd"]]


# --------------------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------------------

def human_ts_range_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"


def format_ytick_plain(x, pos):
    try:
        if abs(x) >= 1_000_000_000:
            return f"{x/1_000_000_000:.1f}B"
        if abs(x) >= 1_000_000:
            return f"{x/1_000_000:.1f}M"
        if abs(x) >= 1_000:
            return f"{x/1_000:.1f}k"
        if x == int(x):
            return f"{int(x)}"
        return f"{x:.2f}"
    except Exception:
        return str(x)


def plot_one_bucket_lines(
    bucket: str,
    window: pd.DataFrame,          # columns: ts, delta, is_intermed
    price: Optional[pd.DataFrame], # columns: ts, price_usd (optional)
    label: str,
    outfile_dir: Path,
    debug: bool = False,
) -> Path:
    """
    Render a single figure of line volumes (buy, sell, intermediary buy) vs price for a bucket.
    - Buy line: sum of positive delta
    - Sell line: sum of absolute(-delta) for negative delta
    - Intermediary buy line: sum of positive delta where is_intermed == True
    """
    if window.empty:
        return Path()

    w = window.set_index("ts")

    # Aggregations
    buy = w.loc[w["delta"] > 0, "delta"].resample(bucket).sum().fillna(0.0)
    sell = (-w.loc[w["delta"] < 0, "delta"]).resample(bucket).sum().fillna(0.0)  # positive magnitude
    inter_buy = (
        w.loc[(w["delta"] > 0) & (w["is_intermed"]), "delta"]
        .resample(bucket).sum().fillna(0.0)
    )

    if buy.empty and sell.empty and inter_buy.empty:
        if debug:
            print(f"[plot] {bucket}: no volume after resample → skipping")
        return Path()

    # Price aligned to same freq (last within bucket)
    p = None
    if price is not None and not price.empty:
        p = price.set_index("ts").resample(bucket)["price_usd"].last().dropna()

    # Figure
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)

    # Lines
    ax.plot(buy.index, buy.values, linewidth=1.5, label="Buy volume (TDCCP)", color="green")
    ax.plot(sell.index, sell.values, linewidth=1.5, label="Sell volume (TDCCP)", color="red")
    ax.plot(inter_buy.index, inter_buy.values, linewidth=1.5, linestyle="--", label="Intermediary buy volume", color="blue")

    ax.set_title(f"TDCCP Volume vs Price  •  {bucket}  •  {label}")
    ax.set_ylabel("Volume (TDCCP units)")
    ax.yaxis.set_major_formatter(FuncFormatter(format_ytick_plain))
    ax.grid(True, axis="both", linestyle="--", alpha=0.5)

    # Secondary axis for price
    if p is not None and not p.empty:
        ax2 = ax.twinx()
        ax2.plot(p.index, p.values, linewidth=1.5, alpha=1, label="Price (USD)", color="black")
        ax2.set_ylabel("Price (USD)")
        ax2.yaxis.set_major_formatter(FuncFormatter(format_ytick_plain))
        ax2.grid(False)

        # Combined legend
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
    out = outfile_dir / f"VolumeLines_Price_{bucket}_{label}.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    if debug:
        print(f"[plot] wrote {out}")
    return out


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Plot TDCCP buy/sell/intermediary-buy volumes (lines) vs price, across time buckets."
    )
    ap.add_argument("--start", help="ISO date YYYY-MM-DD (default: settings START)")
    ap.add_argument("--end", help="ISO date YYYY-MM-DD (default: settings END)")
    ap.add_argument(
        "--buckets",
        default="1d,12h,6h,1h,30min,10min",
        help="Comma-separated resample buckets (default: 1d,12h,6h,1h,30min,10min)",
    )
    ap.add_argument(
        "--symbol",
        default="TDCCP",
        help="TDCCP symbol to match when IDs are symbols (default: TDCCP)",
    )
    ap.add_argument(
        "--swaps",
        default=str(DATA / "swaps.csv"),
        help="Path to swaps.csv (default: data/swaps.csv)",
    )
    ap.add_argument(
        "--price",
        default=str(DATA / "tdccp_price_history.csv"),
        help="Path to TDCCP price history CSV (default: data/tdccp_price_history.csv). If missing, no price is plotted.",
    )
    ap.add_argument(
        "--outfile-dir",
        default=str(OUT_DIR),
        help=f"Output directory for figures (default: {OUT_DIR})",
    )
    ap.add_argument("--debug", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    # Settings defaults (start/end + MINT)
    s_def, e_def = default_window_from_settings()
    if not args.start and not s_def:
        sys.exit("[error] --start not provided and START missing in settings.csv")
    if not args.end and not e_def:
        sys.exit("[error] --end not provided and END missing in settings.csv")

    start = pd.to_datetime(args.start or s_def, utc=True, errors="coerce")
    end = pd.to_datetime(args.end or e_def, utc=True, errors="coerce")
    if pd.isna(start) or pd.isna(end) or end <= start:
        sys.exit("[error] invalid start/end; ensure ISO YYYY-MM-DD and end > start")

    label = human_ts_range_label(start, end)

    # TDCCP mint (for fallback derivation if needed)
    tdccp_mint = tdccp_mint_from_settings()
    if not tdccp_mint:
        sys.exit("[error] MINT not found in settings.csv (Key= MINT). Needed to identify TDCCP swaps.")

    # Load swaps
    swaps_path = Path(args.swaps)
    if not swaps_path.exists():
        sys.exit(f"[error] swaps csv not found: {swaps_path}")
    df = pd.read_csv(swaps_path, low_memory=False)
    if df.empty:
        sys.exit(f"[error] swaps csv is empty: {swaps_path}")

    # Time filter
    tcol = detect_time_column(df)
    df["ts"] = ensure_ts(df, tcol)
    window = df[(df["ts"] >= start) & (df["ts"] < end)].copy()
    if args.debug:
        print(f"[swaps] {len(df)} total rows; window {start}→{end} → {len(window)} rows")
        print(f"[swaps] columns: {list(df.columns)}")

    if window.empty:
        print("[warn] No swaps in selected window; figures will be empty.")

    # Signed TDCCP delta (prefer pipeline's 'net_tdccp')
    if "net_tdccp" in window.columns:
        delta = pd.to_numeric(window["net_tdccp"], errors="coerce")
        if args.debug:
            used = delta.notna().sum()
            print(f"[flows] using 'net_tdccp' (usable rows: {used}/{len(delta)})")
    else:
        if args.debug:
            print("[flows] 'net_tdccp' missing—deriving TDCCP delta from token columns.")
        delta = derive_tdccp_delta_ui(window, tdccp_mint, args.symbol, debug=args.debug)

    # Intermediary label flag (best-effort)
    if "intermediary_label" in window.columns:
        inter = window["intermediary_label"].astype(str).str.lower()
        is_intermed = inter.str.contains("intermed")  # matches 'intermediary'
    else:
        is_intermed = pd.Series(False, index=window.index)

    # Prepare narrow frame for plotting
    flows = window.assign(delta=delta, is_intermed=is_intermed)[["ts", "delta", "is_intermed"]]
    flows = flows.dropna(subset=["delta"])
    if args.debug:
        print(f"[flows] sample:\n{flows.head(5)}")

    # Load price (optional)
    price_df = load_price_history(debug=args.debug)
    if price_df is not None:
        price_df = price_df[(price_df["ts"] >= start) & (price_df["ts"] < end)]

    # Plot per bucket
    buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]
    OUT = Path(args.outfile_dir)
    written = []
    for b in buckets:
        out = plot_one_bucket_lines(b, flows, price_df, label=label, outfile_dir=OUT, debug=args.debug)
        if out and out.exists():
            written.append(out)

    if not written:
        sys.exit(
            "[error] No figures were written.\n"
            "Possible causes:\n"
            "  • No TDCCP-involving swaps with determinable sign in the selected window\n"
            "  • Price history missing or empty (the price line is optional)\n"
            "  • Bucket selection resulted in empty resamples\n"
            "Try:  --debug  and verify detected columns, or confirm settings MINT & the swaps schema."
        )
    else:
        print(f"[done] wrote {len(written)} figures to {OUT}")


if __name__ == "__main__":
    main()

