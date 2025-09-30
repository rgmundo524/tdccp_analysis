#!/usr/bin/env python3
# apply_address_attribution.py
#
# Finalize data/swaps.csv schema + insert address_attribution from settings.csv.
# - Normalizes header aliases into the canonical schema used by all downstream scripts
# - Ensures time_iso <-> block_time are both present
# - Keeps any extra columns you already had (activity_type/platform/etc.)
# - Rebuilds tdccp_price_history.csv (same behavior as before)

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SWAPS_CSV = DATA_DIR / "swaps.csv"
SETTINGS_CSV = ROOT / "settings.csv"
POINTS_CSV = DATA_DIR / "tdccp_points.csv"
HISTORY_CSV = DATA_DIR / "tdccp_price_history.csv"

TDCCP_SYMBOL = "TDCCP"  # after apply_symbols.py, router_token{1,2} will often be this symbol

# ------------------------------------------------------------------------------
# Settings helpers (Category=address → {address: label})
# ------------------------------------------------------------------------------
def load_address_labels() -> Dict[str, str]:
    if not SETTINGS_CSV.exists():
        return {}
    # Sniff delimiter, fallback to comma
    try:
        sample = SETTINGS_CSV.read_text(encoding="utf-8", errors="ignore")[:2048]
        dialect = csv.Sniffer().sniff(sample) if sample.strip() else csv.get_dialect("excel")
    except Exception:
        dialect = csv.get_dialect("excel")

    labels: Dict[str, str] = {}
    with SETTINGS_CSV.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        rdr = csv.reader(f, dialect)
        header = next(rdr, None)
        if header:
            lower = [c.strip().lower() for c in header]
            try:
                idx_cat = lower.index("category")
                idx_key = lower.index("key")
                idx_val = lower.index("value")
            except ValueError:
                # Assume first three are Category,Key,Value
                idx_cat, idx_key, idx_val = 0, 1, 2
                # If the "header" was actually data, handle below as a row
                _maybe_row_into_labels(header, idx_cat, idx_key, idx_val, labels)
        else:
            return {}

        for row in rdr:
            _maybe_row_into_labels(row, idx_cat, idx_key, idx_val, labels)

    return labels


def _maybe_row_into_labels(row: List[str], i_cat: int, i_key: int, i_val: int, out: Dict[str, str]):
    if not row or len(row) <= max(i_cat, i_key, i_val):
        return
    cat = (row[i_cat] or "").strip().lower()
    key = (row[i_key] or "").strip()
    val = (row[i_val] or "").strip()
    if cat == "address" and key:
        out[key] = val


def insert_attribution_next_to(df: pd.DataFrame, *, col: str = "from_address") -> pd.DataFrame:
    labels = load_address_labels()
    if "address_attribution" not in df.columns:
        # create blank column
        df.insert(
            loc=max(0, df.columns.get_indexer([col])[0] + 1) if col in df.columns else len(df.columns),
            column="address_attribution",
            value=""
        )
    if col in df.columns:
        # fill labels where available
        df["address_attribution"] = df[col].map(labels).fillna(df.get("address_attribution", ""))
    return df

# ------------------------------------------------------------------------------
# Canonicalization
# ------------------------------------------------------------------------------
CANON = [
    "block_time","time_iso","trans_id","from_address","address_attribution",
    "amount_token_1","router_token1","amount_token_1_usd",
    "amount_token_2","router_token2","amount_token_2_usd",
]

ALIASES = {
    "trans_id": [
        "trans_id","txid","tx_id","signature","transaction_id","sig","tx"
    ],
    "from_address": [
        "from_address","owner","from","wallet","signer","source_address","address"
    ],
    "router_token1": [
        "router_token1","token1_mint","mint1","token_1_mint","token1","in_mint","amount_in_mint"
    ],
    "router_token2": [
        "router_token2","token2_mint","mint2","token_2_mint","token2","out_mint","amount_out_mint"
    ],
    "amount_token_1": [
        "amount_token_1","token1_ui","amount1_ui","ui_token1","token1_amount","amount_token1","raw_token1","amount_in_ui"
    ],
    "amount_token_2": [
        "amount_token_2","token2_ui","amount2_ui","ui_token2","token2_amount","amount_token2","raw_token2","amount_out_ui"
    ],
    "amount_token_1_usd": [
        "amount_token_1_usd","token1_usd","amount1_usd","token1_amount_usd","token_1_usd","amount_in_usd"
    ],
    "amount_token_2_usd": [
        "amount_token_2_usd","token2_usd","amount2_usd","token2_amount_usd","token_2_usd","amount_out_usd"
    ],
    "block_time": [
        "block_time","blockTime","blocktime"
    ],
    "time_iso": [
        "time_iso","time","timestamp","datetime"
    ],
}

def _first_present(cols: List[str], df_cols: pd.Index) -> str | None:
    for c in cols:
        if c in df_cols:
            return c
    return None

def canonicalize_swaps_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) rename aliases → canonical (without clobbering existing canonical)
    rename_map: Dict[str, str] = {}
    for canon, variants in ALIASES.items():
        if canon in df.columns:
            continue
        present = _first_present(variants, df.columns)
        if present and present != canon:
            rename_map[present] = canon
    if rename_map:
        df = df.rename(columns=rename_map)

    # 2) ensure time columns exist and are coherent
    have_block = "block_time" in df.columns
    have_iso   = "time_iso" in df.columns

    if have_block and not have_iso:
        ts = pd.to_datetime(df["block_time"], unit="s", utc=True, errors="coerce")
        df["time_iso"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    elif have_iso and not have_block:
        ts = pd.to_datetime(df["time_iso"], utc=True, errors="coerce")
        df["block_time"] = (ts.view("int64") // 1_000_000_000)

    # 3) enforce dtypes
    for c in ["amount_token_1","amount_token_2","amount_token_1_usd","amount_token_2_usd"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # strip whitespace on id/address/symbol columns
    for c in ["trans_id","from_address","router_token1","router_token2"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # 4) insert attribution next to from_address
    df = insert_attribution_next_to(df, col="from_address")

    # 5) order columns: canonical first (in order), then the rest
    canon_present = [c for c in CANON if c in df.columns]
    extras = [c for c in df.columns if c not in canon_present]
    df = df[canon_present + extras]

    # 6) drop rows without valid timestamp (rare)
    if "time_iso" in df.columns:
        df = df[pd.to_datetime(df["time_iso"], utc=True, errors="coerce").notna()].copy()

    return df

# ------------------------------------------------------------------------------
# Price history (same intent as your prior version)
# ------------------------------------------------------------------------------
def build_history_from_swaps(df: pd.DataFrame) -> pd.DataFrame:
    """Derive per-transaction TDCCP USD price from swaps where USD notional is present."""
    parts = []
    # rows where TDCCP appears as token1 and we have amounts + usd on the *other* side
    is_td1 = df.get("router_token1","").astype(str).eq(TDCCP_SYMBOL)
    if is_td1.any():
        sub = df.loc[is_td1, ["block_time","trans_id","amount_token_1","amount_token_1_usd"]].copy()
        sub = sub[(sub["amount_token_1"].fillna(0) > 0) & sub["amount_token_1_usd"].notna()]
        if not sub.empty:
            sub["ts"] = pd.to_datetime(sub["block_time"], unit="s", utc=True)
            sub["price_usd"] = sub["amount_token_1_usd"].astype(float) / sub["amount_token_1"].astype(float)
            parts.append(sub[["ts","trans_id","price_usd"]])

    is_td2 = df.get("router_token2","").astype(str).eq(TDCCP_SYMBOL)
    if is_td2.any():
        sub = df.loc[is_td2, ["block_time","trans_id","amount_token_2","amount_token_2_usd"]].copy()
        sub = sub[(sub["amount_token_2"].fillna(0) > 0) & sub["amount_token_2_usd"].notna()]
        if not sub.empty:
            sub["ts"] = pd.to_datetime(sub["block_time"], unit="s", utc=True)
            sub["price_usd"] = sub["amount_token_2_usd"].astype(float) / sub["amount_token_2"].astype(float)
            parts.append(sub[["ts","trans_id","price_usd"]])

    if not parts:
        return pd.DataFrame(columns=["ts","trans_id","price_usd"])

    out = pd.concat(parts, ignore_index=True).replace([np.inf,-np.inf], np.nan)
    out = out.dropna(subset=["price_usd","ts"])
    out = out[out["price_usd"] > 0].sort_values("ts")
    return out

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    if not SWAPS_CSV.exists():
        raise SystemExit(f"[error] {SWAPS_CSV} not found. Run the collection step first.")

    # Load whatever the collector produced and make it canonical
    df = pd.read_csv(SWAPS_CSV)
    df = canonicalize_swaps_headers(df)

    # Write back canonical swaps
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SWAPS_CSV, index=False)
    print(f"[done] {SWAPS_CSV}  (rows={len(df)})")

    # Build/refresh TDCCP price history for downstream plots
    if POINTS_CSV.exists():
        hist = pd.read_csv(POINTS_CSV, parse_dates=["ts"])
        hist = hist.sort_values("ts")[["ts","txid","price_usd"]].rename(columns={"txid":"trans_id"})
    else:
        hist = build_history_from_swaps(df)
    hist.to_csv(HISTORY_CSV, index=False)
    print(f"[done] {HISTORY_CSV}  (rows={len(hist)})")

