#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refine swaps:
 - coalesce duplicate columns
 - compute UI amounts from raw + decimals
 - attach USD value for non-TDCCP legs using CoinGecko Pro SOL price (close)
 - write back in place to data/swaps.csv

Assumptions after the pricing change:
 - data/prices_long.csv now comes from CoinGecko Pro and contains:
      ts, coin_id, vs_currency, interval, open, high, low, close
 - We only guarantee SOL pricing is present (coin_id = "solana").
 - Router decimals are taken from router_decimals1/2 when available,
   otherwise default to 9 (previously we also looked up token_meta).
"""

from __future__ import annotations

import pathlib
import re
import numpy as np
import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent.parent
SWAPS_CSV   = BASE / "data" / "swaps.csv"
PRICES_CSV  = BASE / "data" / "prices_long.csv"

# TDCCP & SOL identifiers
TDCCP_MINT  = "Hg8bKz4mvs8KNj9zew1cEF9tDw1x2GViB4RFZjVEmfrD"
SOL_MINT    = "So11111111111111111111111111111111111111112"
WSOL_MINT   = SOL_MINT  # treat WSOL == SOL
SOL_SYMBOLS = {"SOL", "WSOL"}

# ------------------- duplicate header hygiene (same as before) -------------------
def _coalesce_dupe(df: pd.DataFrame, base_name: str) -> None:
    cols = [c for c in df.columns if c == base_name or re.fullmatch(rf"{re.escape(base_name)}\.\d+", c)]
    if len(cols) > 1:
        merged = df[cols].bfill(axis=1).iloc[:, 0]
        for c in cols:
            if c != base_name and c in df.columns:
                df.drop(columns=c, inplace=True, errors="ignore")
        df[base_name] = merged

def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    bases = [
        "block_time","time_iso","trans_id","from_address",
        "router_token1","router_token2",
        "router_amount1_raw","router_amount2_raw",
        "router_decimals1","router_decimals2",
        "amount_token_1","amount_token_2",
        "amount_token_1_usd","amount_token_2_usd",
        "address_attribution",
    ]
    for b in bases:
        if any(c == b or c.startswith(b + ".") for c in df.columns):
            _coalesce_dupe(df, b)
    df = df.loc[:, ~df.columns.duplicated()]
    return df
# -------------------------------------------------------------------------------

def _norm_amount(raw, decimals, default_dec=9) -> float:
    try:
        d = int(decimals) if pd.notna(decimals) else int(default_dec)
        return float(raw) / (10 ** d)
    except Exception:
        return np.nan

def _is_sol_like(token: str) -> bool:
    if not isinstance(token, str):
        return False
    t = token.strip()
    return (
        t == SOL_MINT
        or t == WSOL_MINT
        or t.upper() in SOL_SYMBOLS
    )

def _merge_usd_sol_only(df: pd.DataFrame, prices: pd.DataFrame, mint_col: str, amt_col: str) -> pd.Series:
    """
    Return USD series for the given side:
      - If mint_col is SOL/WSOL (by mint or symbol), price it using CoinGecko 'solana' close.
      - Otherwise NaN (we only guarantee SOL pricing).
    """
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    if prices.empty or "coin_id" not in prices.columns:
        return out

    # Filter CoinGecko prices for SOL
    sol = prices.loc[prices["coin_id"].astype(str).str.lower().eq("solana")].copy()
    if sol.empty:
        return out

    # Use close price as USD
    # prices_long.csv: ts (datetime), open/high/low/close (float)
    if "ts" not in sol.columns or "close" not in sol.columns:
        return out

    sol = sol[["ts","close"]].dropna()
    # Ensure ts is UTC datetime
    sol["ts"] = pd.to_datetime(sol["ts"], utc=True, errors="coerce")
    sol = sol.dropna(subset=["ts"]).sort_values("ts")

    # Pick only the rows on this side that are SOL-like
    side_mask = df[mint_col].astype(str).apply(_is_sol_like)
    if not side_mask.any():
        return out

    left = df.loc[side_mask, ["block_time", amt_col]].rename(columns={"block_time":"ts"})
    if left.empty:
        return out

    left = left.copy()
    # block_time is unix seconds
    left["ts"] = pd.to_datetime(left["ts"], unit="s", utc=True, errors="coerce")
    left = left.dropna(subset=["ts"]).reset_index().rename(columns={"index":"_orig_idx"}).sort_values("ts")

    # asof nearest within 1 hour (hourly data)
    merged = pd.merge_asof(left, sol, on="ts", direction="nearest", tolerance=pd.Timedelta("1h"))

    # backward fallback if nearest failed
    need = merged["close"].isna()
    if need.any():
        fb = pd.merge_asof(
            merged.loc[need, ["ts"]].sort_values("ts"),
            sol, on="ts", direction="backward"
        )
        merged.loc[need, "close"] = fb["close"].values

    usd = merged[amt_col].astype(float) * merged["close"].astype(float)
    out.loc[merged["_orig_idx"]] = usd.values
    return out

if __name__ == "__main__":
    # Load source files
    swaps  = pd.read_csv(SWAPS_CSV)
    prices = pd.read_csv(PRICES_CSV, parse_dates=["ts"])

    # Compute UI amounts from raw & decimals
    swaps["amount_token_1"] = swaps.apply(
        lambda r: _norm_amount(r.get("router_amount1_raw"), r.get("router_decimals1"), default_dec=9),
        axis=1
    )
    swaps["amount_token_2"] = swaps.apply(
        lambda r: _norm_amount(r.get("router_amount2_raw"), r.get("router_decimals2"), default_dec=9),
        axis=1
    )

    # USD attach (only SOL/WSOL on either side, using CoinGecko 'close')
    swaps["amount_token_1_usd"] = _merge_usd_sol_only(swaps, prices, "router_token1", "amount_token_1")
    swaps["amount_token_2_usd"] = _merge_usd_sol_only(swaps, prices, "router_token2", "amount_token_2")

    # De-dup and write in place
    swaps = _dedupe_columns(swaps)
    swaps.to_csv(SWAPS_CSV, index=False)
    print(f"[done] {SWAPS_CSV}")

