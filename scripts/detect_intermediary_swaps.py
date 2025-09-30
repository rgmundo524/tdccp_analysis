#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detect intermediary swaps and add signed TDCCP flow.

- Input:  data/swaps.csv
- Output: data/swaps.csv  (in-place; adds columns)
  * intermediary_label: "intermediary" if tx used TDCCP as a hop (both sides in same tx),
                        else "direct"
  * net_tdccp: signed TDCCP flow for the row (+ when TDCCP received as token2, - when sent as token1)
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path

# Canonical paths
ROOT = Path(__file__).resolve().parents[1]
SWAPS = ROOT / "data" / "swaps.csv"

TDCCP_SYMBOL = "TDCCP"  # we operate post-symbol replacement

def _is_tdccp_token1(s: pd.Series) -> pd.Series:
    return (s == TDCCP_SYMBOL)

def _is_tdccp_token2(s: pd.Series) -> pd.Series:
    return (s == TDCCP_SYMBOL)

def main():
    if not SWAPS.exists():
        raise SystemExit(f"[error] {SWAPS} not found. Run the core pipeline first.")

    df = pd.read_csv(SWAPS)

    # Ensure required columns exist
    required = {
        "trans_id",
        "from_address",
        "router_token1",
        "router_token2",
        "amount_token_1",
        "amount_token_2",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"[error] {SWAPS} missing required columns: {sorted(missing)}")

    # Signed net TDCCP for each row
    tdccp_out = _is_tdccp_token1(df["router_token1"])  # user sends TDCCP
    tdccp_in  = _is_tdccp_token2(df["router_token2"])  # user receives TDCCP

    df["net_tdccp"] = 0.0
    # subtract what was sent as TDCCP
    df.loc[tdccp_out, "net_tdccp"] = df.loc[tdccp_out, "amount_token_1"] * (-1.0)
    # add what was received as TDCCP
    df.loc[tdccp_in,  "net_tdccp"] = df.loc[tdccp_in,  "net_tdccp"] + df.loc[tdccp_in, "amount_token_2"]

    # Intermediary detection:
    # A transaction is "intermediary" when, for the same trans_id (and same from_address),
    # we see BOTH sides involving TDCCP (one row with TDCCP as token1 and another as token2).
    # That means TDCCP was only a hop inside a route.
    # We detect at (trans_id, from_address) grain, then broadcast to rows.
    key_cols = ["trans_id", "from_address"]

    # flags per row
    f_out = tdccp_out
    f_in  = tdccp_in

    # build per-key stats
    grp = df.groupby(key_cols)
    has_out = grp["router_token1"].transform(lambda s: (s == TDCCP_SYMBOL).any())
    has_in  = grp["router_token2"].transform(lambda s: (s == TDCCP_SYMBOL).any())

    df["intermediary_label"] = "direct"
    df.loc[(has_in & has_out), "intermediary_label"] = "intermediary"

    # Write back in place
    df.to_csv(SWAPS, index=False)
    print(f"[done] {SWAPS} (added intermediary_label, net_tdccp)")

if __name__ == "__main__":
    main()

