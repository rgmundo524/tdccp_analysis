#!/usr/bin/env python3
import pathlib
import pandas as pd

BASE        = pathlib.Path(__file__).resolve().parent.parent
META_CSV    = BASE / "data/token_meta.csv"
SWAPS_CSV   = BASE / "data/swaps.csv"

if __name__ == "__main__":
    df   = pd.read_csv(SWAPS_CSV)
    meta = pd.read_csv(META_CSV)

    # Build mapping: mint (address) -> SYMBOL (upper)
    sym_map = dict(zip(meta["mint"].astype(str), meta["symbol"].astype(str).str.upper()))

    # Map only when value matches a known mint address; leave as-is if already a symbol
    df["router_token1"] = df["router_token1"].astype(str).map(lambda v: sym_map.get(v, v))
    df["router_token2"] = df["router_token2"].astype(str).map(lambda v: sym_map.get(v, v))

    df.to_csv(SWAPS_CSV, index=False)
    print(f"[done] {SWAPS_CSV}")

