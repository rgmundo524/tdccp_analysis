#!/usr/bin/env python3
import pathlib, pandas as pd, numpy as np, re

BASE = pathlib.Path(__file__).resolve().parent.parent
SWAPS_CSV   = BASE / "data/swaps.csv"
POINTS_CSV  = BASE / "data/tdccp_points.csv"   # optional

TDCCP_MINT = "Hg8bKz4mvs8KNj9zew1cEF9tDw1x2GViB4RFZjVEmfrD"

# ---------- duplicate header hygiene ----------
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
# ---------------------------------------------

if __name__=="__main__":
    df = pd.read_csv(SWAPS_CSV)

    is_td1 = df["router_token1"].astype(str).eq(TDCCP_MINT)
    is_td2 = df["router_token2"].astype(str).eq(TDCCP_MINT)

    q1 = is_td1 & df["amount_token_2_usd"].notna() & (df["amount_token_1"].fillna(0)>0)
    q2 = is_td2 & df["amount_token_1_usd"].notna() & (df["amount_token_2"].fillna(0)>0)

    if q1.any(): df.loc[q1,"amount_token_1_usd"] = df.loc[q1,"amount_token_2_usd"]
    if q2.any(): df.loc[q2,"amount_token_2_usd"] = df.loc[q2,"amount_token_1_usd"]

    # optional per-tx price series
    pts=[]
    if q1.any():
        p1=(df.loc[q1,"amount_token_2_usd"].astype(float)/df.loc[q1,"amount_token_1"].astype(float))
        pts.append(pd.DataFrame({"ts":pd.to_datetime(df.loc[q1,"block_time"],unit="s",utc=True),"txid":df.loc[q1,"trans_id"],"price_usd":p1}))
    if q2.any():
        p2=(df.loc[q2,"amount_token_1_usd"].astype(float)/df.loc[q2,"amount_token_2"].astype(float))
        pts.append(pd.DataFrame({"ts":pd.to_datetime(df.loc[q2,"block_time"],unit="s",utc=True),"txid":df.loc[q2,"trans_id"],"price_usd":p2}))
    if pts:
        pd.concat(pts, ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna(subset=["price_usd","ts"]).to_csv(POINTS_CSV, index=False)

    # de-dup & write IN PLACE
    df = _dedupe_columns(df)
    df.to_csv(SWAPS_CSV, index=False)
    print(f"[done] {SWAPS_CSV}")

