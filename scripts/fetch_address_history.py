#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import requests

# ---------------- paths ----------------
ROOT = Path(__file__).resolve().parents[1]
SETTINGS_CSV = ROOT / "settings.csv"
OUT_ROOT = ROOT / "data" / "addresses"

SOLSCAN_BASE = "https://pro-api.solscan.io/v2.0"

# ---------------- utils ----------------
def _parse_utc(s: str) -> pd.Timestamp:
    """
    Accepts flexible forms like:
      2025-03-05
      20250305
      2025-03-05T14:58
      20250305T1458
    Returns UTC-normalized timestamp.
    """
    ts = pd.to_datetime(s, utc=True, errors="raise")
    return ts

def _to_unix(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).tz_convert("UTC").timestamp())

def read_settings_value(key_name: str) -> Optional[str]:
    if not SETTINGS_CSV.exists():
        return None
    with SETTINGS_CSV.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # default to (Category, Key, Value, ...)
        key_idx, val_idx = 1, 2
        if header:
            for i, col in enumerate(header):
                lc = (col or "").strip().lower()
                if lc == "key":
                    key_idx = i
                elif lc == "value":
                    val_idx = i
        target = key_name.strip().upper()
        for row in reader:
            if not row or len(row) <= max(key_idx, val_idx):
                continue
            if (row[key_idx] or "").strip().upper() == target:
                return (row[val_idx] or "").strip()
    return None

def defaults_from_settings() -> tuple[Optional[str], Optional[str]]:
    # Only these two are read from settings.csv
    return read_settings_value("START"), read_settings_value("END")

def require_api_key() -> str:
    key = os.environ.get("SOLSCAN_API_KEY", "").strip()
    if not key:
        sys.exit("[error] SOLSCAN_API_KEY not found in environment (.env). "
                 "Export it or add to your shell env before running.")
    return key

def http_get_json(url: str, headers: Dict[str, str], params: Dict[str, Any], retries: int = 3, backoff: float = 0.7) -> Dict[str, Any]:
    last = None
    for attempt in range(retries):
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code == 200:
            try:
                data = r.json()
            except Exception as e:
                last = f"invalid JSON: {e}"
                time.sleep(backoff)
                continue
            if isinstance(data, dict) and data.get("success", False):
                return data
            last = f"API error payload: {data}"
        else:
            last = f"HTTP {r.status_code}: {r.text}"
        time.sleep(backoff)
    raise RuntimeError(last or "HTTP/request failed")

# ---------------- Solscan fetchers ----------------
def fetch_token_accounts(owner: str, api_key: str, page_size: int = 40) -> pd.DataFrame:
    """
    GET /account/token-accounts
    Returns columns: token_account, token_address, amount, token_decimals, owner
    """
    headers = {"token": api_key}
    url = f"{SOLSCAN_BASE}/account/token-accounts"

    # The endpoint supports page & page_size (and page_size must be in {10,20,30,40})
    page = 1
    rows: List[Dict[str, Any]] = []
    while True:
        params = {
            "address": owner,
            "type": "token",
            "page": page,
            "page_size": page_size,
            "hide_zero": "false",
        }
        data = http_get_json(url, headers, params)
        items = data.get("data", []) or []
        rows.extend(items)
        # If we got fewer than requested, pagination finished
        if len(items) < page_size:
            break
        page += 1
        # be gentle
        time.sleep(0.15)

    if not rows:
        return pd.DataFrame(columns=["token_account", "token_address", "amount", "token_decimals", "owner"])

    df = pd.DataFrame(rows)
    for c in ["token_account", "token_address", "owner"]:
        if c not in df.columns:
            df[c] = None
        df[c] = df[c].astype(str)

    for c in ["amount", "token_decimals"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    return df[["token_account", "token_address", "amount", "token_decimals", "owner"]]

def fetch_balance_changes(
    owner: str,
    token_account: str,
    token_mint: str,
    start_iso: str,
    end_iso: str,
    api_key: str,
    page_size: int = 40,   # keep within {10,20,30,40} for safety
) -> pd.DataFrame:
    """
    GET /account/balance_change
    Returns columns:
      block_id, block_time, trans_id, address, token_address, token_account,
      token_decimals, amount, pre_balance, post_balance, change_type, fee, time
    """
    headers = {"token": api_key}
    url = f"{SOLSCAN_BASE}/account/balance_change"

    t0 = _parse_utc(start_iso)
    t1 = _parse_utc(end_iso)
    start_ts = _to_unix(t0)
    end_ts   = _to_unix(t1)

    page = 1
    rows: List[Dict[str, Any]] = []
    while True:
        params = {
            "address": owner,
            "token_account": token_account,
            "token": token_mint,
            "from_time": start_ts,
            "to_time": end_ts,
            "page_size": page_size,
            "page": page,
            "remove_spam": "true",
            "sort_by": "block_time",
            "sort_order": "desc",
        }
        data = http_get_json(url, headers, params)
        items = data.get("data", []) or []
        rows.extend(items)
        if len(items) < page_size:
            break
        page += 1
        time.sleep(0.15)

    if not rows:
        return pd.DataFrame(columns=[
            "block_id","block_time","trans_id","address","token_address","token_account",
            "token_decimals","amount","pre_balance","post_balance","change_type","fee","time"
        ])

    df = pd.DataFrame(rows)

    # normalize string columns
    for c in ["trans_id","time","address","token_address","token_account","change_type"]:
        df[c] = df.get(c).astype(str)

    # numeric columns
    for c in ["block_id","block_time","token_decimals","amount","pre_balance","post_balance","fee"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    # UI helpers
    denom = (10 ** df["token_decimals"].fillna(0))
    df["amount_ui"] = df["amount"] / denom
    df["pre_ui"]    = df["pre_balance"] / denom
    df["post_ui"]   = df["post_balance"] / denom

    return df

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(
        description="Fetch per-address TDCCP balance-change history (single owner)."
    )
    ap.add_argument("--owner", required=True, help="Owner (from_address) public key")
    ap.add_argument("--token-mint", required=True, help="Token mint (e.g. TDCCP mint)")
    ap.add_argument("--start", help="UTC start (e.g. 2025-03-01 or 20250301T0000). Defaults to settings START.")
    ap.add_argument("--end",   help="UTC end   (e.g. 2025-05-04 or 20250504T0000). Defaults to settings END.")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if output CSV already exists.")
    ap.add_argument("--verbose", action="store_true", help="Extra logging.")
    args = ap.parse_args()

    api_key = require_api_key()

    s_default, e_default = defaults_from_settings()
    start = (args.start or s_default)
    end   = (args.end   or e_default)
    if not start or not end:
        sys.exit("[error] --start/--end not provided and START/END not found in settings.csv")

    owner = args.owner.strip()
    token_mint = args.token_mint.strip()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Output paths
    accounts_csv = OUT_ROOT / f"{owner}_token_accounts.csv"
    window_tag = f"{_parse_utc(start).strftime('%Y%m%dT%H%M')}-{_parse_utc(end).strftime('%Y%m%dT%H%M')}"
    hist_csv = OUT_ROOT / f"{owner}_{window_tag}.csv"

    # 1) token accounts (always refresh unless skip-existing is set and file exists)
    if args.skip_existing and accounts_csv.exists():
        if args.verbose:
            print(f"[skip] token account map exists → {accounts_csv}")
        acc_df = pd.read_csv(accounts_csv)
    else:
        if args.verbose:
            print(f"[fetch] token accounts for owner={owner}")
        acc_df = fetch_token_accounts(owner, api_key, page_size=40)
        acc_df.to_csv(accounts_csv, index=False)
        if args.verbose:
            print(f"[done] token account map → {accounts_csv}")

    # restrict to this mint
    mint_rows = acc_df.loc[acc_df["token_address"].astype(str) == token_mint]
    if mint_rows.empty:
        # Emit empty history csv to keep downstream robust
        empty_cols = [
            "block_id","block_time","trans_id","address","token_address","token_account",
            "token_decimals","amount","pre_balance","post_balance","change_type","fee","time",
            "amount_ui","pre_ui","post_ui","owner"
        ]
        pd.DataFrame(columns=empty_cols).to_csv(hist_csv, index=False)
        if args.verbose:
            print(f"[info] no token accounts for mint on this owner → {owner}")
            print(f"[done] balance history → {hist_csv}  (rows=0)")
        else:
            print(f"[done] balance history → {hist_csv}  (rows=0)")
        return

    # 2) balance changes across all token accounts for this mint
    all_hist: List[pd.DataFrame] = []
    for _, row in mint_rows.iterrows():
        ta = str(row["token_account"])
        if args.verbose:
            print(f"[fetch] history owner={owner} ta={ta}")
        df = fetch_balance_changes(owner, ta, token_mint, start, end, api_key, page_size=40)
        if not df.empty:
            df["owner"] = owner
        all_hist.append(df)
        time.sleep(0.15)

    hist = pd.concat(all_hist, ignore_index=True) if all_hist else pd.DataFrame()
    if not hist.empty and "time" in hist.columns:
        hist["ts"] = pd.to_datetime(hist["time"], errors="coerce", utc=True)
        hist = hist.sort_values("ts").drop(columns=["ts"])

    hist.to_csv(hist_csv, index=False)
    print(f"[done] balance history → {hist_csv}  (rows={len(hist)})")

if __name__ == "__main__":
    main()

