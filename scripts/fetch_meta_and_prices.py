#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import pandas as pd
import requests

# --------------------------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SETTINGS_CSV = ROOT / "settings.csv"
DATA_DIR = ROOT / "data"
PRICES_CSV = DATA_DIR / "prices_long.csv"
TOKEN_META_CSV = DATA_DIR / "token_meta.csv"
SWAPS_CSV = DATA_DIR / "swaps.csv"

# CoinGecko Pro (prices)
COINGECKO_PRO_BASE = "https://pro-api.coingecko.com/api/v3"
COINGECKO_KEY_ENV = "COINGECKO_API_KEY"   # must be present in environment
DEFAULT_VS = "usd"
DEFAULT_INTERVAL = "hourly"               # prefer hourly
MAX_DAYS_HOURLY = 31                      # per API docs: <= 31 days per request
MAX_DAYS_DAILY = 180

# Solscan Pro (token metadata)
SOLSCAN_BASE = "https://pro-api.solscan.io/v2.0"
SOLSCAN_KEY_ENV = "SOLSCAN_API_KEY"
SOLSCAN_META_MULTI = f"{SOLSCAN_BASE}/token/meta/multi"
SOLSCAN_BATCH = 20

# --------------------------------------------------------------------------------------
# Settings helpers
# --------------------------------------------------------------------------------------
def _read_settings_value(key_name: str) -> str | None:
    if not SETTINGS_CSV.exists():
        return None
    key_idx, val_idx = 1, 2
    try:
        with SETTINGS_CSV.open("r", encoding="utf-8") as f:
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

def default_start_end() -> tuple[str | None, str | None]:
    return _read_settings_value("START"), _read_settings_value("END")

def core_tdccp_mint() -> str | None:
    return _read_settings_value("MINT")

# --------------------------------------------------------------------------------------
# Date utilities
# --------------------------------------------------------------------------------------
def parse_ymd(s: str) -> datetime:
    dt = pd.to_datetime(s, utc=True)
    if isinstance(dt, pd.Timestamp):
        return dt.to_pydatetime()
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)

def iso_ymd(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")

def chunk_ranges(start: datetime, end: datetime, interval: str) -> List[Tuple[datetime, datetime]]:
    step_days = MAX_DAYS_HOURLY if interval == "hourly" else MAX_DAYS_DAILY
    out: List[Tuple[datetime, datetime]] = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=step_days), end)
        out.append((cur, nxt))
        cur = nxt
    return out

# --------------------------------------------------------------------------------------
# CoinGecko Pro: OHLC range (hourly/daily)
# --------------------------------------------------------------------------------------
def cg_pro_get_ohlc_range(
    coin_id: str,
    vs_currency: str,
    start: datetime,
    end: datetime,
    interval: str,
    api_key: str,
    max_retries: int = 5,
    base_sleep: float = 1.0,
) -> list:
    url = f"{COINGECKO_PRO_BASE}/coins/{coin_id}/ohlc/range"
    params = {
        "vs_currency": vs_currency,
        "from": iso_ymd(start),  # Pro docs: ISO strings are best
        "to": iso_ymd(end),
        "interval": interval,
    }
    headers = {"x-cg-pro-api-key": api_key}

    last_status = None
    for attempt in range(max_retries):
        r = requests.get(url, headers=headers, params=params, timeout=30)
        last_status = r.status_code
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                return data
            raise RuntimeError(f"Unexpected payload for {coin_id}: {data}")
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(base_sleep * (2 ** attempt))
            continue
        r.raise_for_status()
    raise RuntimeError(f"CoinGecko Pro OHLC failed after retries (last={last_status})")

def fetch_ohlc_merged(
    coin_id: str,
    vs_currency: str,
    start: datetime,
    end: datetime,
    interval: str,
    api_key: str,
    pace: float = 0.25,
) -> pd.DataFrame:
    chunks = chunk_ranges(start, end, interval)
    frames: List[pd.DataFrame] = []
    for (a, b) in chunks:
        raw = cg_pro_get_ohlc_range(coin_id, vs_currency, a, b, interval, api_key)
        if raw:
            df = pd.DataFrame(raw, columns=["ms", "open", "high", "low", "close"])
            df["ts"] = pd.to_datetime(df["ms"], unit="ms", utc=True)
            df["coin_id"] = coin_id
            df["vs_currency"] = vs_currency
            df["interval"] = interval
            frames.append(df[["ts", "coin_id", "vs_currency", "interval", "open", "high", "low", "close"]])
        time.sleep(pace)  # gentle pacing
    if not frames:
        return pd.DataFrame(columns=["ts","coin_id","vs_currency","interval","open","high","low","close"])
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["ts","coin_id","interval"]).sort_values("ts")

# --------------------------------------------------------------------------------------
# Solscan Pro: token metadata (multi)
# --------------------------------------------------------------------------------------
def _looks_like_mint(val: str) -> bool:
    # quick, permissive base58-ish heuristic: long & alnum
    s = (val or "").strip()
    return len(s) >= 32 and s.isalnum()

def _collect_mints_from_swaps(swaps_path: Path) -> list[str]:
    if not swaps_path.exists():
        return []
    df = pd.read_csv(swaps_path)
    cols = [c for c in ["router_token1","router_token2"] if c in df.columns]
    if not cols:
        return []
    vals = pd.unique(pd.concat([df[c].astype(str).str.strip() for c in cols], ignore_index=True))
    mints = [v for v in vals if _looks_like_mint(v)]
    # inject TDCCP mint from settings, if present
    tdccp = core_tdccp_mint()
    if tdccp and tdccp not in mints:
        mints.append(tdccp)
    return mints

def _fetch_solscan_meta_batch(addresses: list[str], api_key: str) -> list[dict]:
    if not addresses:
        return []
    # Build query like address[]=mint1&address[]=mint2...
    params = []
    for a in addresses:
        params.append(("address[]", a))
    headers = {"token": api_key}
    r = requests.get(SOLSCAN_META_MULTI, headers=headers, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Solscan meta error {r.status_code}: {r.text}")
    js = r.json()
    if not isinstance(js, dict) or not js.get("success", False):
        raise RuntimeError(f"Solscan meta payload error: {js}")
    return js.get("data", []) or []

def fetch_solscan_meta(mints: list[str], api_key: str, pace: float = 0.15) -> pd.DataFrame:
    if not mints:
        return pd.DataFrame(columns=["mint","symbol","decimals","name","icon"])
    rows: list[dict] = []
    for i in range(0, len(mints), SOLSCAN_BATCH):
        batch = mints[i:i+SOLSCAN_BATCH]
        data = _fetch_solscan_meta_batch(batch, api_key)
        for it in data:
            rows.append({
                "mint": (it.get("address") or "").strip(),
                "symbol": (it.get("symbol") or "").strip().upper(),
                "decimals": it.get("decimals"),
                "name": (it.get("name") or "").strip(),
                "icon": (it.get("icon") or "").strip(),
            })
        time.sleep(pace)
    if not rows:
        return pd.DataFrame(columns=["mint","symbol","decimals","name","icon"])
    df = pd.DataFrame(rows)
    # de-dup by mint, keep last non-null symbol/decimals/name/icon
    df = df.sort_values("mint").drop_duplicates(subset=["mint"], keep="last")
    # enforce dtypes
    df["decimals"] = pd.to_numeric(df["decimals"], errors="coerce").astype("Int64")
    df["symbol"] = df["symbol"].fillna("")
    df["name"] = df["name"].fillna("")
    df["icon"] = df["icon"].fillna("")
    return df[["mint","symbol","decimals","name","icon"]]

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Fetch hourly OHLC via CoinGecko Pro and token metadata via Solscan Pro."
    )
    ap.add_argument("--start", help="ISO start (YYYY-MM-DD or YYYY-MM-DDTHH:MM). Defaults to settings START.")
    ap.add_argument("--end", help="ISO end (YYYY-MM-DD or YYYY-MM-DDTHH:MM). Defaults to settings END.")
    ap.add_argument("--vs", default=DEFAULT_VS, help=f"vs_currency (default: {DEFAULT_VS})")
    ap.add_argument("--interval", default=DEFAULT_INTERVAL, choices=["hourly","daily"], help=f"data interval (default: {DEFAULT_INTERVAL})")
    ap.add_argument("--coins", default="solana", help="Comma-separated CoinGecko coin IDs (default: solana)")
    ap.add_argument("--pace", type=float, default=0.25, help="Sleep seconds between chunk requests (CoinGecko) (default: 0.25)")
    args = ap.parse_args()

    # ENV KEYS (no fallback/renames)
    cg_key = os.environ.get(COINGECKO_KEY_ENV)
    if not cg_key:
        raise SystemExit(f"[error] {COINGECKO_KEY_ENV} not set in environment. Please export it in your shell/.env.")
    solscan_key = os.environ.get(SOLSCAN_KEY_ENV)
    if not solscan_key:
        raise SystemExit(f"[error] {SOLSCAN_KEY_ENV} not set in environment. Please export it in your shell/.env.")

    # Date window
    s_default, e_default = default_start_end()
    if not args.start and not s_default:
        raise SystemExit("[error] --start not provided and START not found in settings.csv")
    if not args.end and not e_default:
        raise SystemExit("[error] --end not provided and END not found in settings.csv")
    start_dt = parse_ymd(args.start or s_default)
    end_dt   = parse_ymd(args.end or e_default)
    if end_dt <= start_dt:
        raise SystemExit("[error] end must be after start")

    # 1) Prices (CoinGecko Pro) — usually Solana only for USD normalization
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    coin_ids = [c.strip() for c in args.coins.split(",") if c.strip()]
    all_frames: List[pd.DataFrame] = []
    for cid in coin_ids:
        print(f"[fetch] {cid} {args.interval} {iso_ymd(start_dt)} → {iso_ymd(end_dt)}")
        df = fetch_ohlc_merged(
            coin_id=cid,
            vs_currency=args.vs,
            start=start_dt,
            end=end_dt,
            interval=args.interval,
            api_key=cg_key,
            pace=args.pace,
        )
        all_frames.append(df)
    prices = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame(
        columns=["ts","coin_id","vs_currency","interval","open","high","low","close"]
    )
    prices.to_csv(PRICES_CSV, index=False)
    print(f"[done] {PRICES_CSV}  (rows={len(prices)})")

    # 2) Token meta (Solscan Pro) — create schema that downstream expects
    mints = _collect_mints_from_swaps(SWAPS_CSV)
    meta_df = fetch_solscan_meta(mints, solscan_key)
    meta_df.to_csv(TOKEN_META_CSV, index=False)
    print(f"[done] {TOKEN_META_CSV}")

if __name__ == "__main__":
    main()

