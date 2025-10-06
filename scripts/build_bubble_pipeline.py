#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SWAPS = DATA / "swaps.csv"
OUT_DEFAULT = ROOT / "data" / "addresses" / "tdccp_address_metrics.csv"
BALANCE_HISTORY_DIR = OUT_DEFAULT.parent
SETTINGS = ROOT / "settings.csv"


# --------------------------- settings helpers ---------------------------

def _read_settings_value(key_name: str) -> Optional[str]:
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
    return _read_settings_value("START"), _read_settings_value("END")


# --------------------------- schema helpers ----------------------------

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_time_column(df: pd.DataFrame) -> str:
    candidates = ["ts", "time_iso", "block_time_iso", "block_time", "timestamp", "datetime", "time"]
    col = pick_col(df, candidates)
    if not col:
        raise SystemExit(
            "[error] cannot find a time column in swaps.csv; "
            f"tried: {', '.join(candidates)}"
        )
    return col


def ensure_ts(df: pd.DataFrame, time_col: str) -> pd.Series:
    # handle seconds epoch if block_time numeric; else ISO/strings
    if pd.api.types.is_numeric_dtype(df[time_col]):
        return pd.to_datetime(df[time_col], unit="s", utc=True, errors="coerce")
    return pd.to_datetime(df[time_col], utc=True, errors="coerce")


# --------------------------- metrics builder ---------------------------

def _format_window_tag(start: pd.Timestamp, end: pd.Timestamp) -> str:
    def _normalize(ts: pd.Timestamp) -> pd.Timestamp:
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    s = _normalize(start).strftime("%Y%m%dT%H%M")
    e = _normalize(end).strftime("%Y%m%dT%H%M")
    return f"{s}-{e}"


def _extract_peak_from_transactions(
    path: Path,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    *,
    debug: bool = False,
) -> Optional[float]:
    """Return the maximum running peak recorded in the aggregated transaction CSV."""

    try:
        tx = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # pragma: no cover - defensive logging
        if debug:
            print(f"[warn] failed to read {path}: {exc}")
        return None

    if tx.empty:
        return None

    time_col = pick_col(tx, ["first_seen", "block_time", "time", "ts", "block_time_iso"])
    if not time_col:
        if debug:
            print(f"[warn] transaction export lacks timestamp column: {path}")
        return None

    ts = pd.to_datetime(tx[time_col], utc=True, errors="coerce")
    tx = tx.loc[pd.notna(ts)].copy()
    if tx.empty:
        return None

    tx["__ts"] = ts
    tx = tx[(tx["__ts"] >= start_utc) & (tx["__ts"] < end_utc)]
    if tx.empty:
        return None

    if "running_peak_ui" in tx.columns:
        series = pd.to_numeric(tx["running_peak_ui"], errors="coerce").dropna()
        if not series.empty:
            return float(series.max())

    candidates = []
    for col in ["running_balance_ui", "post_balance_ui", "pre_balance_ui"]:
        if col in tx.columns:
            candidates.append(pd.to_numeric(tx[col], errors="coerce"))

    if not candidates:
        return None

    combined = pd.concat(candidates, axis=0).dropna()
    if combined.empty:
        return None

    return float(combined.max())


def _extract_peak_from_history(
    path: Path,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    *,
    debug: bool = False,
) -> Optional[float]:
    """Return the maximum balance observed in the raw Solscan history CSV."""

    try:
        hist = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # pragma: no cover - defensive logging
        if debug:
            print(f"[warn] failed to read {path}: {exc}")
        return None

    if hist.empty:
        return None

    time_col = pick_col(hist, ["time", "ts", "block_time_iso", "block_time", "datetime"])
    if not time_col:
        if debug:
            print(f"[warn] balance history lacks timestamp column: {path}")
        return None

    ts = pd.to_datetime(hist[time_col], utc=True, errors="coerce")
    hist = hist.loc[pd.notna(ts)].copy()
    if hist.empty:
        return None

    hist["__ts"] = ts
    hist = hist[(hist["__ts"] >= start_utc) & (hist["__ts"] < end_utc)]
    if hist.empty:
        return None

    cols = []
    if "pre_ui" in hist.columns:
        cols.append(pd.to_numeric(hist["pre_ui"], errors="coerce"))
    if "post_ui" in hist.columns:
        cols.append(pd.to_numeric(hist["post_ui"], errors="coerce"))

    if not cols:
        return None

    combined = pd.concat(cols, axis=0).dropna()
    if combined.empty:
        return None

    return float(combined.max())


def _load_balance_peaks(
    balance_dir: Optional[Path],
    addresses: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    debug: bool = False,
) -> Tuple[Dict[str, float], List[str]]:
    """Return mapping of address → peak balance (UI) sourced from Solscan history.

    The helper also returns a list of addresses for which we could not find a
    usable history file so the caller can surface that gap to the analyst. We do
    **not** fall back to swap-based peaks anymore – the on-chain history is the
    single source of truth for peak balances.
    """

    if balance_dir is None:
        return {}, sorted({a for a in addresses if a})

    if not balance_dir.exists():
        if debug:
            print(f"[warn] balance history directory missing: {balance_dir}")
        return {}, sorted({a for a in addresses if a})

    window_tag = _format_window_tag(start, end)
    peaks: Dict[str, float] = {}
    missing: List[str] = []

    # Normalise timestamps once for filtering.
    start_utc = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end_utc = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")

    for addr in sorted({a for a in addresses if a}):
        history_path = balance_dir / f"{addr}_{window_tag}.csv"
        tx_path = balance_dir / f"{addr}_{window_tag}_transactions.csv"

        peak_val = None
        if tx_path.exists():
            peak_val = _extract_peak_from_transactions(
                tx_path, start_utc, end_utc, debug=debug
            )

        if peak_val is None and history_path.exists():
            peak_val = _extract_peak_from_history(
                history_path, start_utc, end_utc, debug=debug
            )

        if peak_val is None:
            if debug:
                print(
                    f"[debug] unable to derive peak for {addr}; missing or incomplete Solscan history"
                )
            missing.append(addr)
            continue

        if peak_val < 0:
            peak_val = 0.0

        peaks[addr] = max(peaks.get(addr, 0.0), peak_val)

    if debug:
        print(
            f"[info] loaded balance peaks for {len(peaks)}/{len({a for a in addresses if a})} addresses"
        )
        if missing:
            print(f"[warn] missing balance history for {len(missing)} addresses")

    return peaks, sorted(set(missing))


def build_metrics(
    swaps_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    balance_dir: Optional[Path] = None,
    debug: bool = False,
) -> pd.DataFrame:
    if not swaps_path.exists():
        raise SystemExit(f"[error] swaps csv not found: {swaps_path}")

    df = pd.read_csv(swaps_path, low_memory=False)
    if df.empty:
        raise SystemExit(f"[error] swaps csv is empty: {swaps_path}")

    # Require signed TDCCP flow (added by detect_intermediary_swaps step)
    if "net_tdccp" not in df.columns:
        raise SystemExit(
            "[error] swaps.csv is missing 'net_tdccp'. "
            "Run the main pipeline first (scripts/run_tdccp_pipeline.py) "
            "so detect_intermediary_swaps adds net_tdccp."
        )

    # who (from address)
    addr_col = pick_col(df, ["from_address", "owner", "wallet", "signer"])
    if not addr_col:
        raise SystemExit("[error] swaps.csv has no from_address/owner/wallet column.")

    # time
    tcol = detect_time_column(df)
    df["__ts"] = ensure_ts(df, tcol)
    df = df.dropna(subset=["__ts"])

    # window filter
    window = df[(df["__ts"] >= start) & (df["__ts"] < end)].copy()
    if window.empty:
        print("[warn] no swaps in the selected window; metrics will be empty.")
        return pd.DataFrame(columns=[
            "from_address","net_ui","peak_balance_ui","first_seen","last_seen",
            "direct_txn_count","intermediary_txn_count","percent_intermediary"
        ])

    # normalize needed columns
    window["net_tdccp"] = pd.to_numeric(window["net_tdccp"], errors="coerce").fillna(0.0)

    window["__addr"] = window[addr_col].astype(str).str.strip()
    window = window[window["__addr"] != ""].copy()

    addresses = window["__addr"].dropna().unique().tolist()
    balance_peaks, missing_history = _load_balance_peaks(
        balance_dir, addresses, start, end, debug=debug
    )

    # direct vs intermediary
    ilabel = pick_col(window, ["intermediary_label", "route_label", "routing_label"])
    if ilabel is None:
        # treat all as direct if no label column present
        window["__is_direct"] = True
    else:
        s = window[ilabel].astype(str).str.lower()
        # heuristic: consider "routed" or "intermedi" strings as intermediary; else direct
        is_inter = s.str.contains("rout") | s.str.contains("intermed")
        window["__is_direct"] = ~is_inter

    # group + compute
    out_rows = []
    for addr, g in window.sort_values("__ts").groupby("__addr"):
        net_ui = float(g["net_tdccp"].sum())

        hist_peak = balance_peaks.get(addr)
        peak_source = "history" if hist_peak is not None else "missing_history"
        peak = hist_peak if hist_peak is not None else 0.0

        n_total = int(len(g))
        n_direct = int(g["__is_direct"].sum())
        n_inter  = n_total - n_direct
        pct_inter = (n_inter / n_total) if n_total > 0 else 0.0

        first_seen = pd.to_datetime(g["__ts"].min(), utc=True)
        last_seen  = pd.to_datetime(g["__ts"].max(), utc=True)

        out_rows.append({
            "from_address": addr,
            "net_ui": net_ui,
            "peak_balance_ui": peak,
            "peak_balance_source": peak_source,
            "first_seen": first_seen.isoformat(),
            "last_seen":  last_seen.isoformat(),
            "direct_txn_count": n_direct,
            "intermediary_txn_count": n_inter,
            "percent_intermediary": round(pct_inter, 6),
        })

    if missing_history and debug:
        print(
            "[warn] peak balances defaulted to 0.0 for addresses lacking Solscan history:",
            ", ".join(missing_history[:10]) + (" …" if len(missing_history) > 10 else ""),
        )

    metrics = pd.DataFrame(out_rows)
    # Stable ordering: larger peak first, then abs(net), then address
    metrics = metrics.sort_values(
        by=["peak_balance_ui", "net_ui", "from_address"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return metrics


def main():
    ap = argparse.ArgumentParser(
        description="Build per-address metrics for TDCCP bubble charts from data/swaps.csv."
    )
    ap.add_argument(
        "--swaps", default=str(SWAPS),
        help=f"Path to swaps.csv (default: {SWAPS})"
    )
    ap.add_argument(
        "--metrics", default=str(OUT_DEFAULT),
        help=f"Output metrics CSV path (default: {OUT_DEFAULT})"
    )
    ap.add_argument(
        "--balance-history-dir",
        default=str(BALANCE_HISTORY_DIR),
        help=(
            "Directory containing per-address balance history CSVs "
            "(<address>_<start>-<end>.csv). Defaults to data/addresses."
        ),
    )
    ap.add_argument("--start", help="ISO start (YYYY-MM-DD). Defaults to settings START.")
    ap.add_argument("--end",   help="ISO end   (YYYY-MM-DD). Defaults to settings END.")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    s_def, e_def = default_window_from_settings()
    if not args.start and not s_def:
        sys.exit("[error] --start not provided and START missing in settings.csv")
    if not args.end and not e_def:
        sys.exit("[error] --end not provided and END missing in settings.csv")

    start = pd.to_datetime(args.start or s_def, utc=True, errors="coerce")
    end   = pd.to_datetime(args.end or e_def,   utc=True, errors="coerce")
    if pd.isna(start) or pd.isna(end) or end <= start:
        sys.exit("[error] invalid start/end")

    swaps_path   = Path(args.swaps)
    metrics_path = Path(args.metrics)
    balance_dir  = Path(args.balance_history_dir) if args.balance_history_dir else None
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if args.debug:
        print(f"[info] building metrics from {swaps_path}")
        print(f"[info] window {start} → {end}")
        print(f"[info] balance history dir: {balance_dir}")
        print(f"[info] writing to {metrics_path}")

    m = build_metrics(swaps_path, start, end, balance_dir=balance_dir, debug=args.debug)
    m.to_csv(metrics_path, index=False)
    print(f"[done] metrics → {metrics_path}  (rows={len(m)})")


if __name__ == "__main__":
    main()

