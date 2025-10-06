#!/usr/bin/env python3
"""Augment tdccp_address_metrics.csv with TDCCP USD buy/sell aggregates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_SWAPS = DATA_DIR / "swaps.csv"
DEFAULT_METRICS = DATA_DIR / "addresses" / "tdccp_address_metrics.csv"
SETTINGS = ROOT / "settings.csv"

TDCCP_SYMBOL_DEFAULT = "TDCCP"
TDCCP_MINT_FALLBACK = "Hg8bKz4mvs8KNj9zew1cEF9tDw1x2GViB4RFZjVEmfrD"


# --------------------------- settings helpers ---------------------------
def _read_settings_value(key_name: str) -> Optional[str]:
    if not SETTINGS.exists():
        return None

    key_idx, val_idx = 1, 2
    try:
        with SETTINGS.open("r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header:
                for idx, col in enumerate(header):
                    norm = (col or "").strip().lower()
                    if norm == "key":
                        key_idx = idx
                    elif norm == "value":
                        val_idx = idx

            target = (key_name or "").strip().upper()
            for row in reader:
                if not row or len(row) <= max(key_idx, val_idx):
                    continue
                if (row[key_idx] or "").strip().upper() == target:
                    return (row[val_idx] or "").strip()
    except Exception:
        return None
    return None


def default_mint_from_settings() -> Optional[str]:
    return _read_settings_value("MINT")


# --------------------------- dataframe helpers ---------------------------
def pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def match_token_series(series: pd.Series, *, mint: Optional[str], symbol: Optional[str]) -> pd.Series:
    if series is None:
        raise ValueError("Token series is required to match TDCCP swaps.")

    values = series.astype(str).str.strip()
    mask = pd.Series(False, index=series.index, dtype=bool)

    if symbol:
        mask |= values.str.casefold().eq(str(symbol).casefold())
    if mint:
        mask |= values.str.casefold().eq(str(mint).casefold())

    return mask


# --------------------------- main logic ---------------------------
def augment_metrics(
    swaps_path: Path,
    metrics_path: Path,
    *,
    out_path: Optional[Path] = None,
    mint: Optional[str] = None,
    symbol: str = TDCCP_SYMBOL_DEFAULT,
    debug: bool = False,
) -> Path:
    if not swaps_path.exists():
        raise SystemExit(f"[error] swaps csv not found: {swaps_path}")
    if not metrics_path.exists():
        raise SystemExit(f"[error] metrics csv not found: {metrics_path}")

    out_path = out_path or metrics_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    swaps = pd.read_csv(swaps_path, low_memory=False)
    metrics = pd.read_csv(metrics_path, low_memory=False)

    if swaps.empty:
        raise SystemExit(f"[error] swaps csv is empty: {swaps_path}")
    if metrics.empty:
        raise SystemExit(f"[error] metrics csv is empty: {metrics_path}")

    addr_col_swaps = pick_col(swaps, ["from_address", "owner", "wallet", "signer"])
    if not addr_col_swaps:
        raise SystemExit("[error] swaps.csv missing from_address/owner column")

    addr_col_metrics = pick_col(metrics, ["from_address", "owner", "wallet", "address"])
    if not addr_col_metrics:
        raise SystemExit("[error] metrics csv missing from_address/owner column")

    missing_cols = [
        col
        for col in ("router_token1", "router_token2", "amount_token_1_usd", "amount_token_2_usd")
        if col not in swaps.columns
    ]
    if missing_cols:
        raise SystemExit(
            "[error] swaps.csv missing required column(s): " + ", ".join(missing_cols)
        )

    amt1_col = pick_col(
        swaps,
        [
            "amount_token_1",
            "amount_token1",
            "token1_amount",
            "amount_1",
            "token_a_amount",
        ],
    )
    amt2_col = pick_col(
        swaps,
        [
            "amount_token_2",
            "amount_token2",
            "token2_amount",
            "amount_2",
            "token_b_amount",
        ],
    )

    if not amt1_col or not amt2_col:
        raise SystemExit(
            "[error] swaps.csv missing token amount columns for TDCCP (expected amount_token_1/2)"
        )

    effective_mint = mint or default_mint_from_settings() or TDCCP_MINT_FALLBACK
    if debug:
        print(f"[info] using TDCCP mint: {effective_mint}")
        print(f"[info] using TDCCP symbol: {symbol}")

    addr_series = swaps[addr_col_swaps].astype(str).str.strip()
    buy_mask = match_token_series(swaps["router_token1"], mint=effective_mint, symbol=symbol)
    sell_mask = match_token_series(swaps["router_token2"], mint=effective_mint, symbol=symbol)

    amt1 = pd.to_numeric(swaps["amount_token_1_usd"], errors="coerce").fillna(0.0)
    amt2 = pd.to_numeric(swaps["amount_token_2_usd"], errors="coerce").fillna(0.0)
    amt1_tokens = pd.to_numeric(swaps[amt1_col], errors="coerce").fillna(0.0)
    amt2_tokens = pd.to_numeric(swaps[amt2_col], errors="coerce").fillna(0.0)

    buy_usd = pd.Series(0.0, index=swaps.index, dtype="float64")
    sell_usd = pd.Series(0.0, index=swaps.index, dtype="float64")
    buy_tokens = pd.Series(0.0, index=swaps.index, dtype="float64")
    sell_tokens = pd.Series(0.0, index=swaps.index, dtype="float64")

    buy_usd.loc[buy_mask] = amt1.loc[buy_mask]
    sell_usd.loc[sell_mask] = amt2.loc[sell_mask]
    buy_tokens.loc[buy_mask] = amt1_tokens.loc[buy_mask]
    sell_tokens.loc[sell_mask] = amt2_tokens.loc[sell_mask]

    grouped = pd.DataFrame(
        {
            "buy_usd": buy_usd,
            "sell_usd": sell_usd,
            "buy_tdccp": buy_tokens,
            "sell_tdccp": sell_tokens,
        }
    )
    grouped["__addr"] = addr_series
    grouped = grouped.groupby("__addr", as_index=True).sum()

    metrics = metrics.copy()
    metrics["__addr"] = metrics[addr_col_metrics].astype(str).str.strip()

    metrics["buy_usd"] = metrics["__addr"].map(grouped["buy_usd"]).fillna(0.0)
    metrics["sell_usd"] = metrics["__addr"].map(grouped["sell_usd"]).fillna(0.0)
    metrics["buy_tdccp"] = metrics["__addr"].map(grouped["buy_tdccp"]).fillna(0.0)
    metrics["sell_tdccp"] = metrics["__addr"].map(grouped["sell_tdccp"]).fillna(0.0)
    metrics["buy_minus_sell_usd"] = metrics["buy_usd"] - metrics["sell_usd"]

    metrics.drop(columns=["__addr"], inplace=True, errors="ignore")
    metrics.to_csv(out_path, index=False)

    if debug:
        print(
            "[done] wrote augmented metrics to"
            f" {out_path} (rows={len(metrics)}, columns={len(metrics.columns)})"
        )

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add TDCCP USD buy/sell aggregates to tdccp_address_metrics.csv "
            "using data from swaps.csv."
        )
    )
    parser.add_argument(
        "--swaps",
        default=str(DEFAULT_SWAPS),
        help=f"Path to swaps.csv (default: {DEFAULT_SWAPS})",
    )
    parser.add_argument(
        "--metrics",
        default=str(DEFAULT_METRICS),
        help=f"Path to tdccp_address_metrics.csv (default: {DEFAULT_METRICS})",
    )
    parser.add_argument(
        "--out",
        help=(
            "Optional output path. Defaults to in-place overwrite of the metrics CSV."
        ),
    )
    parser.add_argument(
        "--mint",
        help=(
            "TDCCP mint to match in swaps.csv. Defaults to settings.csv MINT or fallback."
        ),
    )
    parser.add_argument(
        "--symbol",
        default=TDCCP_SYMBOL_DEFAULT,
        help="TDCCP symbol to match in swaps.csv (default: TDCCP)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    swaps_path = Path(args.swaps)
    metrics_path = Path(args.metrics)
    out_path = Path(args.out) if args.out else None

    augment_metrics(
        swaps_path,
        metrics_path,
        out_path=out_path,
        mint=args.mint,
        symbol=args.symbol,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
