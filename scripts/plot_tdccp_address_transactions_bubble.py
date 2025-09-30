#!/usr/bin/env python3
"""Plot a per-transaction TDCCP bubble chart for a single address."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# Re-use the Solscan helpers from fetch_address_history
from fetch_address_history import (  # type: ignore
    defaults_from_settings as history_window_defaults,
    fetch_balance_changes,
    fetch_token_accounts,
    require_api_key,
)

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_CSV = ROOT / "settings.csv"
DATA_DIR = ROOT / "data"
SWAPS_CSV = DATA_DIR / "swaps.csv"
PRICE_HISTORY_CSV = DATA_DIR / "tdccp_price_history.csv"
OUT_FIG_DIR = ROOT / "outputs" / "figures"
OUT_TX_DIR = DATA_DIR / "addresses"

TDCCP_MINT_FALLBACK = "Hg8bKz4mvs8KNj9zew1cEF9tDw1x2GViB4RFZjVEmfrD"
TDCCP_SYMBOL = "TDCCP"
_MINT_CACHE: Optional[str] = None


# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------

def read_settings_value(key_name: str) -> Optional[str]:
    if not SETTINGS_CSV.exists():
        return None
    target = (key_name or "").strip().upper()
    key_idx, val_idx = 1, 2
    with SETTINGS_CSV.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header:
            for i, col in enumerate(header):
                lc = (col or "").strip().lower()
                if lc == "key":
                    key_idx = i
                elif lc == "value":
                    val_idx = i
        for row in reader:
            if not row or len(row) <= max(key_idx, val_idx):
                continue
            if (row[key_idx] or "").strip().upper() == target:
                return (row[val_idx] or "").strip()
    return None


def window_defaults() -> Tuple[Optional[str], Optional[str]]:
    start, end = history_window_defaults()
    return start, end


def mint_default() -> str:
    global _MINT_CACHE
    if _MINT_CACHE is not None:
        return _MINT_CACHE
    mint = read_settings_value("MINT")
    _MINT_CACHE = mint or TDCCP_MINT_FALLBACK
    return _MINT_CACHE


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------


def load_price_history(path: Path, *, verbose: bool = False) -> Optional[pd.DataFrame]:
    if not path.exists():
        if verbose:
            print(f"[price] missing → {path}")
        return None
    df = pd.read_csv(path)
    if "ts" not in df.columns:
        if verbose:
            print(f"[price] 'ts' column missing → {path}")
        return None
    ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    price_col = None
    for cand in ["price_usd", "close", "price"]:
        if cand in df.columns:
            price_col = cand
            break
    if price_col is None:
        if verbose:
            print(f"[price] price column missing in {path}")
        return None
    clean = (
        pd.DataFrame({"ts": ts, "price_usd": pd.to_numeric(df[price_col], errors="coerce")})
        .dropna(subset=["ts", "price_usd"])
        .sort_values("ts")
    )
    if clean.empty and verbose:
        print(f"[price] no valid rows in {path}")
    return clean if not clean.empty else None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SwapClassification:
    label: str
    router_token1: Optional[str] = None
    router_token2: Optional[str] = None
    intermediary_label: Optional[str] = None


def is_tdccp_token(value: str | float | int | None) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    v = str(value).strip()
    if not v:
        return False
    return v.upper() in {TDCCP_SYMBOL, TDCCP_MINT_FALLBACK.upper(), mint_default().upper()}


def load_swaps_for_owner(owner: str, swaps_path: Path, *, verbose: bool = False) -> pd.DataFrame:
    if not swaps_path.exists():
        if verbose:
            print(f"[info] swaps.csv not found at {swaps_path}; treating all txns as transfers")
        return pd.DataFrame()
    df = pd.read_csv(swaps_path, low_memory=False)
    if df.empty:
        return df
    if "from_address" not in df.columns:
        if verbose:
            print(f"[warn] swaps.csv missing 'from_address'; cannot match transactions")
        return pd.DataFrame()
    df["from_address"] = df["from_address"].astype(str)
    owner_rows = df[df["from_address"] == owner].copy()
    if owner_rows.empty and verbose:
        print(f"[info] swaps.csv has no rows for owner {owner}")
    return owner_rows


def classify_from_swaps(tx_hash: str, swaps: pd.DataFrame) -> Optional[SwapClassification]:
    if swaps.empty:
        return None
    if "trans_id" not in swaps.columns:
        return None
    matches = swaps[swaps["trans_id"].astype(str) == tx_hash]
    if matches.empty:
        return None
    if "intermediary_label" in matches.columns:
        direct = matches[matches["intermediary_label"].astype(str).str.lower() == "direct"]
        if not direct.empty:
            matches = direct
    # Check token sides
    rt1 = matches.get("router_token1")
    rt2 = matches.get("router_token2")
    lbl = matches.get("intermediary_label")
    first_lbl = (lbl.iloc[0] if lbl is not None and not lbl.empty else None)
    if rt1 is not None and rt1.astype(str).apply(is_tdccp_token).any():
        token1 = rt1.iloc[0]
        token2 = rt2.iloc[0] if rt2 is not None and not rt2.empty else None
        return SwapClassification("buy", token1, token2, first_lbl)
    if rt2 is not None and rt2.astype(str).apply(is_tdccp_token).any():
        token1 = rt1.iloc[0] if rt1 is not None and not rt1.empty else None
        token2 = rt2.iloc[0]
        return SwapClassification("sell", token1, token2, first_lbl)
    return SwapClassification("transfer", rt1.iloc[0] if rt1 is not None and not rt1.empty else None,
                              rt2.iloc[0] if rt2 is not None and not rt2.empty else None,
                              first_lbl)


def detect_airdrop(amount: float, classification: Optional[SwapClassification], *, change_direction: str) -> bool:
    if classification is not None and classification.label in {"buy", "sell"}:
        return False
    if change_direction.lower() != "inc":
        return False
    return math.isclose(amount, 777.0, rel_tol=1e-6, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def compute_bubble_sizes(amounts: pd.Series) -> np.ndarray:
    base = amounts.abs().clip(lower=1e-4)
    return np.sqrt(base) * 80.0


def legend_handles(color_map: Dict[str, str]) -> Iterable[Line2D]:
    for label, color in color_map.items():
        yield Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=12,
            label=label.title(),
        )


def fmt_tdccp(v, _pos):
    try:
        if abs(v) >= 1_000_000:
            return f"{v/1_000_000:.1f}M"
        if abs(v) >= 1_000:
            return f"{v/1_000:.1f}k"
        if abs(v - int(v)) < 1e-9:
            return f"{int(v)}"
        return f"{v:.2f}"
    except Exception:
        return str(v)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_figsize(text: str) -> Tuple[float, float]:
    try:
        if "x" in text:
            w, h = text.split("x", 1)
        elif "," in text:
            w, h = text.split(",", 1)
        else:
            raise ValueError
        return float(w), float(h)
    except Exception as exc:
        raise SystemExit("[error] --figsize must be WIDTHxHEIGHT or WIDTH,HEIGHT (e.g. 20,10)") from exc


def human_window_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"


def gather_transactions(
    owner: str,
    mint: str,
    start_iso: str,
    end_iso: str,
    api_key: str,
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    accounts_df = fetch_token_accounts(owner, api_key, page_size=40)
    if accounts_df.empty:
        if verbose:
            print(f"[warn] no token accounts returned for owner {owner}")
        return pd.DataFrame()
    mint_rows = accounts_df[accounts_df["token_address"].astype(str) == mint]
    if mint_rows.empty:
        if verbose:
            print(f"[warn] owner {owner} has no token account for mint {mint}")
        return pd.DataFrame()
    all_hist: list[pd.DataFrame] = []
    for _, row in mint_rows.iterrows():
        token_account = str(row["token_account"])
        if verbose:
            print(f"[fetch] balance changes owner={owner} token_account={token_account}")
        hist = fetch_balance_changes(owner, token_account, mint, start_iso, end_iso, api_key, page_size=40)
        if hist.empty:
            continue
        hist["token_account"] = token_account
        all_hist.append(hist)
    if not all_hist:
        return pd.DataFrame()
    hist_df = pd.concat(all_hist, ignore_index=True)
    hist_df["time_iso"] = pd.to_datetime(hist_df.get("time"), utc=True, errors="coerce")
    hist_df["direction"] = hist_df.get("change_type", "").astype(str)
    hist_df["direction"] = hist_df["direction"].str.lower().where(hist_df["direction"].notna(), "")
    hist_df["amount_ui"] = pd.to_numeric(hist_df.get("amount_ui"), errors="coerce").fillna(0.0)
    hist_df["signed_amount_ui"] = np.where(
        hist_df["direction"] == "inc",
        hist_df["amount_ui"],
        -hist_df["amount_ui"],
    )
    hist_df = hist_df.dropna(subset=["trans_id", "time_iso"])
    return hist_df


def aggregate_transactions(hist_df: pd.DataFrame) -> pd.DataFrame:
    if hist_df.empty:
        return pd.DataFrame(columns=[
            "trans_id",
            "first_seen",
            "block_time",
            "net_amount_ui",
            "abs_amount_ui",
            "direction",
            "token_accounts",
        ])
    grouped = hist_df.groupby("trans_id")
    records: list[dict] = []
    for tx_id, grp in grouped:
        ts = grp["time_iso"].min()
        block_time = pd.to_numeric(grp.get("block_time"), errors="coerce").min()
        net = grp["signed_amount_ui"].sum()
        direction = "inc" if net > 0 else "dec" if net < 0 else "flat"
        token_accounts = sorted(set(grp.get("token_account", "").astype(str)))
        records.append({
            "trans_id": str(tx_id),
            "first_seen": ts,
            "block_time": block_time,
            "net_amount_ui": net,
            "abs_amount_ui": abs(net),
            "direction": direction,
            "token_accounts": ";".join(token_accounts),
        })
    out = pd.DataFrame(records)
    out = out.sort_values("first_seen")
    return out


def build_plot_dataframe(
    aggregated: pd.DataFrame,
    swaps: pd.DataFrame,
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    if aggregated.empty:
        return aggregated
    classifications: list[str] = []
    router1: list[Optional[str]] = []
    router2: list[Optional[str]] = []
    intermed: list[Optional[str]] = []
    for _, row in aggregated.iterrows():
        tx_hash = str(row["trans_id"])
        classification = classify_from_swaps(tx_hash, swaps)
        label = None if classification is None else classification.label
        if label is None:
            label = "transfer"
        if label == "transfer" and detect_airdrop(row["abs_amount_ui"], classification, change_direction=row["direction"]):
            label = "airdrop"
        classifications.append(label)
        router1.append(None if classification is None else classification.router_token1)
        router2.append(None if classification is None else classification.router_token2)
        intermed.append(None if classification is None else classification.intermediary_label)
    aggregated = aggregated.copy()
    aggregated["classification"] = classifications
    aggregated["router_token1"] = router1
    aggregated["router_token2"] = router2
    aggregated["intermediary_label"] = intermed
    aggregated["signed_amount"] = aggregated["net_amount_ui"]
    aggregated["timestamp"] = aggregated["first_seen"]
    aggregated = aggregated.dropna(subset=["timestamp"])
    aggregated["timestamp"] = pd.to_datetime(aggregated["timestamp"], utc=True)
    aggregated = aggregated[aggregated["abs_amount_ui"] > 0]
    if verbose and aggregated.empty:
        print("[warn] after filtering zero-amount transactions nothing remains")
    return aggregated


def write_transactions_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[done] wrote transactions → {path} (rows={len(df)})")


def plot_transactions(
    df: pd.DataFrame,
    price: Optional[pd.DataFrame],
    outfile: Path,
    *,
    owner: str,
    window_label: str,
    figsize: Tuple[float, float],
    dpi: int,
) -> None:
    if df.empty:
        print("[warn] no transactions to plot; skipping figure")
        return
    colors = {
        "buy": "#2ca02c",
        "sell": "#d62728",
        "transfer": "#1f77b4",
        "airdrop": "#9467bd",
    }
    df["classification"] = df["classification"].astype(str)
    df["color"] = df["classification"].map(lambda v: colors.get(v, "#7f7f7f"))
    sizes = compute_bubble_sizes(df["abs_amount_ui"])

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("white")

    ax.scatter(
        df["timestamp"],
        df["signed_amount"],
        s=sizes,
        c=df["color"],
        alpha=0.7,
        edgecolors="black",
        linewidths=0.6,
    )

    ax.axhline(0, color="#606060", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("Signed TDCCP amount")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_tdccp))
    ax.grid(True, which="major", axis="both", linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", labelsize=12)

    ax.set_xlabel("Transaction time (UTC)")
    title = f"TDCCP Transactions • {owner} • {window_label}"
    ax.set_title(title, fontsize=18, pad=14)

    # Legend
    handles = list(legend_handles(colors))
    ax.legend(handles=handles, title="Classification", loc="upper left", frameon=True, framealpha=0.9)

    # Secondary axis for price
    if price is not None and not price.empty:
        ax2 = ax.twinx()
        ax2.plot(price["ts"], price["price_usd"], color="black", linewidth=1.2, alpha=0.85, label="TDCCP Price (USD)")
        ax2.set_ylabel("TDCCP Price (USD)")
        ax2.yaxis.set_major_formatter(FuncFormatter(fmt_tdccp))
        ax2.grid(False)
        # combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, framealpha=0.9)

    fig.autofmt_xdate()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
    print(f"[done] wrote figure → {outfile}")


def main():
    ap = argparse.ArgumentParser(
        description="Fetch TDCCP transactions for an address and plot a bubble chart with TDCCP price overlay.",
    )
    ap.add_argument("--owner", required=True, help="Owner (from_address) public key")
    ap.add_argument("--start", help="Start date/time in ISO (default: settings START)")
    ap.add_argument("--end", help="End date/time in ISO (default: settings END)")
    ap.add_argument("--mint", help="TDCCP mint address (default: settings MINT)")
    ap.add_argument("--swaps-csv", default=str(SWAPS_CSV), help="Path to swaps.csv (default: data/swaps.csv)")
    ap.add_argument(
        "--price-csv",
        default=str(PRICE_HISTORY_CSV),
        help="Path to TDCCP price history CSV (default: data/tdccp_price_history.csv)",
    )
    ap.add_argument("--out-csv", help="Optional explicit path for the output transactions CSV")
    ap.add_argument("--out-fig", help="Optional explicit path for the output figure")
    ap.add_argument("--figsize", default="20,10", help="Figure size as WIDTH,HEIGHT (default: 20,10)")
    ap.add_argument("--dpi", type=int, default=220, help="Figure DPI (default: 220)")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    owner = args.owner.strip()
    if not owner:
        raise SystemExit("[error] --owner must not be blank")

    start_default, end_default = window_defaults()
    start_str = args.start or start_default
    end_str = args.end or end_default
    if not start_str or not end_str:
        raise SystemExit("[error] start/end not provided and not found in settings.csv")

    mint = args.mint or mint_default()

    start_ts = pd.to_datetime(start_str, utc=True, errors="raise")
    end_ts = pd.to_datetime(end_str, utc=True, errors="raise")
    window_label = human_window_label(start_ts, end_ts)

    if args.verbose:
        print(f"[info] owner={owner}")
        print(f"[info] window={start_ts.isoformat()} → {end_ts.isoformat()}")
        print(f"[info] mint={mint}")

    api_key = require_api_key()

    hist = gather_transactions(owner, mint, start_str, end_str, api_key, verbose=args.verbose)
    if hist.empty:
        print("[warn] no TDCCP balance changes found for this owner/window")
        empty_df = pd.DataFrame(columns=[
            "trans_id",
            "timestamp",
            "net_amount_ui",
            "abs_amount_ui",
            "direction",
            "classification",
            "router_token1",
            "router_token2",
            "intermediary_label",
            "token_accounts",
        ])
        out_csv = Path(args.out_csv) if args.out_csv else OUT_TX_DIR / f"{owner}_{window_label}_tdccp_transactions.csv"
        write_transactions_csv(empty_df, out_csv)
        return

    aggregated = aggregate_transactions(hist)
    swaps = load_swaps_for_owner(owner, Path(args.swaps_csv), verbose=args.verbose)
    plot_df = build_plot_dataframe(aggregated, swaps, verbose=args.verbose)

    out_csv = Path(args.out_csv) if args.out_csv else OUT_TX_DIR / f"{owner}_{window_label}_tdccp_transactions.csv"
    write_transactions_csv(plot_df, out_csv)

    price = load_price_history(Path(args.price_csv), verbose=args.verbose)
    if price is not None:
        price = price[(price["ts"] >= start_ts) & (price["ts"] <= end_ts)]

    out_fig = Path(args.out_fig) if args.out_fig else OUT_FIG_DIR / f"TDCCP_Transactions_{owner}_{window_label}.png"
    figsize = parse_figsize(args.figsize)
    plot_transactions(plot_df, price, out_fig, owner=owner, window_label=window_label, figsize=figsize, dpi=args.dpi)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
