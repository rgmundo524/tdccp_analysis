#!/usr/bin/env python3
from __future__ import annotations

"""Export TDCCP addresses with negative net volume for bubble-chart review.

This helper mirrors the data preparation used by the address bubble plots but
emits a CSV that focuses exclusively on wallets whose cumulative TDCCP flow is
negative within the configured window.  Addresses that *are not* listed as
"Airdrop recipient" in ``settings.csv`` are flagged via a ``highlight`` column
so the downstream bubble charts (or manual review) can emphasise organic
sell-heavy participants instead of the known airdrop cohort.
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, Set

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SETTINGS_CSV = ROOT / "settings.csv"
DEFAULT_METRICS = DATA_DIR / "addresses" / "tdccp_address_metrics.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "analysis" / "tdccp_negative_net_addresses.csv"


# ---------------------------------------------------------------------------
# settings helpers

def read_airdrop_addresses(settings_path: Path) -> Set[str]:
    """Return the set of addresses labelled "Airdrop recipient" in settings."""

    if not settings_path.exists():
        raise SystemExit(f"[error] settings.csv not found: {settings_path}")

    df = pd.read_csv(settings_path)
    required = {"Category", "Key", "Value"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            "[error] settings.csv missing required columns: " f"{sorted(missing)}"
        )

    cat = df["Category"].astype(str).str.strip().str.lower()
    label = df["Value"].astype(str).str.strip().str.lower()
    mask = (cat == "address") & (label == "airdrop recipient")
    addrs = (
        df.loc[mask, "Key"].dropna().astype(str).str.strip().tolist()
    )
    return {addr for addr in addrs if addr}


# ---------------------------------------------------------------------------
# metrics helpers

def load_metrics(metrics_path: Path) -> pd.DataFrame:
    if not metrics_path.exists():
        raise SystemExit(f"[error] metrics csv not found: {metrics_path}")

    df = pd.read_csv(metrics_path)
    required = {"from_address", "net_ui", "peak_balance_ui"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            "[error] metrics csv missing required columns: " f"{sorted(missing)}"
        )

    df["from_address"] = df["from_address"].astype(str).str.strip()
    df["net_ui"] = pd.to_numeric(df["net_ui"], errors="coerce")
    df["peak_balance_ui"] = pd.to_numeric(df["peak_balance_ui"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# core export logic

def build_negative_dataframe(
    metrics: pd.DataFrame,
    *,
    include_columns: Iterable[str],
    airdrop_addresses: Set[str],
    min_abs_net: float,
) -> pd.DataFrame:
    """Return the filtered dataframe with highlight metadata."""

    if "net_ui" not in metrics.columns:
        raise SystemExit("[error] metrics dataframe missing 'net_ui' column")

    working = metrics.copy()
    working = working[pd.notna(working["net_ui"])]
    working = working[working["net_ui"] < 0.0]

    if min_abs_net > 0:
        working = working[working["net_ui"].abs() >= min_abs_net]

    if working.empty:
        raise SystemExit("[error] no addresses with negative net_ui matched the filters")

    include_cols = list({"from_address", "net_ui", "peak_balance_ui", *include_columns})
    missing = [col for col in include_cols if col not in working.columns]
    if missing:
        raise SystemExit(
            "[error] requested columns missing from metrics csv: " f"{missing}"
        )

    export = working[include_cols].copy()
    export["airdrop_recipient"] = export["from_address"].isin(airdrop_addresses)
    export["highlight"] = ~export["airdrop_recipient"]
    export["highlight_reason"] = export["highlight"].map(
        lambda flag: "net_negative_non_airdrop" if flag else ""
    )
    export["abs_net_ui"] = export["net_ui"].abs()

    export = export.sort_values(by=["highlight", "net_ui"], ascending=[False, True])
    return export


def export_negative_addresses(
    metrics_path: Path,
    settings_path: Path,
    output_path: Path,
    *,
    extra_columns: Iterable[str],
    min_abs_net: float,
) -> Path:
    metrics = load_metrics(metrics_path)
    airdrops = read_airdrop_addresses(settings_path)
    negative_df = build_negative_dataframe(
        metrics,
        include_columns=extra_columns,
        airdrop_addresses=airdrops,
        min_abs_net=min_abs_net,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    negative_df.to_csv(output_path, index=False)

    highlight_count = int(negative_df["highlight"].sum())
    total = int(len(negative_df))
    print(
        "[done] wrote", output_path,
        "—", highlight_count,
        "of", total,
        "addresses flagged as net-negative non-airdrop",
    )
    return output_path


# ---------------------------------------------------------------------------
# cli

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Export TDCCP addresses with negative net volume, annotating non-"
            "airdrop participants for bubble-chart highlighting."
        )
    )
    ap.add_argument(
        "--metrics-csv",
        default=str(DEFAULT_METRICS),
        help="Path to tdccp_address_metrics.csv (default: settings DATA_DIR).",
    )
    ap.add_argument(
        "--settings",
        default=str(SETTINGS_CSV),
        help="Path to settings.csv (used to detect airdrop recipients).",
    )
    ap.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Where to write the filtered CSV (default: outputs/analysis/...).",
    )
    ap.add_argument(
        "--include-column",
        action="append",
        default=[],
        help=(
            "Additional column from metrics csv to carry into the export. "
            "Use multiple times for more columns."
        ),
    )
    ap.add_argument(
        "--min-abs-net",
        type=float,
        default=0.0,
        help=(
            "Optional absolute net_ui floor (TDCCP). Only addresses with |net_ui| "
            "≥ this value are kept."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    try:
        export_negative_addresses(
            metrics_path=Path(args.metrics_csv),
            settings_path=Path(args.settings),
            output_path=Path(args.output),
            extra_columns=args.include_column,
            min_abs_net=args.min_abs_net,
        )
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive guardrail
        sys.exit(f"[error] unexpected failure: {exc}")


if __name__ == "__main__":
    main()

