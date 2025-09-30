#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Tuple, Optional

PY = sys.executable
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
ANALYZE = SCRIPTS / "analyze_spikes.py"
PRESSURE = SCRIPTS / "plot_tdccp_pressure_vs_price.py"
SETTINGS_CSV = ROOT / "settings.csv"
FIG_DIR = ROOT / "outputs" / "figures"

def read_start_end_from_settings() -> Tuple[Optional[str], Optional[str]]:
    if not SETTINGS_CSV.exists():
        return None, None
    # try to sniff a delimiter but don’t overcomplicate
    try:
        sample = SETTINGS_CSV.read_text(encoding="utf-8", errors="ignore")[:2048]
        dialect = csv.Sniffer().sniff(sample) if sample.strip() else csv.get_dialect("excel")
    except Exception:
        dialect = csv.get_dialect("excel")

    s = e = None
    with SETTINGS_CSV.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.reader(f, dialect)
        header = next(r, None)
        if not header:
            return None, None
        # locate indices (fallback to 0,1,2)
        lower = [c.strip().lower() for c in header]
        try:
            i_cat = lower.index("category")
            i_key = lower.index("key")
            i_val = lower.index("value")
        except ValueError:
            i_cat, i_key, i_val = 0, 1, 2

        # if header is actually a row in your format, also consider it
        rows = [header] + list(r)
        for row in rows:
            if not row or len(row) <= max(i_cat, i_key, i_val): 
                continue
            cat = (row[i_cat] or "").strip().lower()
            key = (row[i_key] or "").strip().upper()
            val = (row[i_val] or "").strip()
            if cat == "core" and key == "START":
                s = val
            elif cat == "core" and key == "END":
                e = val
    return s or None, e or None

def run(cmd: list[str], step: str, debug: bool) -> None:
    if debug:
        print("[run]", " ".join(cmd))
    # inherit stdout/stderr so you can see child script messages
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"[error] {step} failed → rc={e.returncode}: {' '.join(cmd)}")

def ensure_figures(label_hint: str | None) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs = list(FIG_DIR.glob("Volume_Price_*.png"))
    if figs:
        # at least one figure exists; done
        return
    # nothing found — fail with guidance
    hint = f" --start {label_hint[:8]} --end {label_hint[9:]}" if label_hint and len(label_hint) == 17 else ""
    sys.exit(
        "[error] No figures found in outputs/figures after plotting.\n"
        "Possible causes:\n"
        "  • No TDCCP-involving swaps in the selected window\n"
        "  • The plot script exited early due to empty data\n"
        "Next step (verbose):\n"
        f"  {PY} {PRESSURE} --debug{hint}\n"
        "Optionally add:  --buckets 1d,12h,6h,3h,1h,30min,10min"
    )

def main():
    ap = argparse.ArgumentParser(
        description="Pipeline: analyze spikes (direct-only) + plot TDCCP pressure vs price."
    )
    ap.add_argument("--start", help="UTC start (YYYY-mm-dd). Defaults to core:START in settings.csv")
    ap.add_argument("--end",   help="UTC end   (YYYY-mm-dd). Defaults to core:END   in settings.csv")
    ap.add_argument(
        "--buckets",
        default=None,
        help="Comma list (e.g. 1d,12h,6h,3h,1h,30min,10min). If omitted, each step uses its defaults.",
    )
    group = ap.add_mutually_exclusive_group()
    group.add_argument(
        "--min-delta-pct",
        type=float,
        default=None,
        help=(
            "Minimum sell-heavy delta_direct_pct required to treat a bucket as a spike. "
            "Default 25 when --top-sell-count is not provided."
        ),
    )
    group.add_argument(
        "--top-sell-count",
        type=int,
        default=None,
        help=(
            "When provided, ignore --min-delta-pct and instead focus on the top N sell-heavy buckets "
            "sorted by most-negative delta_direct."
        ),
    )
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--routing-thresh", type=float, default=0.25)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    s_def, e_def = read_start_end_from_settings()
    start = args.start or s_def
    end   = args.end   or e_def
    if not start or not end:
        sys.exit("[error] --start/--end not provided and START/END not found in settings.csv.")

    use_top_sell = args.top_sell_count is not None
    if use_top_sell and args.top_sell_count <= 0:
        sys.exit("[error] --top-sell-count must be positive when supplied.")
    min_delta = args.min_delta_pct if args.min_delta_pct is not None else 25.0

    # 1) analyze_spikes.py (no window-label; the script doesn’t accept it)
    analyze_cmd = [
        PY, str(ANALYZE),
        "--start", start,
        "--end", end,
        "--top-n", str(args.top_n),
        "--routing-thresh", str(args.routing_thresh),
    ]
    if use_top_sell:
        analyze_cmd += ["--top-sell-count", str(args.top_sell_count)]
    else:
        analyze_cmd += ["--min-delta-pct", str(min_delta)]
    if args.buckets:
        analyze_cmd += ["--buckets", args.buckets]
    if args.debug:
        analyze_cmd += ["--debug"]
    run(analyze_cmd, "analyze_spikes", args.debug)

    # 2) plot_tdccp_pressure_vs_price.py
    plot_cmd = [PY, str(PRESSURE), "--start", start, "--end", end]
    if args.buckets:
        plot_cmd += ["--buckets", args.buckets]
    if args.debug:
        plot_cmd += ["--debug"]
    run(plot_cmd, "plot_tdccp_pressure_vs_price", args.debug)

    # 3) verify figure(s) exist
    label_hint = f"{start.replace('-','')}-{end.replace('-','')}"
    ensure_figures(label_hint)

    print("[pipeline] spikes + pressure/price done.")

if __name__ == "__main__":
    main()

