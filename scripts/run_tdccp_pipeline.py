#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end TDCCP pipeline runner with a simple progress bar and optional debug logs.

Steps:
1) collect_swaps.py
2) fetch_meta_and_prices.py
3) refine_and_usd_non_tdccp.py
4) tdccp_parity_price.py
5) apply_symbols.py
6) apply_address_attribution.py
7) detect_intermediary_swaps.py   <-- reads & writes data/swaps.csv (in-place)

Usage:
  python scripts/run_tdccp_pipeline.py [--debug]
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("Collect swaps",               ["collect_swaps.py"],                             ["data/swaps.csv"]),
    ("Meta & prices",               ["fetch_meta_and_prices.py"],                    ["data/token_meta.csv", "data/prices_long.csv"]),
    ("Normalize & USD (non-TDCCP)", ["refine_and_usd_non_tdccp.py"],                 ["data/swaps.csv"]),
    ("TDCCP parity pricing",        ["tdccp_parity_price.py"],                       ["data/swaps.csv"]),
    ("Replace symbols/tickers",     ["apply_symbols.py"],                            ["data/swaps.csv"]),
    ("Address attribution",         ["apply_address_attribution.py"],                ["data/swaps.csv", "data/tdccp_price_history.csv"]),
    ("Detect intermediary swaps",   ["detect_intermediary_swaps.py"],                ["data/swaps.csv"]),
]

def run(cmd: list[str], debug: bool) -> None:
    if debug:
        print("[run]", *cmd)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"[error] Command failed with code {r.returncode}: {' '.join(cmd)}")

def progress_bar(step_idx: int, total: int, title: str, start_time: float, state: str) -> None:
    # state: "run", "ok", "fail"
    width = 26
    filled = int(width * step_idx / total)
    bar = "#" * filled + "-" * (width - filled)
    suffix = {"run": "…", "ok": "✓", "fail": "✕"}[state]
    elapsed = time.time() - start_time
    print(f"[{bar}] {step_idx}/{total}  {title:<24} ({elapsed:.1f}s) {suffix}")

def main():
    ap = argparse.ArgumentParser(description="TDCCP Pipeline")
    ap.add_argument("--debug", action="store_true", help="print [run]/[done] lines for each step")
    args = ap.parse_args()

    total = len(STEPS)
    print("TDCCP Pipeline\n")

    start0 = time.time()
    for i, (title, script_parts, artifacts) in enumerate(STEPS, start=1):
        # pre-line (state=run, but index before launching)
        progress_bar(i-1, total, title, start0, "run")

        # build absolute path to script
        script_path = ROOT / "scripts" / script_parts[0]
        cmd = [PY, str(script_path)]
        if len(script_parts) > 1:
            cmd += script_parts[1:]

        run(cmd, args.debug)

        if args.debug:
            # show where artifacts went when we know the canonical destinations
            outs = "  ".join(str(ROOT / p) for p in artifacts)
            print("[done]", outs)

        # post-line success
        progress_bar(i, total, title, start0, "ok")

    print("\n[pipeline] complete → data/swaps.csv (now includes intermediary_label, net_tdccp)")

if __name__ == "__main__":
    main()

