#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, subprocess, sys
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / "settings.csv"

DEFAULT_METRICS = ROOT / "data" / "addresses" / "tdccp_address_metrics.csv"
BUILDER = ROOT / "scripts" / "build_bubble_pipeline.py"
PLOT_A  = ROOT / "scripts" / "plot_tdccp_address_bubble.py"
PLOT_B  = ROOT / "scripts" / "plot_tdccp_address_bubble_by_label.py"
PLOT_C  = ROOT / "scripts" / "plot_tdccp_address_bubble_with_spikes.py"

def read_settings_value(key_name: str) -> str | None:
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
                    if lc == "key":   key_idx = i
                    elif lc == "value": val_idx = i
            target = (key_name or "").strip().upper()
            for row in rdr:
                if not row or len(row) <= max(key_idx, val_idx): continue
                if (row[key_idx] or "").strip().upper() == target:
                    return (row[val_idx] or "").strip()
    except Exception:
        return None
    return None

def window_label_from_settings() -> str | None:
    s = read_settings_value("START"); e = read_settings_value("END")
    if not s or not e: return None
    return f"{s.replace('-','')}-{e.replace('-','')}"

def run(cmd: list[str], step: str = ""):
    rc = subprocess.call(cmd)
    if rc != 0:
        msg = f"[error] command failed ({rc}): {' '.join(cmd)}"
        if step: msg = f"[error] {step} failed → {msg}"
        sys.exit(msg)

def main():
    ap = argparse.ArgumentParser(description="TDCCP bubble pipeline: build metrics then plot bubbles.")
    ap.add_argument("--metrics", default=str(DEFAULT_METRICS),
                    help=f"Path for tdccp_address_metrics.csv (default: {DEFAULT_METRICS})")
    ap.add_argument("--top-labels", type=int, default=20, help="How many top points to annotate (default: 20)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--spike-addresses", dest="spike_addresses", default=None,
                    help="Optional CSV of spike addresses to highlight in a third render")
    args = ap.parse_args()

    metrics_path = Path(args.metrics)
    spike_path = Path(args.spike_addresses) if args.spike_addresses else None

    if spike_path and not spike_path.exists():
        sys.exit(f"[error] spike CSV not found: {spike_path}")

    # 1) Build metrics if missing
    if not metrics_path.exists():
        build_cmd = [PY, str(BUILDER), "--metrics", str(metrics_path)]
        run(build_cmd, step="build metrics")

    # 2) Address bubble (size=color off tx count bins)
    label = window_label_from_settings()
    plotA = [PY, str(PLOT_A), "--metrics", str(metrics_path), "--top-labels", str(args.top_labels)]
    if label: plotA += ["--window-label", label]
    run(plotA, step="plot address bubble")

    # 3) Labeled address bubble (color from settings.csv labels)
    plotB = [PY, str(PLOT_B), "--metrics", str(metrics_path), "--top-labels", str(args.top_labels)]
    if label: plotB += ["--window-label", label]
    run(plotB, step="plot labeled address bubble")

    if spike_path:
        plotC = [PY, str(PLOT_C), "--metrics", str(metrics_path), "--spike-addresses", str(spike_path)]
        if label: plotC += ["--window-label", label]
        run(plotC, step="plot spike-highlight bubble")

    if spike_path:
        print("[pipeline] bubble charts plus spike-highlight render complete.")
    else:
        print("[pipeline] bubble charts complete.")

if __name__ == "__main__":
    main()

