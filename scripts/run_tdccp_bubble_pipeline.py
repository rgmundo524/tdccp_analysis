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


def _csv_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
    except Exception:
        return []
    return [h for h in header if h]


def _has_address_column(header: list[str]) -> bool:
    normalized = [(col, col.strip().lower()) for col in header if col]
    preferred = {
        "from_address",
        "address",
        "addr",
        "wallet_address",
        "wallet",
        "spike_address",
        "highlight_address",
        "account_address",
        "account",
    }
    for original, lowered in normalized:
        if lowered in preferred:
            return True
    for original, lowered in normalized:
        if "address" in lowered and not lowered.startswith("to_"):
            return True
    return False


def _resolve_spike_addresses(path: Path) -> tuple[Path | None, list[str]]:
    """Return (resolved_path, log_messages)."""

    messages: list[str] = []
    resolved = path

    if "spike_buckets_" in path.name and "spike_addresses_" not in path.name:
        candidate_name = path.name.replace("spike_buckets_", "spike_addresses_", 1)
        candidate = path.with_name(candidate_name)
        if candidate.exists():
            messages.append(
                f"[pipeline] Detected spike bucket CSV; using companion address file {candidate_name}."
            )
            resolved = candidate
        else:
            messages.append(
                "[warn] Spike CSV lacks address column and matching file "
                f"{candidate_name} was not found. Skipping highlight overlays."
            )
            return None, messages

    header = _csv_header(resolved)
    if not header:
        messages.append(
            f"[warn] Could not read header from spike CSV {resolved}. Skipping highlight overlays."
        )
        return None, messages

    if not _has_address_column(header):
        cols = ", ".join(header)
        messages.append(
            "[warn] Spike CSV does not include an address column (expected something like "
            f"'from_address'). Columns: {cols}. Skipping highlight overlays."
        )
        return None, messages

    return resolved, messages

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
    raw_spike_path = Path(args.spike_addresses) if args.spike_addresses else None
    spike_path: Path | None = None
    if raw_spike_path:
        if not raw_spike_path.exists():
            sys.exit(f"[error] spike CSV not found: {raw_spike_path}")
        resolved, logs = _resolve_spike_addresses(raw_spike_path)
        for line in logs:
            print(line)
        spike_path = resolved

    # 1) Build metrics if missing
    if not metrics_path.exists():
        build_cmd = [PY, str(BUILDER), "--metrics", str(metrics_path)]
        run(build_cmd, step="build metrics")

    # 2) Address bubble (size=color off tx count bins)
    label = window_label_from_settings()
    plotA = [PY, str(PLOT_A), "--metrics", str(metrics_path), "--top-labels", str(args.top_labels)]
    if spike_path:
        plotA += ["--highlight-addresses", str(spike_path)]
    if label: plotA += ["--window-label", label]
    run(plotA, step="plot address bubble")

    # 3) Labeled address bubble (color from settings.csv labels)
    plotB = [PY, str(PLOT_B), "--metrics", str(metrics_path), "--top-labels", str(args.top_labels)]
    if spike_path:
        plotB += ["--highlight-addresses", str(spike_path)]
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

