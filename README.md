# TDCCP Analysis Pipeline

This repository contains a full pipeline for ingesting, pricing, labeling, and analyzing **TDCCP token swaps on Solana**. It builds a canonical dataset of swaps, enriches them with metadata and USD pricing, detects intermediary routing, analyzes spikes in buy/sell pressure, and generates a variety of plots including pressure vs. price and address bubble charts.

The project is designed for **reproducibility, clarity, and forensic analysis**. It uses Solscan Pro for balance changes and CoinGecko Pro for pricing (Pro-only endpoints).

---

## Goals

1. Produce a **canonical `data/swaps.csv`** with:
   - All TDCCP-involving swaps,
   - USD values for both legs,
   - Attribution labels,
   - Intermediary detection and net TDCCP column.

2. Maintain token meta and price history:
   - `data/prices_long.csv` (CoinGecko Pro OHLC),
   - `data/token_meta.csv`,
   - `data/tdccp_price_history.csv` (used for plots).

3. Perform **spike analysis** and generate **pressure vs. price** plots.

4. Generate **bubble charts** showing address-level and label-level net flows and balances.

---

## Configuration

### Environment Variables (`.env`)
Required API keys:

- `HELIUS_API_KEY` – if used for swap collection.
- `SOLSCAN_API_KEY` – required for `fetch_address_history.py`.
- `COINGECKO_API_KEY` – required for `fetch_meta_and_prices.py`.

### Settings (`settings.csv`)
Human-maintained config file. Important rows:

- `core,START,YYYY-MM-DD`
- `core,END,YYYY-MM-DD`
- `core,CHUNK,6h`
- `core,DATA_DIR,data`
- `core,OUTPUTS_DIR,outputs`
- `core,MINT,<tdccp mint>`
- `address,<from_address>,<Label>` – label specific addresses (used in labeled bubble charts).
- `Program,<program_id>,<Label>` – label DEX programs.

---

## Data Layout

- `data/`
  - `swaps.csv` – canonical swaps file (master dataset).
  - `swaps_raw.jsonl` – optional raw swaps file.
  - `prices_long.csv` – OHLC data from CoinGecko Pro.
  - `token_meta.csv` – token metadata.
  - `tdccp_price_history.csv` – per-transaction TDCCP prices.
  - `addresses/`
    - `<OWNER>_token_accounts.csv` – token accounts for owner.
    - `<OWNER>_<start>-<end>.csv` – balance history for owner.
    - `tdccp_address_metrics.csv` – aggregated per-address metrics.

- `outputs/`
  - `figures/`
    - `VolumeLines_Price_<bucket>_<window>.png` – pressure vs. price charts.
    - `VolumeLines_Price_<bucket>_<window>_spikes.png` – pressure vs. price with red-outlined spike buckets.
    - `Address_Bubbles_addresses_<window>.png` – bubble chart (by transaction count bins).
    - `Address_Bubbles_byLabel_<window>.png` – bubble chart (by label) with dynamic colors sourced from `settings.csv` address groups, plus `Address_Bubbles_byLabel_<window>_highlight_<group>.png` variants for each label active in the selected window.
  - `analysis/`
    - `spike_buckets_<bucket>_<window>.csv`
    - `spike_addresses_<bucket>_<window>.csv`
    - `spike_swaps_<bucket>_<window>.csv`

---

## Workflows

### 1. Full Pipeline
Run:
```bash
python scripts/run_tdccp_pipeline.py
```

### 2. Address transaction bubble chart
Render a per-transaction bubble chart (with TDCCP price overlay) for a specific wallet:

```bash
python scripts/plot_tdccp_address_transactions_bubble.py --owner <FROM_ADDRESS>
```

The script automatically respects the `START`/`END` window from `settings.csv`, fetches the owner's TDCCP balance-change history from Solscan, classifies each transaction (buy, sell, transfer, or airdrop), writes a CSV summary to `data/addresses/`, and saves the bubble chart to `outputs/figures/`. The overlayed TDCCP price line is resampled to an hourly series for consistency with the balance-change cadence.

### 3. Spike-highlight pressure vs. price plots
Run the direct-flow spike scanner to emit analysis CSVs and spike-highlight pressure/price plots:

```bash
python scripts/analyze_spikes.py --start <YYYY-MM-DD> --end <YYYY-MM-DD>
```

For every requested bucket the script now invokes `plot_tdccp_pressure_vs_price_spikes.py`, which reuses the swaps window to draw the standard volume-vs-price lines and outlines sell-heavy spike buckets with red rectangles. Use either the default `--min-delta-pct` threshold or enable `--top-sell-count 5` to highlight the five most negative direct-flow buckets per chart. Sequential spike buckets are merged into a single red block so extended sell programs are easier to spot. Resulting figures are written alongside the base pressure/price plots in `outputs/figures/`.

