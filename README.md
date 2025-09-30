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
    - `Volume_Price_<bucket>_<window>.png` – pressure vs. price charts.
    - `Address_Bubbles_addresses_<window>.png` – bubble chart (by transaction count bins).
    - `Address_Bubbles_by_label_<window>.png` – bubble chart (by label).
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

