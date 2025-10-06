# TDCCP peak balances and their data sources

Early versions of the bubble metrics inferred the per-address `peak_balance_ui`
by integrating **only** the swap ledger (`data/swaps.csv`). If a wallet received
TDCCP via a plain transfer or airdrop and subsequently routed every token out
through swaps, the running sum of `net_tdccp` inside the window never moved above
zero. The CSV therefore reported a "peak" that was either zero or, worse,
negative—an artefact of ignoring the non-swap inflows.

The metrics builder (`scripts/build_bubble_pipeline.py`) now looks for the
per-address balance histories emitted by `scripts/fetch_address_history.py`
(`data/addresses/<owner>_<start>-<end>.csv`). When those files are present the
script computes each wallet's peak TDCCP exposure from the Solscan balance
change stream instead of the swap ledger. The resulting `peak_balance_ui` is the
maximum `pre_ui`/`post_ui` observed during the analysis window, so it always
reflects the true highest balance the owner held—even if every swap in the
window was an outflow.

Each metrics row now carries a helper column, `peak_balance_source`, which is
`history` when Solscan data was available and `missing_history` when the
analysis had to fall back to `0.0` because no balance history file existed for
that address/window. There is **no** swap-derived fallback anymore; fetch the
history, rerun the pipeline, and the peak will automatically be sourced from the
on-chain balance changes. The fetcher writes two artefacts per address/window:

* `<owner>_<start>-<end>.csv` — raw Solscan balance-change rows (with
  `pre_ui`/`post_ui`).
* `<owner>_<start>-<end>_transactions.csv` — one row per TDCCP transaction with
  the signed `net_amount_ui`, the reconstructed running balance, and a
  cumulative peak column you can compare directly to the bubble chart's running
  sums.

Run it like this:

```bash
python scripts/fetch_address_history.py \
  --owner <FROM_ADDRESS> \
  --token-mint <TDCCP_MINT> \
  --start 2025-03-01 --end 2025-09-23
python scripts/build_bubble_pipeline.py --debug
```

Replace `<TDCCP_MINT>` with the mint listed under `core,MINT` in `settings.csv`.

To manually verify a peak that was sourced from history, open the corresponding
CSV and check the highest `pre_ui`/`post_ui` value within the window—the metrics
CSV will match that number. If the `peak_balance_source` says `missing_history`,
the address simply lacks a fetched balance history; fetching it will upgrade the
metrics on the next run.
