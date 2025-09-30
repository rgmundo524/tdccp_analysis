#!/usr/bin/env python3
import os, json, csv, pathlib, requests
from requests.adapters import HTTPAdapter, Retry
from datetime import datetime, timezone

BASE = pathlib.Path(__file__).resolve().parent.parent
SETTINGS_CSV = BASE / "settings.csv"
OUT_JSONL = BASE / "data/swaps_raw.jsonl"
OUT_CSV   = BASE / "data/swaps.csv"
SOLSCAN_BASE = "https://pro-api.solscan.io/v2.0"

def load_core():
    s = {}
    if SETTINGS_CSV.exists():
        with SETTINGS_CSV.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (row.get("Category") or "").strip().lower() == "core":
                    s[row.get("Key","").strip()] = (row.get("Value") or "").strip()
    return s

def iso(ts): return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

def solscan_session():
    key = os.getenv("SOLSCAN_API_KEY", "").strip()
    if not key: raise SystemExit("SOLSCAN_API_KEY missing")
    s = requests.Session()
    s.headers.update({"accept":"application/json","token":key})
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=6, backoff_factor=0.6, allowed_methods=["GET"], status_forcelist=[429,500,502,503,504])))
    return s

def crawl(start_ts, end_ts, mint):
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    if OUT_JSONL.exists(): OUT_JSONL.unlink()
    page = 1
    with solscan_session() as sess, OUT_JSONL.open("a", encoding="utf-8") as f:
        while True:
            r = sess.get(f"{SOLSCAN_BASE}/token/defi/activities", params={
                "address": mint,
                "activity_type[]":["ACTIVITY_TOKEN_SWAP","ACTIVITY_AGG_TOKEN_SWAP"],
                "from_time": start_ts,
                "to_time": end_ts,
                "page": page,
                "page_size": 100,
                "sort_by":"block_time",
                "sort_order":"asc",
            }, timeout=30)
            r.raise_for_status()
            data = r.json().get("data") or []
            if not data: break
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if len(data) < 100: break
            page += 1

def flatten():
    rows = []
    with OUT_JSONL.open(encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            bt = int(j.get("block_time", 0))
            tx = (j.get("trans_id") or j.get("tx_hash") or j.get("signature") or "")
            if not bt or not tx: continue

            # capture from_address (Solscan v2 places it at top level)
            from_addr = (j.get("from_address") or j.get("owner") or j.get("address") or "")

            routers = j.get("routers") or {}
            a1 = routers.get("amount1"); a2 = routers.get("amount2")
            t1 = routers.get("token1");  t2 = routers.get("token2")
            d1 = routers.get("token1_decimals"); d2 = routers.get("token2_decimals")
            if (not t1 or not t2 or a1 is None or a2 is None) and isinstance(routers.get("child_routers"), list) and routers["child_routers"]:
                last = routers["child_routers"][-1]
                t1 = t1 or last.get("token1"); t2 = t2 or last.get("token2")
                a1 = a1 if a1 is not None else last.get("amount1")
                a2 = a2 if a2 is not None else last.get("amount2")
                d1 = d1 if d1 is not None else last.get("token1_decimals")
                d2 = d2 if d2 is not None else last.get("token2_decimals")
            if not t1 or not t2: continue

            rows.append({
                "block_time": bt,
                "time_iso": iso(bt),
                "trans_id": tx,
                "from_address": from_addr,                 # <-- keep this
                "router_amount1_raw": a1,
                "router_amount2_raw": a2,
                "router_token1": t1,
                "router_token2": t2,
                "router_decimals1": d1,
                "router_decimals2": d2,
            })
    if not rows: raise SystemExit("No rows flattened")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    core = load_core()
    START = core.get("START", "2025-03-01")
    END   = core.get("END",   "2025-09-23")
    MINT  = core.get("MINT",  "Hg8bKz4mvs8KNj9zew1cEF9tDw1x2GViB4RFZjVEmfrD")

    start_ts = int(datetime.strptime(START, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    end_ts   = int(datetime.strptime(END,   "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

    crawl(start_ts, end_ts, MINT)
    flatten()
    print(f"[done] {OUT_CSV}")

