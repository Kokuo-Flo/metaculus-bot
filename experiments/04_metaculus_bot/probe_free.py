"""MEASURE: which free OpenRouter models actually answer, how fast, and what the daily quota really is.

    python3 probe_free.py --passes 2 --interval 300 --out probe.json

One tiny request per free model (max_tokens 8, no retry, ~3.5 s apart to stay under 20 requests/minute), repeated
`passes` times `interval` seconds apart; later passes keep only the models that answered or were merely
rate-limited. Prints a table (answers / attempts, median latency, last error) and writes the raw log as JSON.
Reads OPENROUTER_API_KEY from the environment or ../../.env. Costs nothing: free models only.
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent))
from bot import load_dotenv  # noqa: E402

API = "https://openrouter.ai/api/v1"
PROMPT = "Reply with exactly one word: OK"


def free_models() -> list[dict]:
    with urllib.request.urlopen(urllib.request.Request(f"{API}/models", headers={"User-Agent": "probe"}), timeout=60) as r:
        data = json.loads(r.read())["data"]
    out = []
    for model in data:
        pricing = model.get("pricing") or {}
        if float(pricing.get("prompt") or 0) == 0 and float(pricing.get("completion") or 0) == 0 and model["id"].endswith(":free"):
            out.append({"id": model["id"], "context": model.get("context_length"),
                        "params": set(model.get("supported_parameters") or [])})
    return sorted(out, key=lambda m: m["id"])


def probe(model: dict, key: str, prompt: str = PROMPT, max_tokens: int = 8) -> dict:
    body = {"model": model["id"], "messages": [{"role": "user", "content": prompt}]}
    if "max_tokens" in model["params"]:
        body["max_tokens"] = max_tokens
    request = urllib.request.Request(f"{API}/chat/completions", data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                              "User-Agent": "revenue-lab-bot/probe"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.loads(response.read())
        text = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        error = payload.get("error")
        return {"ok": not error and bool(text.strip()), "status": 200, "latency": round(time.monotonic() - started, 1),
                "text": text.strip()[-400:], "error": json.dumps(error)[:160] if error else ("empty answer" if not text.strip() else "")}
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            raw = json.loads(detail).get("error", {})
            message = (raw.get("metadata") or {}).get("raw") or raw.get("message") or detail
        except Exception:
            message = detail
        return {"ok": False, "status": e.code, "latency": round(time.monotonic() - started, 1), "error": str(message)[:160]}
    except Exception as e:
        return {"ok": False, "status": None, "latency": round(time.monotonic() - started, 1), "error": f"{type(e).__name__}: {str(e)[:120]}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--interval", type=int, default=300, help="seconds between passes")
    parser.add_argument("--gap", type=float, default=3.5, help="seconds between requests (20/min limit)")
    parser.add_argument("--max-models", type=int, default=60)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--models", nargs="*", default=None, help="probe only these model ids")
    args = parser.parse_args()
    load_dotenv(HERE.parent.parent / ".env")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY missing")
    models = free_models()[:args.max_models]
    if args.models:
        models = [m for m in models if m["id"] in set(args.models)]
    print(f"{len(models)} free models listed", flush=True)
    log: dict[str, list[dict]] = defaultdict(list)
    for number in range(1, args.passes + 1):
        if number > 1:
            keep = {m for m, results in log.items() if any(r["ok"] or r["status"] == 429 for r in results)}
            models = [m for m in models if m["id"] in keep]
            print(f"--- pass {number} in {args.interval}s on {len(models)} models", flush=True)
            time.sleep(args.interval)
        for model in models:
            result = probe(model, key, max_tokens=args.max_tokens)
            result["ts"] = time.strftime("%H:%M:%S")
            log[model["id"]].append(result)
            flag = "OK " if result["ok"] else f"{result['status'] or 'ERR'}"
            print(f"  [{number}] {flag:<4} {result['latency']:>5}s  {model['id']:<50} {result.get('error', '')[:70]}", flush=True)
            time.sleep(args.gap)
    print("\n=== summary (answers/attempts, median latency) ===")
    rows = []
    for model_id, results in log.items():
        oks = [r for r in results if r["ok"]]
        rows.append((len(oks), len(results), statistics.median(r["latency"] for r in oks) if oks else None, model_id,
                     next((r["error"] for r in reversed(results) if r.get("error")), "")))
    for oks, n, latency, model_id, error in sorted(rows, key=lambda r: (-r[0] / r[1], r[2] or 999)):
        print(f"  {oks}/{n}  {('%5.1fs' % latency) if latency else '    -'}  {model_id:<50} {error[:60]}")
    if args.out:
        args.out.write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
