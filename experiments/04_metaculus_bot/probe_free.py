"""MEASURE: which free OpenRouter models actually answer, how fast, and what the daily quota really is.

    python3 probe_free.py --passes 2 --interval 300 --out probe.json       # OpenRouter's :free models
    python3 probe_free.py --provider groq [--realistic] [--models id ...]  # another provider, once its key is set

With --provider, the provider's own /models catalogue is listed, the ids written "<provider>/<id>" in
bot_config.json are checked against it, and the configured (or --models) ids are probed; --realistic sends a
forecast-sized prompt with 8 000 output tokens and checks that a "Probability:" line comes back.

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
from bot import CONFIG, load_dotenv  # noqa: E402
from clients import PROVIDERS, split_model  # noqa: E402

API = "https://openrouter.ai/api/v1"
PROMPT = "Reply with exactly one word: OK"


def provider_models(base_url: str, key: str) -> list[str]:
    """Ids from an OpenAI-compatible /models listing (Gemini prefixes them with 'models/': stripped)."""
    request = urllib.request.Request(f"{base_url}/models", headers={"Authorization": f"Bearer {key}", "User-Agent": "probe"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read())
    items = data.get("data") if isinstance(data, dict) else data
    ids = []
    for item in items or []:
        model_id = item.get("id") if isinstance(item, dict) else str(item)
        if model_id:
            ids.append(model_id[7:] if model_id.startswith("models/") else model_id)
    return sorted(ids)


def configured_models(provider: str) -> list[str]:
    """Model ids of this provider found anywhere in bot_config.json (profiles' models, research, reformat)."""
    raw = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    found = []
    for profile in (raw.get("profiles") or {}).values():
        for model in [*(profile.get("models") or []), profile.get("research_model"), profile.get("reformat_model")]:
            if model and split_model(model)[0] == provider and split_model(model)[1] not in found:
                found.append(split_model(model)[1])
    return found


def realistic_prompt() -> str:
    from forecaster import BINARY_TAIL, COMMON
    filler = " ".join(f"On 2026-10-{(i % 28) + 1:02d}, a daily report noted the index at {100 + i * 0.37:.2f} points."
                      for i in range(120))
    context = {"title": "Will the Example Index close above 150 points on 2026-11-30?",
               "background": "The Example Index is a synthetic daily series used for this probe.",
               "criteria": "Resolves YES if the official close on 2026-11-30 is strictly above 150.", "fine_print": "",
               "today": "2026-10-19", "close": "2026-11-29T23:00:00Z", "resolve": "2026-12-01T00:00:00Z",
               "research": "## Recent reports\n" + filler}
    return COMMON.format(**context) + BINARY_TAIL


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


def probe(model: dict, key: str, prompt: str = PROMPT, max_tokens: int = 8, base_url: str = API) -> dict:
    body = {"model": model["id"], "messages": [{"role": "user", "content": prompt}]}
    if "max_tokens" in model["params"]:
        body["max_tokens"] = max_tokens
    request = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(body).encode(), method="POST",
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
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default=None,
                        help="probe another provider (its key must be set) instead of OpenRouter's free models")
    parser.add_argument("--realistic", action="store_true", help="forecast-sized prompt, 8000 tokens, parse check")
    args = parser.parse_args()
    load_dotenv(HERE.parent.parent / ".env")
    if args.provider:
        return probe_provider(args)
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


def probe_provider(args) -> None:
    from forecaster import parse_probability
    base_url, env = PROVIDERS[args.provider]
    key = os.environ.get(env)
    if not key:
        sys.exit(f"{env} missing: put the {args.provider} key in .env (and in a GitHub secret of the same name)")
    try:
        catalogue = provider_models(base_url, key)
        print(f"{args.provider}: {len(catalogue)} models listed by {base_url}/models")
        for model_id in catalogue:
            print("   ", model_id)
    except Exception as error:
        catalogue = []
        print(f"{args.provider}: /models listing failed ({str(error)[:120]}); probing the configured ids blindly")
    wanted = args.models or configured_models(args.provider)
    if not wanted:
        print(f"nothing configured for {args.provider} in bot_config.json; pass --models to probe some ids")
        return
    for model_id in wanted:
        if catalogue and model_id not in catalogue:
            close = [c for c in catalogue if model_id.split("/")[-1][:8].lower() in c.lower()]
            print(f"  ?? {model_id} is not in the catalogue" + (f" (similar: {close[:4]})" if close else ""))
    prompt = realistic_prompt() if args.realistic else PROMPT
    max_tokens = 8000 if args.realistic else 300
    print(f"--- probing {len(wanted)} model(s) with {'a forecast prompt' if args.realistic else 'a tiny prompt'}")
    for model_id in wanted:
        result = probe({"id": model_id, "params": {"max_tokens"}}, key, prompt=prompt, max_tokens=max_tokens, base_url=base_url)
        verdict = "OK " if result["ok"] else str(result["status"] or "ERR")
        parsed = ""
        if args.realistic and result["ok"]:
            try:
                parsed = f" → Probability {parse_probability(result['text']):.2f}"
            except Exception as error:
                parsed = f" → PARSE FAIL ({str(error)[:40]})"
        print(f"  {verdict:<4} {result['latency']:>6}s  {args.provider}/{model_id:<45} {result.get('error', '')[:70]}{parsed}")
        time.sleep(args.gap)


if __name__ == "__main__":
    main()
