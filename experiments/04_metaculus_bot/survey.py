"""MEASURE tool (offline after one fetch): what the tournaments actually look like, so settings rest on numbers.

    python3 survey.py --tournaments 33121 minibench --cache /tmp/survey --fetch 30

Pulls every post of each tournament (all statuses, groups included) through the bot's own client, caches the
JSON, then reports: counts by status and type, how long questions stay open, openings per day and per hour (UTC),
how many run at the same time, how many cite a resolution source, and, for a sample of those sources, whether the
page can be read and how noisy the excerpt is. No LLM call, no forecast: only METACULUS_TOKEN is needed.
"""
import argparse
import json
import re
import statistics
import sys
import urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent))

import clients  # noqa: E402
from bot import close_time, expand, load_dotenv  # noqa: E402
from sources import excerpt, extract_urls, fetch  # noqa: E402


def parse_time(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def all_posts(metaculus, tournament, limit: int = 100) -> list[dict]:
    """Every post of a tournament, whatever its status (list_posts filters on one status)."""
    posts, offset = [], 0
    for _ in range(50):
        query = urllib.parse.urlencode({"limit": limit, "offset": offset, "tournaments": tournament,
                                        "include_description": "true"})
        page = clients._request(f"{clients.METACULUS_API}/posts/?{query}", headers=metaculus.headers)
        results = page.get("results") or []
        posts.extend(results)
        if len(results) < limit or not page.get("next"):
            return posts
        offset += limit
    return posts


def noise_ratio(text: str) -> float:
    """Share of excerpt lines that look like page chrome: short, or repeated elsewhere in the excerpt."""
    lines = [line.strip() for line in text.splitlines() if line.strip() and line.strip() != "[…]"]
    if not lines:
        return 1.0
    seen = Counter(lines)
    noisy = sum(1 for line in lines if len(line) < 25 or seen[line] > 1)
    return noisy / len(lines)


def survey(posts: list[dict], tournament, fetch_limit: int, fetcher=fetch) -> dict:
    now = datetime.now(timezone.utc)
    report: dict = {"tournament": str(tournament), "posts": len(posts)}
    report["by_status"] = dict(Counter(p.get("status") or "?" for p in posts))
    report["group_posts"] = sum(1 for p in posts if p.get("group_of_questions"))
    questions = [q for p in posts for q in expand(p)]
    report["questions"] = len(questions)
    report["by_type"] = dict(Counter(q["question"].get("type") or "?" for q in questions))
    report["resolved"] = sum(1 for q in questions if q["question"].get("resolution") not in (None, ""))

    windows, opens, intervals = [], [], []
    for q in questions:
        opened = parse_time(q["question"].get("open_time") or q.get("open_time") or q.get("published_at"))
        closes = close_time(q)
        if opened and closes:
            windows.append((closes - opened).total_seconds() / 3600)
            opens.append(opened)
            intervals.append((opened, closes))
    if windows:
        report["open_window_hours"] = {"min": round(min(windows), 2), "median": round(statistics.median(windows), 2),
                                       "max": round(max(windows), 2)}
    if opens:
        first, last = min(opens), max(opens)
        days = max(1, (last - first).days + 1)
        report["openings"] = {"first": first.isoformat(), "last": last.isoformat(), "per_day": round(len(opens) / days, 1),
                              "by_weekday": dict(sorted(Counter(d.strftime("%a") for d in opens).items())),
                              "by_hour_utc": dict(sorted(Counter(d.hour for d in opens).items()))}
        # how many questions are open at the same moment (what one run must absorb)
        events = sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals])
        peak = level = 0
        for _, delta in events:
            level += delta
            peak = max(peak, level)
        report["max_simultaneously_open"] = peak
        last_week = [d for d in opens if d > now - timedelta(days=7)]
        report["openings_last_7_days"] = len(last_week)

    with_urls = [(q, extract_urls(q["question"])) for q in questions]
    cited = [(q, urls) for q, urls in with_urls if urls]
    report["questions_citing_a_source"] = len(cited)
    report["source_domains"] = dict(Counter(urllib.parse.urlparse(u).netloc for _, urls in cited for u in urls).most_common(12))

    sample: dict[str, dict] = {}
    for q, urls in cited:
        for url in urls:
            if url not in sample and len(sample) < fetch_limit:
                sample[url] = q["question"]
    fetched = {}
    if sample:
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {url: pool.submit(fetcher, url) for url in sample}
            for url, future in futures.items():
                try:
                    text = future.result(timeout=40)
                    cut = excerpt(text, sample[url].get("title") or "")
                    fetched[url] = {"ok": True, "chars": len(text), "excerpt_chars": len(cut), "noise": round(noise_ratio(cut), 2),
                                    "has_number": bool(re.search(r"\d", cut))}
                except Exception as error:
                    fetched[url] = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:80]}"}
    oks = [f for f in fetched.values() if f["ok"]]
    report["sources_sampled"] = len(fetched)
    report["sources_readable"] = len(oks)
    if oks:
        report["excerpt_noise"] = {"median": round(statistics.median(f["noise"] for f in oks), 2),
                                   "max": round(max(f["noise"] for f in oks), 2)}
    report["source_failures"] = dict(Counter(f["error"].split(":")[0] for f in fetched.values() if not f["ok"]))
    report["source_detail"] = fetched
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tournaments", nargs="+", default=["33121", "minibench"])
    parser.add_argument("--cache", type=Path, default=Path("/tmp/survey"))
    parser.add_argument("--fetch", type=int, default=20, help="resolution-source pages to try per tournament")
    parser.add_argument("--offline", action="store_true", help="use the cache only")
    args = parser.parse_args()
    load_dotenv(HERE.parent.parent / ".env")
    args.cache.mkdir(parents=True, exist_ok=True)
    metaculus = None if args.offline else clients.MetaculusClient()
    for tournament in args.tournaments:
        path = args.cache / f"posts_{tournament}.json"
        if path.exists() and (args.offline or True):
            posts = json.loads(path.read_text())
        else:
            posts = all_posts(metaculus, tournament)
            path.write_text(json.dumps(posts))
        report = survey(posts, tournament, args.fetch)
        (args.cache / f"survey_{tournament}.json").write_text(json.dumps(report, indent=1))
        detail = report.pop("source_detail")
        print(json.dumps(report, indent=1, ensure_ascii=False))
        for url, info in detail.items():
            print("  ", "OK " if info["ok"] else "KO ", url[:90], {k: v for k, v in info.items() if k != "ok"})


if __name__ == "__main__":
    main()
