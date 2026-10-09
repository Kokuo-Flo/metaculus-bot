"""Metaculus FutureEval bot — one cycle: discover open questions → forecast → submit → log costs.

Usage:
  python3 bot.py --mode fixtures            # offline: fixture questions + fake LLM, nothing leaves the machine
  python3 bot.py --mode test [--dry-run]    # bot-testing-area tournament (needs METACULUS_TOKEN, OPENROUTER_API_KEY)
  python3 bot.py --mode tournament          # every tournament where bot accounts may win prizes (TOURNAMENTS)
  --profile full|standard|cheap|free        # force one profile; default: bot_config.json `profile_chain`
  --parallel N --time-budget S              # questions handled at once; stop taking new ones after S seconds
Rules honoured (Metaculus): no human in the loop, one entry per competition (the bot account only), a reasoning
note with each pass on a post (private, short, one per post per run; retried next run if posting failed: no comment,
no prize), one bot per user.
FutureEval/MiniBench score the last forecast at close and stay open ~1.5 h: coverage matters, re-forecasting does
not. Market Pulse and Animal Futures score over the whole lifetime, so `tournaments.<id>.reforecast_days` refreshes
those forecasts periodically. Group posts are unpacked into their sub-questions.
Spend: only capped, prepaid credits, paced to last the season (see budget.py).
"""
import argparse
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))  # repo root → engine.*

from budget import Budget, BudgetExceeded  # noqa: E402
from clients import AskNews, FakeLLM, FakeNews, FixtureMetaculus, MetaculusClient, OpenRouter, model_catalog  # noqa: E402
from engine.costs import usd_to_eur  # noqa: E402
from engine.ledger import Ledger  # noqa: E402
from forecaster import ForecastError, Forecaster, resolve_config  # noqa: E402
from sources import fetch  # noqa: E402

STRATEGY_ID = "metaculus-futureeval"
# Bot accounts are prize-eligible here (tournament rules checked 2026-10-09; everywhere else bots forecast for
# practice only). Ids/slugs follow forecasting-tools' MetaculusClient constants.
TOURNAMENTS = {
    "test": ["bot-testing-area"],
    "tournament": [33121, "minibench", "market-pulse-26q4", 33016],  # Fall FutureEval, MiniBench, Market Pulse 26Q4, Animal Futures
    "fixtures": ["fixtures"],
}
CONFIG = HERE / "bot_config.json"
MIN_TIME_TO_CLOSE = timedelta(minutes=5)  # a forecast that cannot land before close only costs money


def load_dotenv(path: Path) -> None:
    """Minimal .env loader for local runs (never overrides variables already set)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def init_store(connection: sqlite3.Connection) -> None:
    connection.execute("""CREATE TABLE IF NOT EXISTS forecast (
        id INTEGER PRIMARY KEY AUTOINCREMENT, question_id INTEGER, post_id INTEGER, tournament TEXT, ts TEXT,
        type TEXT, payload TEXT, summary TEXT, cost_usd REAL, submitted INTEGER)""")
    columns = {row[1] for row in connection.execute("PRAGMA table_info(forecast)")}
    for column, kind in (("profile", "TEXT"), ("comment", "TEXT"), ("commented", "INTEGER")):
        if column not in columns:  # session-1 ledgers predate these columns
            connection.execute(f"ALTER TABLE forecast ADD COLUMN {column} {kind}")
    connection.commit()


def expand(post: dict) -> list[dict]:
    """A post as a list of forecastable (post, question) pairs: itself, or one pseudo-post per sub-question
    of a group (the official client does the same: group-level text copied onto each sub-question)."""
    if post.get("question"):
        return [post]
    group = post.get("group_of_questions")
    if not group:
        return []
    pseudo_posts = []
    for sub in group.get("questions") or []:
        question = {**sub, "description": group.get("description") or "",
                    "resolution_criteria": group.get("resolution_criteria") or "",
                    "fine_print": group.get("fine_print") or ""}
        label = (sub.get("label") or "").strip()
        base = sub.get("title") or post.get("title") or ""
        question["title"] = f"{base} — {label}" if label and label not in base else base
        pseudo_posts.append({**post, "question": question, "group_label": label})
    return pseudo_posts


def last_forecast_time(question: dict) -> datetime | None:
    """When our latest forecast on this question was made; None if there is none. A forecast whose timestamp
    is missing counts as made long ago (so one-shot tournaments skip it and periodic ones refresh it)."""
    try:
        latest = question["my_forecasts"]["latest"]
        if latest["forecast_values"] is None:
            return None
    except (KeyError, TypeError):
        return None
    try:
        return datetime.fromtimestamp(float(latest["start_time"]), tz=timezone.utc)
    except (KeyError, TypeError, ValueError):
        return datetime.fromtimestamp(0, tz=timezone.utc)


def already_forecast(post: dict) -> bool:
    return last_forecast_time(post["question"]) is not None


def needs_forecast(post: dict, reforecast_days: float | None, now: datetime | None = None) -> bool:
    """One forecast per question by default; with `reforecast_days`, refresh once the last one is that old."""
    last = last_forecast_time(post["question"])
    if last is None:
        return True
    if not reforecast_days:
        return False
    return (now or datetime.now(timezone.utc)) - last >= timedelta(days=reforecast_days)


def close_time(post: dict) -> datetime | None:
    raw = (post.get("question") or {}).get("scheduled_close_time") or post.get("scheduled_close_time")
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None


def asknews_calls_this_month(ledger: Ledger) -> int:
    month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return ledger.connection.execute(
        "SELECT COUNT(*) FROM operation WHERE strategy_id=? AND kind='asknews' AND ts >= ?",
        (STRATEGY_ID, month_start.isoformat())).fetchone()[0]


def combined_note(rows: list[sqlite3.Row]) -> str:
    """One private note per post and run: the full reasoning for a single question, a digest for a group."""
    if len(rows) == 1:
        return rows[0]["comment"]
    lines = [f"Automated forecasts on {len(rows)} sub-questions (same pipeline for each):"]
    for row in rows:
        summary = json.loads(row["summary"]) if row["summary"] else ""
        lines.append(f"- q{row['question_id']} ({row['type']}): {summary}")
    lines.append("\n" + rows[0]["comment"].split("\n\n## Run 1")[0][:2500])  # research digest of the first one
    return "\n".join(lines)


def post_notes(ledger: Ledger, metaculus) -> None:
    """Post pending notes, grouped per post; mark their forecast rows as commented."""
    pending = ledger.connection.execute(
        "SELECT id, post_id, question_id, type, summary, comment FROM forecast WHERE submitted=1 AND commented=0"
        " ORDER BY post_id, id").fetchall()
    by_post: dict[int, list[sqlite3.Row]] = {}
    for row in pending:
        by_post.setdefault(row["post_id"], []).append(row)
    for post_id, rows in by_post.items():
        try:
            metaculus.comment(post_id, combined_note(rows))
            ledger.connection.execute(
                f"UPDATE forecast SET commented=1 WHERE id IN ({','.join('?' * len(rows))})", [r["id"] for r in rows])
            ledger.connection.commit()
            ledger.operation(STRATEGY_ID, "comment", True, post=post_id, forecasts=len(rows))
        except Exception as error:  # forecasts are in: retry the note next run
            ledger.operation(STRATEGY_ID, "comment", False, post=post_id, error=str(error)[:300])


def exit_code(done: int, failed: int, discovered: int) -> int:
    """0 = the run did its job or had nothing to do; 1 = nothing worked (no tournament listed, or every question
    failed). Partial failures stay green: the ledger and the daily coverage report carry them, and a red scheduled
    run every 20 min would only mail the repository owner."""
    return 1 if discovered == 0 or (failed and not done) else 0


class Cycle:
    """One run: shared state for the question loop (ledger writes and Metaculus posts stay on the main thread)."""

    def __init__(self, args, ledger: Ledger, metaculus, llm, news, source: str, hide_values: bool):
        self.args, self.ledger, self.metaculus, self.llm, self.news = args, ledger, metaculus, llm, news
        self.source, self.hide_values = source, hide_values
        self.done = self.failed = self.skipped = self.runs_ok = self.runs_total = 0
        self.stop: str | None = None

    def book(self, post: dict, profile: str, cost_usd: float, news_calls: int) -> None:
        if cost_usd:
            self.ledger.money(STRATEGY_ID, "api_cost", usd_to_eur(cost_usd), self.source,
                              note=f"post{post['id']} profile={profile}")
        for _ in range(news_calls if self.source == "measured" else 0):
            self.ledger.operation(STRATEGY_ID, "asknews", True, post=post["id"])

    def finish(self, future: Future, tournament, post: dict, profile: str, started: float) -> None:
        duration = time.monotonic() - started
        question_id = post["question"].get("id")
        try:
            forecast = future.result()
        except ForecastError as error:
            self.failed += 1
            self.book(post, profile, error.cost_usd, error.news_calls)
            self.ledger.operation(STRATEGY_ID, "forecast", False, duration, question=question_id, profile=profile,
                                  error=str(error)[:500])
            print(f"[fail] q{question_id} [{profile}]: {error}", file=sys.stderr)
            return
        self.book(post, profile, forecast.cost_usd, forecast.news_calls)
        try:
            if not self.args.dry_run:
                self.metaculus.forecast(forecast.question_id, forecast.payload)
        except Exception as error:  # the API refused the payload: a failed question, already paid for
            self.failed += 1
            self.ledger.operation(STRATEGY_ID, "forecast", False, duration, question=forecast.question_id,
                                  profile=profile, error=f"submit: {str(error)[:450]}")
            print(f"[fail] q{question_id} submit: {error}", file=sys.stderr)
            return
        self.ledger.connection.execute(
            "INSERT INTO forecast (question_id, post_id, tournament, ts, type, payload, summary, cost_usd,"
            " submitted, profile, comment, commented)"
            " VALUES (?,?,?,strftime('%Y-%m-%dT%H:%M:%f','now'),?,?,?,?,?,?,?,?)",
            (forecast.question_id, forecast.post_id, str(tournament), forecast.question_type,
             json.dumps(forecast.payload), json.dumps(forecast.summary), forecast.cost_usd,
             int(not self.args.dry_run), profile, forecast.comment, None if self.args.dry_run else 0))
        self.ledger.connection.commit()
        self.runs_ok, self.runs_total = self.runs_ok + forecast.runs_ok, self.runs_total + forecast.runs_total
        self.ledger.operation(STRATEGY_ID, "forecast", True, duration, question=forecast.question_id,
                              type=forecast.question_type, profile=profile, summary=forecast.summary,
                              tournament=str(tournament), submitted=not self.args.dry_run,
                              runs_parsed=f"{forecast.runs_ok}/{forecast.runs_total}",
                              cost_usd=round(forecast.cost_usd, 4))
        self.done += 1
        shown = "(hidden while open)" if self.hide_values else forecast.summary
        print(f"[ok] q{forecast.question_id} {forecast.question_type} [{profile}]: {shown} "
              f"(${forecast.cost_usd:.3f}, runs {forecast.runs_ok}/{forecast.runs_total}, {duration:.0f}s)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=TOURNAMENTS, default="fixtures")
    parser.add_argument("--dry-run", action="store_true", help="forecast but do not submit")
    parser.add_argument("--max-questions", type=int, default=60)
    parser.add_argument("--parallel", type=int, default=3, help="questions forecast at the same time")
    parser.add_argument("--time-budget", type=int, default=1500,
                        help="seconds after which no new question is started (GitHub job timeout is 45 min)")
    parser.add_argument("--profile", default=None, help="force one profile from bot_config.json")
    parser.add_argument("--ledger", type=Path, default=None)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")

    offline = args.mode == "fixtures"
    ledger = Ledger(args.ledger) if args.ledger else Ledger()
    ledger.register(STRATEGY_ID, "Metaculus FutureEval bot tournament", status="paper" if offline else "active",
                    proof_level=1)
    init_store(ledger.connection)
    raw = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    chain = [resolve_config(raw, name) for name in
             ([args.profile] if args.profile else raw.get("profile_chain") or [raw.get("profile")])]
    limits = chain[0]["budget"]
    tournament_settings = {str(key): value for key, value in raw.get("tournaments", {}).items()}

    metaculus = FixtureMetaculus() if offline else MetaculusClient()
    catalog = {} if offline else model_catalog()
    llm = FakeLLM() if offline else OpenRouter(catalog=catalog)
    news = FakeNews() if offline else AskNews.from_env(
        limits["asknews_calls_per_month"] - asknews_calls_this_month(ledger))
    fetcher = (lambda url: f"[offline] would fetch {url}") if offline else fetch
    forecasters = {config["profile"]: Forecaster(llm, config, news, fetcher) for config in chain}
    source = "estimated" if offline else "measured"
    budget = None if offline else Budget(limits, llm, ledger, STRATEGY_ID, usd_to_eur,
                                         {model: info["pricing"] for model, info in catalog.items()})
    # Logs of a public repo's Actions runs are public, and a question stays open ~1.5 h after we forecast it:
    # never print live tournament forecasts there (the ledger keeps them; Metaculus shows them after close).
    hide_values = os.environ.get("GITHUB_ACTIONS") == "true" and args.mode == "tournament" and not args.dry_run
    cycle = Cycle(args, ledger, metaculus, llm, news, source, hide_values)

    ledger.heartbeat(STRATEGY_ID)
    if not args.dry_run:
        post_notes(ledger, metaculus)  # notes left over from a previous run
    posts: dict[int, tuple[object, dict]] = {}
    discovered = 0
    for tournament in TOURNAMENTS[args.mode]:
        try:
            found = metaculus.open_posts(tournament)
        except Exception as error:
            ledger.operation(STRATEGY_ID, "discover", False, tournament=tournament, error=str(error)[:500])
            print(f"[discover] {tournament}: {error}", file=sys.stderr)
            continue
        discovered += 1
        ledger.operation(STRATEGY_ID, "discover", True, tournament=tournament, open_posts=len(found))
        for post in found:
            if post.get("question") or post.get("group_of_questions"):
                posts.setdefault(post["id"], (tournament, post))
    far = datetime.max.replace(tzinfo=timezone.utc)
    queue = sorted(posts.values(), key=lambda item: close_time(item[1]) or far)  # soonest window first

    run_started = time.monotonic()
    executor = ThreadPoolExecutor(max_workers=max(1, args.parallel))
    running: dict[Future, tuple] = {}

    def reap(block: bool) -> None:
        if not running:
            return
        finished, _ = wait(list(running), return_when=FIRST_COMPLETED) if block else wait(list(running), timeout=0)
        for future in finished:
            cycle.finish(future, *running.pop(future))

    for tournament, summary_post in queue:
        if cycle.stop:
            break
        settings = tournament_settings.get(str(tournament), {})
        detail = None
        for post in expand(summary_post):
            reap(block=False)
            if cycle.done + cycle.failed + len(running) >= args.max_questions:
                cycle.stop = f"max-questions {args.max_questions} reached; the next run continues"
                break
            if time.monotonic() - run_started > args.time_budget:
                cycle.stop = f"time budget of {args.time_budget}s reached; the next run continues"
                break
            closes = close_time(post)
            if closes and closes - datetime.now(timezone.utc) < MIN_TIME_TO_CLOSE:
                cycle.skipped += 1
                continue
            if detail is None:
                try:
                    detail = metaculus.post(summary_post["id"])  # fresh my_forecasts; one fetch per post
                except Exception as error:  # one refused post must not end the run: it would, on every run
                    cycle.failed += 1
                    ledger.operation(STRATEGY_ID, "post", False, post=summary_post["id"], error=str(error)[:300])
                    print(f"[fail] post {summary_post['id']} fetch: {error}", file=sys.stderr)
                    break
            fresh = [p for p in expand(detail) if p["question"].get("id") == post["question"].get("id")]
            post = fresh[0] if fresh else post
            if not needs_forecast(post, settings.get("reforecast_days")):
                cycle.skipped += 1
                continue
            while len(running) >= max(1, args.parallel):
                reap(block=True)
            try:
                config, _, skipped_profiles = budget.choose(chain, settings.get("pace_share", 1.0)) if budget \
                    else (chain[0], 0.0, "")
            except BudgetExceeded as error:
                cycle.stop = str(error)
                ledger.operation(STRATEGY_ID, "budget", False, error=cycle.stop)
                print(f"[budget] stopping: {cycle.stop}", file=sys.stderr)
                break
            except Exception as error:  # key status or prices unreachable: stop cleanly, the next run retries
                cycle.stop = f"budget check failed: {str(error)[:200]}"
                ledger.operation(STRATEGY_ID, "budget", False, error=cycle.stop)
                print(f"[budget] stopping: {cycle.stop}", file=sys.stderr)
                break
            if skipped_profiles:
                print(f"[budget] q{post['question'].get('id')} → {config['profile']} ({skipped_profiles})")
            future = executor.submit(forecasters[config["profile"]].forecast, post)
            running[future] = (tournament, post, config["profile"], time.monotonic())
    while running:
        reap(block=True)
    executor.shutdown(wait=True)
    if not args.dry_run:
        post_notes(ledger, metaculus)

    parse_rate = f"{cycle.runs_ok}/{cycle.runs_total}" if cycle.runs_total else "n/a"
    ledger.operation(STRATEGY_ID, "cycle", cycle.failed == 0 and not cycle.stop, mode=args.mode, done=cycle.done,
                     failed=cycle.failed, skipped=cycle.skipped, runs_parsed=parse_rate,
                     cost_usd=round(llm.total_cost_usd, 4), stop=cycle.stop,
                     duration_s=round(time.monotonic() - run_started))
    print(f"done={cycle.done} failed={cycle.failed} skipped={cycle.skipped} runs_parsed={parse_rate} "
          f"llm_cost=${llm.total_cost_usd:.3f} asknews_calls={news.calls if news else 'off'}"
          + (f" stop={cycle.stop}" if cycle.stop else ""))
    if offline:
        print(f"fixture submissions captured: {len(metaculus.submitted)}")
    return exit_code(cycle.done, cycle.failed, discovered)


if __name__ == "__main__":
    sys.exit(main())
