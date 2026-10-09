"""Metaculus FutureEval bot — one cycle: discover open questions → forecast → submit → log costs.

Usage:
  python3 bot.py --mode fixtures            # offline: fixture questions + fake LLM, nothing leaves the machine
  python3 bot.py --mode test [--dry-run]    # bot-testing-area tournament (needs METACULUS_TOKEN, OPENROUTER_API_KEY)
  python3 bot.py --mode tournament          # Fall 2026 FutureEval (33121) + current MiniBench
  --profile full|standard|cheap|free        # force one profile; default: bot_config.json `profile_chain`
  --parallel N --time-budget S              # questions handled at once; stop taking new ones after S seconds
Rules honoured (FutureEval): no human in the loop, one forecast per question, a reasoning comment with each
forecast (retried on the next run if posting it failed: no comment, no prize), one bot per user.
Scoring is the spot peer score at close and questions stay open ~1.5 h: coverage matters, re-forecasting does not,
so the soonest-closing questions go first and several are forecast at once.
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
TOURNAMENTS = {
    "test": ["bot-testing-area"],
    "tournament": [33121, "minibench"],  # fall-futureeval-2026, rolling MiniBench (template constants)
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


def already_forecast(post: dict) -> bool:
    try:
        return post["question"]["my_forecasts"]["latest"]["forecast_values"] is not None
    except (KeyError, TypeError):
        return False


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


def retry_comments(ledger: Ledger, metaculus) -> None:
    pending = ledger.connection.execute(
        "SELECT id, post_id, comment FROM forecast WHERE submitted=1 AND commented=0").fetchall()
    for row in pending:
        try:
            metaculus.comment(row["post_id"], row["comment"])
            ledger.connection.execute("UPDATE forecast SET commented=1 WHERE id=?", (row["id"],))
            ledger.connection.commit()
            ledger.operation(STRATEGY_ID, "comment", True, post=row["post_id"], retry=True)
        except Exception as error:
            ledger.operation(STRATEGY_ID, "comment", False, post=row["post_id"], retry=True, error=str(error)[:300])


def exit_code(done: int, failed: int, discovered: int) -> int:
    """0 = the run did its job or had nothing to do; 1 = nothing worked (no tournament listed, or every question failed).
    Partial failures stay green: the ledger and the daily coverage report carry them, and a red scheduled run every
    20 min would only mail the repository owner."""
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
        try:
            forecast = future.result()
        except ForecastError as error:
            self.failed += 1
            self.book(post, profile, error.cost_usd, error.news_calls)
            self.ledger.operation(STRATEGY_ID, "forecast", False, duration, question=post["question"].get("id"),
                                  profile=profile, error=str(error)[:500])
            print(f"[fail] post {post['id']} [{profile}]: {error}", file=sys.stderr)
            return
        self.book(post, profile, forecast.cost_usd, forecast.news_calls)
        try:
            if not self.args.dry_run:
                self.metaculus.forecast(forecast.question_id, forecast.payload)
        except Exception as error:  # the API refused the payload: a failed question, already paid for
            self.failed += 1
            self.ledger.operation(STRATEGY_ID, "forecast", False, duration, question=forecast.question_id,
                                  profile=profile, error=f"submit: {str(error)[:450]}")
            print(f"[fail] post {post['id']} submit: {error}", file=sys.stderr)
            return
        cursor = self.ledger.connection.execute(
            "INSERT INTO forecast (question_id, post_id, tournament, ts, type, payload, summary, cost_usd,"
            " submitted, profile, comment, commented)"
            " VALUES (?,?,?,strftime('%Y-%m-%dT%H:%M:%f','now'),?,?,?,?,?,?,?,?)",
            (forecast.question_id, forecast.post_id, str(tournament), forecast.question_type,
             json.dumps(forecast.payload), json.dumps(forecast.summary), forecast.cost_usd,
             int(not self.args.dry_run), profile, forecast.comment, None if self.args.dry_run else 0))
        self.ledger.connection.commit()
        if not self.args.dry_run:
            try:
                self.metaculus.comment(forecast.post_id, forecast.comment)
                self.ledger.connection.execute("UPDATE forecast SET commented=1 WHERE id=?", (cursor.lastrowid,))
                self.ledger.connection.commit()
            except Exception as error:  # forecast is in: keep it, retry the comment next run
                self.ledger.operation(STRATEGY_ID, "comment", False, post=forecast.post_id, error=str(error)[:300])
        self.runs_ok, self.runs_total = self.runs_ok + forecast.runs_ok, self.runs_total + forecast.runs_total
        self.ledger.operation(STRATEGY_ID, "forecast", True, duration, question=forecast.question_id,
                              type=forecast.question_type, profile=profile, summary=forecast.summary,
                              submitted=not self.args.dry_run, runs_parsed=f"{forecast.runs_ok}/{forecast.runs_total}",
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
    pace_shares = {str(key): value.get("pace_share", 1.0) for key, value in raw.get("tournaments", {}).items()}

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
        retry_comments(ledger, metaculus)
    posts: dict[int, tuple[object, dict]] = {}
    discovered = 0
    for tournament in TOURNAMENTS[args.mode]:
        try:
            found = metaculus.open_posts(tournament)
        except Exception as error:
            ledger.operation(STRATEGY_ID, "discover", False, tournament=tournament, error=str(error)[:500])
            print(f"[discover] {tournament}: {error}", file=sys.stderr)
            continue
        ledger.operation(STRATEGY_ID, "discover", True, tournament=tournament, open_posts=len(found))
        discovered += 1
        for post in found:
            if post.get("question"):  # group posts are not part of the bot tournaments' scored set
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
        reap(block=False)
        if cycle.done + cycle.failed + len(running) >= args.max_questions:
            break
        if time.monotonic() - run_started > args.time_budget:
            cycle.stop = f"time budget of {args.time_budget}s reached; the next run continues"
            break
        closes = close_time(summary_post)
        if closes and closes - datetime.now(timezone.utc) < MIN_TIME_TO_CLOSE:
            cycle.skipped += 1
            continue
        post = metaculus.post(summary_post["id"])
        if already_forecast(post):
            cycle.skipped += 1
            continue
        while len(running) >= max(1, args.parallel):
            reap(block=True)
        try:
            config, _, skipped_profiles = budget.choose(chain, pace_shares.get(str(tournament), 1.0)) if budget \
                else (chain[0], 0.0, "")
        except BudgetExceeded as error:
            cycle.stop = str(error)
            ledger.operation(STRATEGY_ID, "budget", False, error=cycle.stop)
            print(f"[budget] stopping: {cycle.stop}", file=sys.stderr)
            break
        if skipped_profiles:
            print(f"[budget] post {post['id']} → {config['profile']} ({skipped_profiles})")
        future = executor.submit(forecasters[config["profile"]].forecast, post)
        running[future] = (tournament, post, config["profile"], time.monotonic())
    while running:
        reap(block=True)
    executor.shutdown(wait=True)

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
