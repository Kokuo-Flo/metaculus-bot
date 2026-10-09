"""MEASURE → IMPROVE loop, offline: replay aggregation rules on the stored model runs of resolved questions.

    python3 replay.py                 # ledger + Metaculus API (resolutions), prints a table
    python3 replay.py --json          # machine-readable

Every forecast stores one row per model run in the ledger's `run` table (bot.py). Once questions resolve, the
same runs can be re-aggregated under other rules with no new API call: the rule in production (median per model
family, mean of log-odds across families), stronger extremization, the plain mean or median of all runs, or a
single model family. Each rule is scored exactly like score.py (log score vs the uniform baseline) on each
resolved question, so rules are compared on the same questions. Below ~30 resolved questions the ranking is noise;
the table says so. Nothing here changes bot_config.json: the decision stays with a human reading the numbers.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent))

from cdf import pool  # noqa: E402
from forecaster import aggregate_binary, aggregate_cdfs, aggregate_options, logit  # noqa: E402
from score import score_one  # noqa: E402

MIN_RESOLVED = 30
BATCH_WINDOW_S = 5  # run rows of one forecast are inserted together; later re-forecasts form a new batch


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def latest_batches(connection) -> dict[int, dict]:
    """Per question: the parsed values of its most recent forecast, grouped by model."""
    rows = connection.execute(
        "SELECT question_id, post_id, ts, model, value FROM run WHERE ok=1 AND value IS NOT NULL ORDER BY ts").fetchall()
    by_question: dict[int, list] = defaultdict(list)
    for row in rows:
        by_question[row["question_id"]].append(row)
    batches = {}
    for question_id, question_rows in by_question.items():
        latest = _ts(question_rows[-1]["ts"])
        keep = [row for row in question_rows if abs((latest - _ts(row["ts"])).total_seconds()) <= BATCH_WINDOW_S]
        by_model: dict[str, list] = defaultdict(list)
        for row in keep:
            by_model[row["model"]].append(json.loads(row["value"]))
        batches[question_id] = {"post_id": keep[0]["post_id"], "by_model": dict(by_model)}
    return batches


def _clip(value: float, clamp: tuple[float, float]) -> float:
    return min(clamp[1], max(clamp[0], value))


def rules(kind: str, by_model: dict[str, list], config: dict, options: list[str] | None = None) -> dict[str, dict]:
    """Candidate aggregation rules → Metaculus payloads, for one question."""
    clamp = tuple(config.get("binary_clamp", (0.015, 0.985)))
    floor = config.get("mc_floor", 0.005)
    out: dict[str, dict] = {}
    if kind == "binary":
        all_runs = [value for values in by_model.values() for value in values]

        def payload(probability: float) -> dict:
            return {"probability_yes": _clip(probability, clamp), "probability_yes_per_category": None,
                    "continuous_cdf": None}

        out["current"] = payload(aggregate_binary(by_model, clamp, 1.0))
        for strength in (1.3, 1.6):
            out[f"extremize_{strength}"] = payload(aggregate_binary(by_model, clamp, strength))
        out["mean_all"] = payload(statistics.mean(all_runs))
        out["median_all"] = payload(statistics.median(all_runs))
        out["logodds_mean_all"] = payload(1 / (1 + 2.718281828 ** -statistics.mean(logit(_clip(v, clamp)) for v in all_runs)))
        for model, values in by_model.items():
            out[f"only:{model}"] = payload(statistics.median(values))
    elif kind == "multiple_choice":
        options = options or sorted({option for values in by_model.values() for value in values for option in value})

        def payload(distribution: dict) -> dict:
            floored = {option: max(floor, distribution.get(option, 0.0)) for option in options}
            total = sum(floored.values())
            return {"probability_yes": None, "continuous_cdf": None,
                    "probability_yes_per_category": {option: value / total for option, value in floored.items()}}

        out["current"] = payload(aggregate_options(by_model, options, floor))
        all_runs = [value for values in by_model.values() for value in values]
        out["mean_all"] = payload({option: statistics.mean(run.get(option, 0.0) for run in all_runs) for option in options})
        for model, values in by_model.items():
            out[f"only:{model}"] = payload({option: statistics.mean(run.get(option, 0.0) for run in values)
                                            for option in options})
    elif kind in ("numeric", "discrete"):
        def payload(cdf: list[float]) -> dict:
            return {"probability_yes": None, "probability_yes_per_category": None, "continuous_cdf": cdf}

        out["current"] = payload(aggregate_cdfs(by_model))
        out["pool_all"] = payload(pool([cdf for cdfs in by_model.values() for cdf in cdfs]))
        for model, cdfs in by_model.items():
            out[f"only:{model}"] = payload(pool(cdfs))
    return out


def question_for(metaculus, post_id: int, question_id: int) -> dict | None:
    from bot import expand
    post = metaculus.post(post_id)
    for pseudo in expand(post):
        if pseudo["question"].get("id") == question_id:
            return pseudo["question"]
    return None


def replay(connection, metaculus, config: dict) -> dict:
    per_question: dict[int, dict[str, float]] = {}
    unresolved = failed = 0
    for question_id, batch in latest_batches(connection).items():
        try:
            question = question_for(metaculus, batch["post_id"], question_id)
        except Exception:
            failed += 1
            continue
        if not question or question.get("resolution") in (None, "", "annulled", "ambiguous"):
            unresolved += 1
            continue
        skills = {}
        for name, payload in rules(question["type"], batch["by_model"], config, question.get("options")).items():
            try:
                score = score_one(question, payload)
            except Exception:
                score = None
            if score:
                skills[name] = score["log"] - score["baseline_log"]
        if "current" in skills:
            per_question[question_id] = skills
    table = {}
    for name in sorted({name for skills in per_question.values() for name in skills}):
        values = [skills[name] for skills in per_question.values() if name in skills]
        wins = sum(1 for skills in per_question.values() if name in skills and skills[name] > skills["current"] + 1e-12)
        table[name] = {"n": len(values), "mean_skill": round(statistics.mean(values), 4) if values else None,
                       "wins_vs_current": wins}
    return {"resolved": len(per_question), "unresolved": unresolved, "fetch_failed": failed,
            "reliable": len(per_question) >= MIN_RESOLVED, "rules": table}


def main() -> None:
    from bot import CONFIG, load_dotenv
    from clients import FixtureMetaculus, MetaculusClient
    from engine.ledger import Ledger
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--fixtures", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    load_dotenv(HERE.parent.parent / ".env")
    config = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    ledger = Ledger(args.ledger) if args.ledger else Ledger()
    report = replay(ledger.connection, FixtureMetaculus() if args.fixtures else MetaculusClient(), config)
    if args.json:
        print(json.dumps(report, indent=1))
        return
    print(f"resolved={report['resolved']} unresolved={report['unresolved']} fetch_failed={report['fetch_failed']}"
          + ("" if report["reliable"] else f"  (fewer than {MIN_RESOLVED} resolved: ranking is noise)"))
    for name, row in sorted(report["rules"].items(), key=lambda item: -(item[1]["mean_skill"] or -9)):
        print(f"  {name:<40} n={row['n']:<4} mean log skill vs baseline={row['mean_skill']!s:<8} "
              f"beats current on {row['wins_vs_current']}")


if __name__ == "__main__":
    main()
