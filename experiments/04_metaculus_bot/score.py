"""MEASURE step: coverage of the tournaments, then our own resolved forecasts scored locally.

Coverage first: prizes follow the *sum* of spot peer scores, so a question that closed without our forecast
(runner delayed, budget stop, crash) is a lost score. `--coverage` reports the last 7 days per tournament.

Binary: Brier and log score vs the 50% baseline. Multiple choice: log score vs uniform.
Numeric: log of the PMF bucket containing the resolution vs uniform.
A positive mean `skill_vs_baseline` is the minimum bar before expecting any prize money;
peer score (what prizes use) additionally requires beating the *other bots*, which we can't see locally.
"""
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from cdf import Scale  # noqa: E402
from engine.ledger import Ledger  # noqa: E402


def score_one(question: dict, payload: dict) -> dict | None:
    resolution = question.get("resolution")
    if resolution in (None, "annulled", "ambiguous"):
        return None
    kind = question["type"]
    if kind == "binary":
        probability = payload["probability_yes"]
        outcome = 1.0 if str(resolution).lower() == "yes" else 0.0
        p_outcome = probability if outcome else 1 - probability
        return {"brier": (probability - outcome) ** 2, "log": math.log(p_outcome), "baseline_log": math.log(0.5)}
    if kind == "multiple_choice":
        distribution = payload["probability_yes_per_category"]
        p_outcome = distribution.get(str(resolution), 1e-6)
        return {"log": math.log(p_outcome), "baseline_log": math.log(1 / len(distribution))}
    if kind in ("numeric", "discrete"):
        cdf = payload["continuous_cdf"]
        scaling = question["scaling"]
        scale = Scale(scaling["range_min"], scaling["range_max"], scaling.get("zero_point"),
                      question["open_lower_bound"], question["open_upper_bound"])
        if resolution in ("below_lower_bound", "above_upper_bound"):
            mass = cdf[0] if resolution == "below_lower_bound" else 1 - cdf[-1]
            buckets = 1
        else:
            location = scale.to_location(float(resolution))
            index = min(len(cdf) - 2, max(0, int(location * (len(cdf) - 1))))
            mass, buckets = cdf[index + 1] - cdf[index], len(cdf) - 1
        return {"log": math.log(max(mass, 1e-9)), "baseline_log": math.log(1 / buckets)}
    return None


def coverage(metaculus, tournaments, days: int = 7) -> dict:
    """Questions that closed in the last `days` days, and the ones we did not forecast."""
    from bot import already_forecast, close_time
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    report = {}
    for tournament in tournaments:
        recent = [post for status in ("closed", "resolved") for post in metaculus.list_posts(tournament, status)
                  if post.get("question") and close_time(post) and since < close_time(post) <= now]
        missed = [post["id"] for post in recent if not already_forecast(metaculus.post(post["id"]))]
        report[str(tournament)] = {"closed": len(recent), "missed": len(missed), "missed_posts": missed[:20]}
    return report


def main() -> None:
    from bot import TOURNAMENTS
    from clients import FixtureMetaculus, MetaculusClient
    offline = "--fixtures" in sys.argv
    ledger = Ledger()
    metaculus = FixtureMetaculus() if offline else MetaculusClient()
    if "--coverage" in sys.argv:
        report = coverage(metaculus, TOURNAMENTS["fixtures" if offline else "tournament"])
        missed = sum(item["missed"] for item in report.values())
        ledger.operation("metaculus-futureeval", "coverage", missed == 0, **report)
        print(json.dumps(report, indent=2))
    rows = ledger.connection.execute(
        "SELECT question_id, post_id, payload, MAX(id) FROM forecast GROUP BY question_id").fetchall()
    scores = []
    for row in rows:
        try:
            question = metaculus.post(row["post_id"])["question"]
        except Exception as error:
            print(f"[skip] post {row['post_id']}: {error}", file=sys.stderr)
            continue
        result = score_one(question, json.loads(row["payload"]))
        if result:
            scores.append(result)
    if not scores:
        print("no resolved forecasts yet")
        return
    skill = sum(score["log"] - score["baseline_log"] for score in scores) / len(scores)
    briers = [score["brier"] for score in scores if "brier" in score]
    report = {"resolved": len(scores), "mean_log_skill_vs_baseline": round(skill, 4),
              "mean_brier_binary": round(sum(briers) / len(briers), 4) if briers else None}
    ledger.evaluation("metaculus-futureeval", None, "skill-positive" if skill > 0 else "skill-negative", **report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
