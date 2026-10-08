"""Forecasting pipeline: sources + news + research brief → multi-model ensemble → aggregation → Metaculus payload.

Differences vs the official template (informed by the Spring 2026 results write-up, where every top-10
bot used frontier GPT-5.x models + dedicated research and the plain template bot ranked 18th):
- research = the question's own resolution sources (fetched live: the status quo) + AskNews latest news (free for
  participants) + one search-grounded brief, gathered in parallel and shared by all forecasters
- ensemble across *different* model families (errors decorrelate), not N samples of one model; each family is
  summarised first (median), then families are averaged in log-odds space, so a family with a failed run or a
  wild sample does not dominate
- numeric CDFs are pooled by mean (keeps API constraints) instead of elementwise median
"""
import json
import math
import re
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date

from cdf import Scale, percentiles_to_cdf, pool
from clients import Meter
from sources import fetch, resolution_sources

NEWS_CHARS, BRIEF_CHARS, SOURCES_CHARS = 8_000, 12_000, 9_000  # caps keep prompts under budget.INPUT_TOKENS_BOUND
# Metaculus: "Long bot comments that don't add value will be considered spam [...] Private comments are fine within
# reasonable limits." Ours are private notes; keep them to a few thousand characters: research digest + run endings.
COMMENT_RESEARCH_CHARS, COMMENT_RUN_CHARS = 1_500, 500

DEFAULT_CONFIG = {
    "research_model": "openai/gpt-5.6-terra",
    "research_web": "native",  # provider search, covered by Metaculus credits (Perplexity is not enabled on them)
    "asknews_articles": 8,
    "models": ["openai/gpt-5.6-terra-pro", "anthropic/claude-opus-5.5"],
    "runs_per_model": 3,
    "binary_clamp": [0.015, 0.985],
    "mc_floor": 0.005,
    "extremize": 1.0,  # >1 pushes the pooled log-odds away from 50 %; tune only on measured calibration
    "max_tokens": 12000,  # reasoning tokens count toward it
    "reasoning_effort": None,
    "budget": {"max_usd_per_run": 15.0, "max_usd_per_day": 25.0, "pace_usd_per_day": 6.0, "key_reserve_usd": 0.5,
               "allow_uncapped_key": False, "season_end": None, "asknews_calls_per_month": 900},
}


def resolve_config(raw: dict, profile: str | None = None) -> dict:
    """bot_config.json = shared settings + named model profiles (full / standard / cheap / free)."""
    profiles = raw.get("profiles", {})
    name = profile or raw.get("profile")
    if name and name not in profiles:
        raise ValueError(f"unknown profile {name!r}; known: {sorted(profiles)}")
    shared = {key: value for key, value in raw.items()
              if key not in ("profiles", "profile", "profile_chain", "tournaments")}
    config = {**DEFAULT_CONFIG, **shared, **(profiles.get(name) or {})}
    config["budget"] = {**DEFAULT_CONFIG["budget"], **raw.get("budget", {})}
    config["profile"] = name
    return config


RESEARCH_PROMPT = """RESEARCH BRIEF for a forecasting question. Today is {today}.
Question: {title}
Resolution criteria: {criteria}
Fine print: {fine_print}

Search for the most recent, relevant information. Report concisely:
1. Current status quo and latest developments (with dates and sources).
2. Relevant base rates / historical frequencies.
3. Scheduled events before resolution that could change the outcome.
4. What markets, experts or official forecasts say (if any).
Do not give a final probability."""

COMMON = """You are a superforecaster competing in a forecasting tournament scored with proper scoring rules.
Question: {title}
Background: {background}
Resolution criteria: {criteria}
Fine print: {fine_print}
Today is {today}. Question closes {close} and resolves {resolve}.

Research:
{research}

If resolution sources are given, start from the latest value they show: it is the status quo.
Think step by step, briefly:
(a) Time left until resolution; (b) status quo outcome if nothing changes; (c) base rate from comparable cases;
(d) key evidence that moves you away from the base rate, and by how much;
(e) one surprise scenario in each direction.
Remember: the world changes slowly; resolution criteria are applied literally; avoid overconfidence but do not
hedge toward the middle without reason."""

BINARY_TAIL = """
End with exactly one line: "Probability: ZZ%" (0-100, decimals allowed)."""

MC_TAIL = """
Options (exact labels) — OPTIONS_JSON: {options_json}
Leave some probability on every option. End with one line per option, in order, formatted
"<option label>: XX%" (Option_A: XX% style), summing to 100%."""

NUMERIC_TAIL = """
Units: {unit}. {bounds_text}
BOUNDS_JSON: {lower} {upper}
Write plain numbers in the question's units (no scientific notation, no "million"/"k" shorthand).
Set wide 90/10 intervals for unknown unknowns.
End with exactly these lines, values increasing:
Percentile 10: X
Percentile 20: X
Percentile 40: X
Percentile 60: X
Percentile 80: X
Percentile 90: X"""


@dataclass
class Forecast:
    question_id: int
    post_id: int
    question_type: str
    payload: dict
    summary: object
    comment: str
    cost_usd: float
    news_calls: int = 0
    runs: list[str] = field(default_factory=list)
    runs_ok: int = 0
    runs_total: int = 0


class ForecastError(RuntimeError):
    """A failed question still consumed credits and news calls: the caller books them."""

    def __init__(self, message: str, cost_usd: float = 0.0, news_calls: int = 0):
        super().__init__(message)
        self.cost_usd, self.news_calls = cost_usd, news_calls


# --- parsing ---
NUMBER = r"([-−–]?[\d,]*\.?\d+)"
MARKUP = r"\s*\**\s*"  # models wrap the final answer in **bold** on either side of the colon
MULTIPLIERS = {"k": 1e3, "thousand": 1e3, "million": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9, "trillion": 1e12}


def to_float(raw: str) -> float:
    return float(raw.replace("−", "-").replace("–", "-").replace(",", ""))


def parse_probability(text: str) -> float:
    """Last 'Probability: ZZ%' line. Without the % sign, 0–1 is read as a fraction and 1–100 as a percentage."""
    matches = re.findall(rf"Probability{MARKUP}[:=]{MARKUP}(\d+(?:[.,]\d+)?)\s*(%?)", text, re.IGNORECASE)
    if not matches:
        raise ValueError("no 'Probability: ZZ%' line")
    raw, percent = matches[-1]
    value = float(raw.replace(",", "."))
    probability = value / 100 if percent or value > 1 else value
    if not 0 <= probability <= 1:
        raise ValueError(f"probability {probability} out of range")
    return probability


def parse_options(text: str, options: list[str]) -> dict[str, float]:
    found: dict[str, float] = {}
    for line in text.splitlines():
        for option in options:
            match = re.match(rf"\s*[*\-]*\s*{re.escape(option)}\s*\**\s*:\s*\**\s*([\d.]+)\s*%?", line, re.IGNORECASE)
            if match:
                found[option] = float(match.group(1))
    if len(found) != len(options):
        raise ValueError(f"parsed {len(found)}/{len(options)} options")
    total = sum(found.values())
    if total <= 0:
        raise ValueError("options sum to zero")
    return {option: value / total for option, value in found.items()}


def parse_percentiles(text: str) -> dict[float, float]:
    values: dict[float, float] = {}
    pattern = rf"Percentile\s+(\d+){MARKUP}:{MARKUP}\$?\s*{NUMBER}\s*(k\b|thousand|million|mn\b|billion|bn\b|trillion)?"
    for percent, raw, unit in re.findall(pattern, text, re.IGNORECASE):
        values[float(percent)] = to_float(raw) * MULTIPLIERS.get(unit.lower(), 1.0)
    if len(values) < 4:
        raise ValueError(f"parsed only {len(values)} percentiles")
    return values


# --- aggregation ---
def logit(p: float) -> float:
    return math.log(p / (1 - p))


def aggregate_binary(by_model: dict[str, list[float]], clamp: tuple[float, float], extremize: float = 1.0) -> float:
    """Median within each model family, mean of log-odds across families, clamped."""
    low, high = clamp
    medians = [min(high, max(low, statistics.median(values))) for values in by_model.values() if values]
    pooled = 1 / (1 + math.exp(-extremize * statistics.mean(logit(m) for m in medians)))
    return min(high, max(low, pooled))


def aggregate_options(by_model: dict[str, list[dict]], options: list[str], floor: float) -> dict[str, float]:
    per_model = [{option: statistics.mean(run[option] for run in runs) for option in options}
                 for runs in by_model.values() if runs]
    mean = {option: statistics.mean(dist[option] for dist in per_model) for option in options}
    floored = {option: max(floor, value) for option, value in mean.items()}
    total = sum(floored.values())
    return {option: value / total for option, value in floored.items()}


def aggregate_cdfs(by_model: dict[str, list[list[float]]]) -> list[float]:
    return pool([pool(cdfs) for cdfs in by_model.values() if cdfs])


def failure_summary(texts: list[str]) -> str:
    """'2× HTTP 429, 1× unparsable': why runs were lost, kept in the ledger error (the transcripts are not)."""
    counts: dict[str, int] = {}
    for text in texts:
        if " failed] " not in text:
            continue
        reason = text.split(" failed] ", 1)[1]
        match = re.search(r"HTTP (\d+)", reason)
        key = f"HTTP {match.group(1)}" if match else ("empty answer" if "empty answer" in reason else "unparsable")
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{n}× {key}" for key, n in sorted(counts.items(), key=lambda item: (-item[1], item[0]))) \
        or "no failed runs"


class Forecaster:
    def __init__(self, llm, config: dict | None = None, news=None, fetcher=fetch):
        self.llm, self.news, self.fetcher = llm, news, fetcher
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    def _complete(self, model: str, prompt: str, meter: Meter, **extra):
        return self.llm.complete(model, prompt, max_tokens=self.config["max_tokens"],
                                 reasoning=self.config["reasoning_effort"], meter=meter, **extra)

    def research(self, question: dict, meter: Meter) -> str:
        """Sources, news and brief are independent: fetch them together, keep whatever succeeded."""
        jobs = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            if self.fetcher:
                jobs["sources"] = executor.submit(resolution_sources, question, self.fetcher)
            if self.news:
                jobs["news"] = executor.submit(self.news.latest, question["title"], self.config["asknews_articles"],
                                               meter)
            if self.config["research_model"]:
                prompt = RESEARCH_PROMPT.format(
                    today=date.today().isoformat(), title=question["title"],
                    criteria=question.get("resolution_criteria") or "", fine_print=question.get("fine_print") or "")
                jobs["brief"] = executor.submit(self._complete, self.config["research_model"], prompt, meter,
                                                web=self.config["research_web"])
        parts = []
        for name, future in jobs.items():
            try:
                result = future.result()
            except Exception as error:  # research is best effort; the forecast still happens
                parts.append(f"## {name}\n[unavailable: {str(error)[:150]}]")
                continue
            if name == "sources" and result:
                parts.append(result[:SOURCES_CHARS])
            elif name == "news" and result:
                parts.append(f"## Recent news (AskNews)\n{result[:NEWS_CHARS]}")
            elif name == "brief":
                parts.append(f"## Research brief\n{result.text[:BRIEF_CHARS]}")
        return "\n\n".join(parts) or "No research available."

    def _context(self, question: dict, research: str) -> dict:
        return {
            "title": question["title"], "background": (question.get("description") or "")[:6000],
            "criteria": question.get("resolution_criteria") or "", "fine_print": question.get("fine_print") or "",
            "today": date.today().isoformat(), "close": question.get("scheduled_close_time"),
            "resolve": question.get("scheduled_resolve_time"), "research": research,
        }

    def _ensemble(self, prompt: str, parse, meter: Meter) -> tuple[dict[str, list], list[str], int, int]:
        """Every (model, run) in parallel → parsed values grouped by model, transcripts, parsed/total counts."""
        jobs = [model for model in self.config["models"] for _ in range(self.config["runs_per_model"])]
        by_model: dict[str, list] = {model: [] for model in self.config["models"]}
        texts: list[str] = []

        def run(model: str):
            completion = self._complete(model, prompt, meter, temperature=0.5)
            return completion, parse(completion.text)

        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = [(model, executor.submit(run, model)) for model in jobs]
            for model, future in futures:
                try:
                    completion, value = future.result()
                    by_model[model].append(value)
                    texts.append(f"[{completion.model}] {completion.text[-COMMENT_RUN_CHARS:]}")
                except Exception as error:  # one failed run must not sink the question
                    texts.append(f"[{model} failed] {str(error)[:300]}")
        parsed = sum(len(values) for values in by_model.values())
        if parsed < max(2, len(jobs) // 2):
            raise RuntimeError(f"only {parsed}/{len(jobs)} runs parsed ({failure_summary(texts)})")
        return by_model, texts, parsed, len(jobs)

    def forecast(self, post: dict) -> Forecast:
        meter = Meter()
        try:
            return self._forecast(post, meter)
        except Exception as error:
            raise ForecastError(str(error), meter.cost_usd, meter.news_calls) from error

    def _forecast(self, post: dict, meter: Meter) -> Forecast:
        question = post["question"]
        research = self.research(question, meter)
        context = self._context(question, research)
        kind = question["type"]

        if kind == "binary":
            by_model, texts, ok, total = self._ensemble(COMMON.format(**context) + BINARY_TAIL, parse_probability,
                                                        meter)
            probability = aggregate_binary(by_model, tuple(self.config["binary_clamp"]), self.config["extremize"])
            payload = {"probability_yes": probability, "probability_yes_per_category": None, "continuous_cdf": None}
            summary: object = round(probability, 4)
        elif kind == "multiple_choice":
            options = question["options"]
            tail = MC_TAIL.format(options_json=json.dumps(options))
            by_model, texts, ok, total = self._ensemble(COMMON.format(**context) + tail,
                                                        lambda text: parse_options(text, options), meter)
            distribution = aggregate_options(by_model, options, self.config["mc_floor"])
            payload = {"probability_yes": None, "probability_yes_per_category": distribution, "continuous_cdf": None}
            summary = {option: round(value, 3) for option, value in distribution.items()}
        elif kind in ("numeric", "discrete"):
            scaling = question["scaling"]
            scale = Scale(scaling["range_min"], scaling["range_max"], scaling.get("zero_point"),
                          question["open_lower_bound"], question["open_upper_bound"])
            cdf_size = scaling["inbound_outcome_count"] + 1 if kind == "discrete" else 201
            bounds = []
            if not scale.open_lower:
                bounds.append(f"The outcome cannot be lower than {scale.lower}.")
            if not scale.open_upper:
                bounds.append(f"The outcome cannot be higher than {scale.upper}.")
            tail = NUMERIC_TAIL.format(unit=question.get("unit") or "infer from question",
                                       bounds_text=" ".join(bounds), lower=scale.lower, upper=scale.upper)
            by_model, texts, ok, total = self._ensemble(
                COMMON.format(**context) + tail,
                lambda text: percentiles_to_cdf(parse_percentiles(text), scale, cdf_size), meter)
            pooled = aggregate_cdfs(by_model)
            payload = {"probability_yes": None, "probability_yes_per_category": None, "continuous_cdf": pooled}
            summary = f"median≈{scale.to_value(next(i for i, h in enumerate(pooled) if h >= 0.5) / (cdf_size - 1)):.4g}"
        else:
            raise ValueError(f"unsupported question type {kind}")

        sources = [name for name, used in (("resolution sources", self.fetcher), ("AskNews", self.news),
                                           (self.config["research_model"], True)) if used and name]
        comment = (f"Automated forecast (ensemble of {', '.join(self.config['models'])}, {ok}/{total} runs; "
                   f"research: {', '.join(sources) or 'none'}). Aggregate: {summary}\n\n"
                   f"## Research\n{research[:COMMENT_RESEARCH_CHARS]}\n\n"
                   + "\n\n".join(f"## Run {index + 1}\n{text}" for index, text in enumerate(texts)))
        return Forecast(question["id"], post["id"], kind, payload, summary, comment, meter.cost_usd,
                        meter.news_calls, texts, ok, total)
