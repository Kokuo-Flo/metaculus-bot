"""Spend guard: the bot may only spend prepaid, capped LLM credits — never an open-ended balance — and the
credits must last the whole season.

FutureEval pays on the sum of spot peer scores (squared), and each question is open for only ~1.5 h: a missed
question is a lost score, so running out of credits mid-season is worse than forecasting with cheaper models.
Before every question the bot picks the first profile of `profile_chain` that passes:
- hard stops, on the worst-case cost of the question (input bound × input price + max_tokens × output price +
  per-request fees, for every call): the OpenRouter key's own credit limit (`GET /api/v1/key`, the only cap that
  holds on a stateless runner), a per-run cap and a rolling 24 h cap. A profile that fails them is skipped, not
  the question: the chain ends with a free profile (bound 0) that always passes;
- pacing: the daily allowance is (remaining credit − reserve) / days to season end, capped by `pace_usd_per_day`.
  A profile whose expected cost would overshoot today's allowance is skipped for the next cheaper one, except the
  last of the chain. Under-spent days raise tomorrow's allowance, so the credits are spent evenly and entirely.
A key without a credit limit is refused unless `allow_uncapped_key` is set (Metaculus-issued keys live on their
account) or the chain only uses free models.
"""
from datetime import date, datetime, timedelta, timezone

from clients import model_catalog

# Forecast prompt ≈ template + background (6 000 chars) + sources (9 000) + news (8 000) + brief (12 000), ~3.5 chars/token
INPUT_TOKENS_BOUND = {"research": 1_500, "forecast": 12_500}
TYPICAL_TOKENS = {"research": (1_500, 1_500), "forecast": (6_500, 1_500)}  # (input, output): prior estimate only
MIN_SAMPLES = 3  # measured questions before the measured mean replaces the prior


class BudgetExceeded(RuntimeError):
    pass


def call_usd(price: dict, input_tokens: int, output_tokens: int, web: bool = False) -> float:
    return (input_tokens * float(price.get("prompt") or 0) + output_tokens * float(price.get("completion") or 0)
            + float(price.get("request") or 0) + (float(price.get("web_search") or 0) if web else 0.0))


def _calls(config: dict) -> list[tuple[str, str]]:
    calls = [(model, "forecast") for model in config["models"] for _ in range(config["runs_per_model"])]
    if config.get("research_model"):
        calls.append((config["research_model"], "research"))
    return calls


def _check_prices(config: dict, prices: dict[str, dict]) -> None:
    missing = sorted({model for model, _ in _calls(config) if model not in prices})
    if missing:
        raise BudgetExceeded(f"no public price for {missing}: refusing to spend on an unknown rate")


def question_bound_usd(config: dict, prices: dict[str, dict]) -> float:
    _check_prices(config, prices)
    web = bool(config.get("research_web"))
    return sum(call_usd(prices[model], INPUT_TOKENS_BOUND[kind], config["max_tokens"], web and kind == "research")
               for model, kind in _calls(config))


def question_typical_usd(config: dict, prices: dict[str, dict]) -> float:
    _check_prices(config, prices)
    web = bool(config.get("research_web"))
    return sum(call_usd(prices[model], *TYPICAL_TOKENS[kind], web and kind == "research")
               for model, kind in _calls(config))


class Budget:
    def __init__(self, limits: dict, llm, ledger, strategy_id: str, usd_to_eur, prices: dict | None = None,
                 today: date | None = None):
        self.llm, self.ledger, self.strategy_id, self.usd_to_eur = llm, ledger, strategy_id, usd_to_eur
        self.max_run, self.max_day = limits["max_usd_per_run"], limits["max_usd_per_day"]
        self.pace_ceiling, self.reserve = limits["pace_usd_per_day"], limits["key_reserve_usd"]
        self.allow_uncapped = limits.get("allow_uncapped_key", False)
        self.season_end = date.fromisoformat(limits["season_end"]) if limits.get("season_end") else None
        self.today = today or datetime.now(timezone.utc).date()
        if prices is None:
            prices = {model: info["pricing"] for model, info in model_catalog().items()}
        self.prices = prices
        self.key = None
        self.spent_before_run = self._spent_last_24h_usd()

    def _spent_last_24h_usd(self) -> float:
        since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        row = self.ledger.connection.execute(
            "SELECT COALESCE(SUM(amount_eur), 0) FROM money WHERE strategy_id=? AND kind='api_cost'"
            " AND source='measured' AND ts >= ?", (self.strategy_id, since)).fetchone()
        return row[0] / self.usd_to_eur(1.0)

    def expected_usd(self, config: dict) -> float:
        """Measured mean cost per question of this profile once known, the token-based prior until then."""
        try:
            rows = self.ledger.connection.execute(
                "SELECT cost_usd FROM forecast WHERE profile=? AND submitted=1 ORDER BY id DESC LIMIT 50",
                (config["profile"],)).fetchall()
        except Exception:  # forecast table not created yet
            rows = []
        if len(rows) >= MIN_SAMPLES:
            return sum(row[0] for row in rows) / len(rows)
        return question_typical_usd(config, self.prices)

    def pace_usd_per_day(self) -> float:
        """Today's allowance: spread what is left on the key evenly until the season ends."""
        pace = self.pace_ceiling
        if self.key and self.key.remaining_usd is not None and self.season_end:
            days = max(1, (self.season_end - self.today).days + 1)
            pace = min(pace, max(0.0, self.key.remaining_usd - self.reserve) / days)
        return pace

    def _hard_stop(self, bound: float) -> str | None:
        if bound == 0:
            return None  # free models: nothing to protect
        run = self.llm.total_cost_usd
        if run + bound > self.max_run:
            return f"run cap: spent ${run:.2f} + bound ${bound:.2f} > ${self.max_run}"
        day = self.spent_before_run + run
        if day + bound > self.max_day:
            return f"24 h cap: spent ${day:.2f} + bound ${bound:.2f} > ${self.max_day}"
        if self.key and self.key.remaining_usd is not None:
            left = self.key.remaining_usd - run
            if left - bound < self.reserve:
                return f"key credit: ${left:.2f} left, bound ${bound:.2f}, reserve ${self.reserve}"
        return None

    def _load_key(self) -> None:
        self.key = self.llm.key_status()
        if self.key.limit_usd is None and not self.allow_uncapped:
            raise BudgetExceeded(
                "this OpenRouter key has no credit limit, so a bug could drain the whole balance. On a personal "
                "account, create a key with a credit limit (openrouter.ai/settings/keys). If it is the key Metaculus "
                "issued (their account, not yours), set budget.allow_uncapped_key.")

    def choose(self, chain: list[dict], pace_share: float = 1.0) -> tuple[dict, float, str]:
        """First profile of the chain allowed right now: (config, worst-case bound, why cheaper ones were skipped).

        `pace_share` < 1 makes a tournament fall back sooner, keeping the day's allowance for questions worth more
        (a seasonal question carries ~8× the prize money of a MiniBench one)."""
        bounds = [question_bound_usd(config, self.prices) for config in chain]
        if any(bounds) and self.key is None:
            self._load_key()
        pace, skipped = self.pace_usd_per_day() * pace_share, []
        for position, (config, bound) in enumerate(zip(chain, bounds)):
            stop = self._hard_stop(bound)
            if stop:
                skipped.append(f"{config['profile']}: {stop}")
                continue
            day = self.spent_before_run + self.llm.total_cost_usd
            if position < len(chain) - 1 and bound and day + self.expected_usd(config) > pace:
                skipped.append(f"{config['profile']}: over today's ${pace:.2f} allowance (spent ${day:.2f})")
                continue
            return config, bound, "; ".join(skipped)
        raise BudgetExceeded("; ".join(skipped))
