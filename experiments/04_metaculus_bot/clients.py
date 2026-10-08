"""HTTP clients — stdlib only: Metaculus API, OpenRouter (LLM + online research), and offline fakes."""
import base64
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

METACULUS_API = "https://www.metaculus.com/api"
OPENROUTER_API = "https://openrouter.ai/api/v1"
ASKNEWS_API = "https://api.asknews.app/v1"
ASKNEWS_TOKEN_URL = "https://auth.asknews.app/oauth2/token"
FIXTURES = Path(__file__).parent / "fixtures"


def _request(url: str, method: str = "GET", headers: dict | None = None, body: object = None,
             timeout: int = 180, retries: int = 3) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request_headers = {"Content-Type": "application/json", "User-Agent": "revenue-lab-bot/0.1", **(headers or {})}
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, data=data, method=method, headers=request_headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
                return json.loads(payload) if payload else {}
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:500]
            if error.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"{method} {url} → HTTP {error.code}: {detail}") from error
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


# --- Metaculus ---
class MetaculusClient:
    def __init__(self, token: str | None = None):
        self.token = token or os.environ.get("METACULUS_TOKEN")
        if not self.token:
            raise RuntimeError("METACULUS_TOKEN missing — create one at https://www.metaculus.com/futureeval/participate/")
        self.headers = {"Authorization": f"Token {self.token}"}

    def open_posts(self, tournament: int | str, limit: int = 100) -> list[dict]:
        return self.list_posts(tournament, "open", limit)

    def list_posts(self, tournament: int | str, status: str, limit: int = 100) -> list[dict]:
        """status: open | closed (awaiting resolution) | resolved."""
        posts, offset = [], 0
        while True:
            query = urllib.parse.urlencode({
                "limit": limit, "offset": offset, "order_by": "-hotness", "statuses": status,
                "forecast_type": "binary,multiple_choice,numeric,discrete", "tournaments": tournament,
                "include_description": "true"})
            page = _request(f"{METACULUS_API}/posts/?{query}", headers=self.headers)
            posts.extend(page.get("results", []))
            if not page.get("next"):
                return posts
            offset += limit

    def post(self, post_id: int) -> dict:
        return _request(f"{METACULUS_API}/posts/{post_id}/", headers=self.headers)

    def resolved_posts(self, tournament: int | str, limit: int = 100) -> list[dict]:
        query = urllib.parse.urlencode({"limit": limit, "statuses": "resolved", "tournaments": tournament,
                                        "forecast_type": "binary,multiple_choice,numeric,discrete"})
        return _request(f"{METACULUS_API}/posts/?{query}", headers=self.headers).get("results", [])

    def forecast(self, question_id: int, payload: dict) -> None:
        _request(f"{METACULUS_API}/questions/forecast/", "POST", self.headers,
                 [{"question": question_id, "source": "api", **payload}])

    def comment(self, post_id: int, text: str) -> None:
        _request(f"{METACULUS_API}/comments/create/", "POST", self.headers,
                 {"text": text, "parent": None, "included_forecast": True, "is_private": True, "on_post": post_id})


class FixtureMetaculus:
    """Offline stand-in: serves fixtures/*.json posts, records submissions instead of posting."""

    def __init__(self):
        self.submitted: list[tuple[int, dict]] = []
        self.posts = {post["id"]: post for post in
                      (json.loads(path.read_text()) for path in sorted(FIXTURES.glob("post_*.json")))}

    def open_posts(self, tournament: int | str, limit: int = 100) -> list[dict]:
        return list(self.posts.values())

    def list_posts(self, tournament: int | str, status: str, limit: int = 100) -> list[dict]:
        resolved = [post for post in self.posts.values() if post["question"].get("resolution") is not None]
        return {"open": list(self.posts.values()), "resolved": resolved}.get(status, [])

    def post(self, post_id: int) -> dict:
        return self.posts[post_id]

    def resolved_posts(self, tournament: int | str, limit: int = 100) -> list[dict]:
        return [post for post in self.posts.values() if post["question"].get("resolution") is not None]

    def forecast(self, question_id: int, payload: dict) -> None:
        self.submitted.append((question_id, payload))

    def comment(self, post_id: int, text: str) -> None:
        pass


# --- LLM ---
@dataclass
class KeyStatus:
    limit_usd: float | None  # None = the key can draw on the whole account balance
    remaining_usd: float | None
    usage_usd: float
    is_free_tier: bool


@dataclass
class Completion:
    text: str
    cost_usd: float
    model: str


class Meter:
    """Per-question tally of what a forecast consumed, shared by the threads working on that question."""

    def __init__(self):
        self.cost_usd, self.news_calls = 0.0, 0
        self._lock = threading.Lock()

    def add(self, cost_usd: float = 0.0, news_calls: int = 0) -> None:
        with self._lock:
            self.cost_usd += cost_usd
            self.news_calls += news_calls


def model_catalog() -> dict[str, dict]:
    """Public OpenRouter model list (no key needed): pricing in USD per token / per request, and the request
    parameters each model accepts — GPT-5.x and Claude 5.x reject `temperature`, for instance."""
    return {model["id"]: {"pricing": model.get("pricing") or {},
                          "supported_parameters": set(model.get("supported_parameters") or [])}
            for model in _request(f"{OPENROUTER_API}/models")["data"]}


def strip_thinking(text: str) -> str:
    """Reasoning models (e.g. sonar-reasoning) may inline their chain of thought in <think> tags."""
    return re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.DOTALL).strip()


class OpenRouter:
    def __init__(self, api_key: str | None = None, catalog: dict[str, dict] | None = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY missing (Metaculus grants free credits to bot makers)")
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.catalog = catalog or {}
        self.total_cost_usd = 0.0
        self._lock = threading.Lock()  # the ensemble calls complete() from several threads

    def key_status(self) -> KeyStatus:
        data = _request(f"{OPENROUTER_API}/key", headers=self.headers)["data"]
        return KeyStatus(data.get("limit"), data.get("limit_remaining"), float(data.get("usage") or 0),
                         bool(data.get("is_free_tier")))

    def _supported(self, model: str, parameter: str) -> bool:
        info = self.catalog.get(model)
        return parameter in info["supported_parameters"] if info else True  # unknown model: let the API decide

    def complete(self, model: str, prompt: str, *, max_tokens: int = 8000, temperature: float | None = None,
                 reasoning: str | None = None, web: str | None = None, meter: Meter | None = None) -> Completion:
        """`web="native"`: the provider's own search (covered by Metaculus credits; the Exa engine is not)."""
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "usage": {"include": True}}
        if self._supported(model, "max_tokens"):
            body["max_tokens"] = max_tokens
        if temperature is not None and self._supported(model, "temperature"):
            body["temperature"] = temperature
        if reasoning and self._supported(model, "reasoning"):
            body["reasoning"] = {"effort": reasoning}
        if web:
            body["plugins"] = [{"id": "web", "engine": web}]
        response = _request(f"{OPENROUTER_API}/chat/completions", "POST", self.headers, body)
        usage = response.get("usage") or {}
        cost = float(usage.get("cost") or 0.0)  # OpenRouter-reported, i.e. measured
        with self._lock:
            self.total_cost_usd += cost
        if meter:
            meter.add(cost_usd=cost)
        choice = response["choices"][0]
        text = strip_thinking(choice["message"].get("content") or "")
        if not text:  # reasoning can eat the whole max_tokens budget: count the cost, fail the run
            raise RuntimeError(f"{model}: empty answer (finish_reason={choice.get('finish_reason')})")
        return Completion(text, cost, model)


# --- News ---
class AskNews:
    """Free for FutureEval participants (1k calls/month; one 'latest news' search = 1 call).

    Auth: an API key (`ASKNEWS_API_KEY`, sent as Bearer) or OAuth2 client credentials (`ASKNEWS_CLIENT_ID` +
    `ASKNEWS_SECRET`). `calls_left` is this month's remaining quota, computed by the caller from the ledger.
    """

    def __init__(self, calls_left: int, api_key: str | None = None, client_id: str | None = None,
                 client_secret: str | None = None):
        self.calls_left, self.calls = calls_left, 0
        self.api_key, self.client_id, self.client_secret = api_key, client_id, client_secret
        self._token: tuple[str, float] | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, calls_left: int) -> "AskNews | None":
        api_key = os.environ.get("ASKNEWS_API_KEY")
        client_id, secret = os.environ.get("ASKNEWS_CLIENT_ID"), os.environ.get("ASKNEWS_SECRET")
        if not (api_key or (client_id and secret)):
            return None
        return cls(calls_left, api_key, client_id, secret)

    def _bearer(self) -> str:
        if self.api_key:
            return self.api_key
        if not self._token or self._token[1] < time.time() + 60:
            basic = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
            request = urllib.request.Request(
                ASKNEWS_TOKEN_URL, method="POST",
                data=urllib.parse.urlencode({"grant_type": "client_credentials", "scope": "news"}).encode(),
                headers={"Content-Type": "application/x-www-form-urlencoded", "Authorization": f"Basic {basic}"})
            with urllib.request.urlopen(request, timeout=30) as response:
                token = json.loads(response.read())
            self._token = (token["access_token"], time.time() + float(token.get("expires_in", 3600)))
        return self._token[0]

    def latest(self, query: str, n_articles: int = 8, meter: Meter | None = None) -> str:
        """Recent articles as one LLM-ready string; '' once the monthly quota is used up."""
        with self._lock:
            if self.calls_left <= 0:
                return ""
            self.calls_left, self.calls = self.calls_left - 1, self.calls + 1
        if meter:
            meter.add(news_calls=1)
        params = urllib.parse.urlencode({"query": query[:1000], "n_articles": n_articles, "return_type": "string",
                                         "strategy": "latest news"})
        response = _request(f"{ASKNEWS_API}/news/search?{params}",
                            headers={"Authorization": f"Bearer {self._bearer()}"}, timeout=60)
        return response.get("as_string") or ""


class FakeNews:
    def __init__(self):
        self.calls_left, self.calls = 10**6, 0

    def latest(self, query: str, n_articles: int = 8, meter: Meter | None = None) -> str:
        self.calls += 1
        if meter:
            meter.add(news_calls=1)
        return "[offline] no news fetched."


class FakeLLM:
    """Deterministic offline LLM producing well-formed answers, to exercise parsing/CDF/submission paths."""

    def __init__(self, seed: int = 7):
        self.random = random.Random(seed)
        self.total_cost_usd = 0.0

    def complete(self, model: str, prompt: str, *, max_tokens: int = 8000, temperature: float | None = None,
                 reasoning: str | None = None, web: str | None = None, meter: Meter | None = None) -> Completion:
        if "RESEARCH BRIEF" in prompt:
            text = "No live research in offline mode."
        elif "Percentile 10:" in prompt:
            low, high = (float(x) for x in prompt.split("BOUNDS_JSON:")[1].split()[0:2])
            centre = low + (high - low) * self.random.uniform(0.35, 0.65)
            width = (high - low) * self.random.uniform(0.05, 0.2)
            values = [centre + width * z for z in (-1.28, -0.84, -0.25, 0.25, 0.84, 1.28)]
            text = "Reasoning...\n" + "\n".join(
                f"Percentile {p}: {v:.4f}" for p, v in zip((10, 20, 40, 60, 80, 90), values))
        elif "Option_A" in prompt:
            options = json.loads(prompt.split("OPTIONS_JSON:")[1].split("\n")[0])
            weights = [self.random.uniform(1, 5) for _ in options]
            text = "Reasoning...\n" + "\n".join(
                f"{option}: {100 * weight / sum(weights):.1f}%" for option, weight in zip(options, weights))
        else:
            text = f"Reasoning...\nProbability: {self.random.randint(5, 95)}%"
        return Completion(text, 0.0, f"fake/{model}")
