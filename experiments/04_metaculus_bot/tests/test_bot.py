"""Offline unit tests: answer parsing on real-world LLM formatting, aggregation, profiles, the spend guard,
the OpenRouter client, resolution sources and coverage.

Run from experiments/04_metaculus_bot:  python3 -m unittest discover -s tests -p 'test_*.py'
"""
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bot  # noqa: E402
import clients  # noqa: E402
from budget import Budget, BudgetExceeded, question_bound_usd, question_typical_usd  # noqa: E402
from clients import KeyStatus, Meter, OpenRouter, strip_thinking  # noqa: E402
from forecaster import (ForecastError, Forecaster, aggregate_binary, aggregate_options, parse_options,  # noqa: E402
                        parse_percentiles, parse_probability, resolve_config)
from bot import combined_note, expand, needs_forecast  # noqa: E402
from score import coverage  # noqa: E402
from sources import _Text, excerpt, extract_urls, resolution_sources  # noqa: E402

PRICES = {  # USD per token / per request, OpenRouter format
    "paid/research": {"prompt": "0.000002", "completion": "0.000008", "request": "0", "web_search": "0.005"},
    "paid/a": {"prompt": "0.000002", "completion": "0.000012", "web_search": "0.01"},
    "paid/cheap": {"prompt": "0.0000002", "completion": "0.0000012"},
    "free/a:free": {"prompt": "0", "completion": "0"},
}
LIMITS = {"max_usd_per_run": 1.0, "max_usd_per_day": 2.0, "pace_usd_per_day": 0.5, "key_reserve_usd": 0.1,
          "season_end": None}
PAID = {"profile": "paid", "research_model": "paid/research", "research_web": "native", "models": ["paid/a"],
        "runs_per_model": 2, "max_tokens": 1000}
CHEAP = {**PAID, "profile": "cheap", "research_model": None, "models": ["paid/cheap"]}
FREE = {**PAID, "profile": "free", "research_model": None, "models": ["free/a:free"]}


class Parsing(unittest.TestCase):
    def test_probability_variants(self):
        for text, expected in [("Probability: 23%", 0.23), ("**Probability:** 23%", 0.23),
                               ("Probability: **23.5 %**", 0.235), ("PROBABILITY: 7%", 0.07),
                               ("Probability: 12,5%", 0.125), ("Probability: 0.35", 0.35),
                               ("Probability: 35", 0.35), ("Probability: 1", 1.0),
                               ("Probability: 40% at first...\nProbability: 30%", 0.30)]:
            self.assertAlmostEqual(parse_probability(text), expected, msg=text)

    def test_probability_rejects(self):
        for text in ("no answer", "Probability: 230%", "Probability: 230"):
            with self.assertRaises(ValueError, msg=text):
                parse_probability(text)

    def test_percentiles_variants(self):
        text = ("**Percentile 10:** −5\nPercentile 20: -2.5\nPercentile 40: $1,200\n"
                "percentile 60: 1,500.5\nPercentile 80 : **2000**\nPercentile 90: 3000")
        self.assertEqual(parse_percentiles(text), {10: -5, 20: -2.5, 40: 1200, 60: 1500.5, 80: 2000, 90: 3000})

    def test_percentiles_word_multipliers_but_not_units(self):
        text = ("Percentile 10: 1.2 million\nPercentile 20: 1.5M\nPercentile 40: 2k\nPercentile 60: 3 thousand\n"
                "Percentile 80: 4bn\nPercentile 90: 5 metres")
        self.assertEqual(parse_percentiles(text), {10: 1.2e6, 20: 1.5, 40: 2000, 60: 3000, 80: 4e9, 90: 5})

    def test_options_do_not_confuse_prefixes(self):
        options = ["1", "10", "Other"]
        parsed = parse_options("- 10: 50%\n**1**: 30%\nOther: 20%", options)
        self.assertEqual(parsed, {"10": 0.5, "1": 0.3, "Other": 0.2})

    def test_strip_thinking(self):
        self.assertEqual(strip_thinking("<think>chain\nof thought</think>\nBrief."), "Brief.")
        self.assertEqual(strip_thinking("<think>truncated"), "")


class Aggregation(unittest.TestCase):
    def test_binary_family_then_logodds(self):
        # family A median 0.2, family B median 0.8 → symmetric in log-odds → 0.5
        self.assertAlmostEqual(aggregate_binary({"a": [0.1, 0.2, 0.3], "b": [0.8]}, (0.01, 0.99)), 0.5)
        # a family with no parsed run is ignored; a wild sample is absorbed by the median
        self.assertAlmostEqual(aggregate_binary({"a": [0.3, 0.3, 0.99], "b": []}, (0.01, 0.99)), 0.3)

    def test_binary_clamp_and_extremize(self):
        self.assertEqual(aggregate_binary({"a": [0.999]}, (0.015, 0.985)), 0.985)
        self.assertEqual(aggregate_binary({"a": [0.5], "b": [0.5]}, (0.01, 0.99), extremize=2.0), 0.5)
        self.assertGreater(aggregate_binary({"a": [0.7]}, (0.01, 0.99), extremize=2.0), 0.7)

    def test_options_floor_and_normalise(self):
        by_model = {"a": [{"x": 1.0, "y": 0.0}], "b": [{"x": 0.6, "y": 0.4}, {"x": 0.4, "y": 0.6}]}
        out = aggregate_options(by_model, ["x", "y"], 0.005)
        self.assertAlmostEqual(sum(out.values()), 1.0)
        self.assertAlmostEqual(out["x"], 0.75)  # (1.0 + 0.5) / 2


class Profiles(unittest.TestCase):
    RAW = {"profile": "full", "profile_chain": ["full", "free"],
           "profiles": {"full": {"models": ["x"], "reasoning_effort": "high"},
                        "free": {"models": ["y:free"], "research_model": None}},
           "tournaments": {"minibench": {"pace_share": 0.4}}, "budget": {"max_usd_per_run": 9}}

    def test_default_and_override(self):
        full = resolve_config(self.RAW)
        self.assertEqual((full["models"], full["reasoning_effort"]), (["x"], "high"))
        self.assertNotIn("tournaments", full)
        free = resolve_config(self.RAW, "free")
        self.assertEqual((free["models"], free["research_model"], free["profile"]), (["y:free"], None, "free"))
        self.assertIsNone(free["reasoning_effort"])
        self.assertEqual(free["budget"]["max_usd_per_run"], 9)
        self.assertEqual(free["budget"]["max_usd_per_day"], 25.0)  # default kept

    def test_unknown_profile(self):
        with self.assertRaises(ValueError):
            resolve_config(self.RAW, "nope")


class FakeLedger:
    def __init__(self, spent_eur_24h: float = 0.0):
        self.connection = sqlite3.connect(":memory:")
        self.connection.execute("CREATE TABLE money (strategy_id, ts, kind, amount_eur, source, note)")
        if spent_eur_24h:
            self.connection.execute("INSERT INTO money VALUES ('s', '9999', 'api_cost', ?, 'measured', '')",
                                    (spent_eur_24h,))


class FakeKeyLLM:
    def __init__(self, status: KeyStatus):
        self.status, self.total_cost_usd, self.calls = status, 0.0, 0

    def key_status(self) -> KeyStatus:
        self.calls += 1
        return self.status


def budget(llm, spent_eur_24h=0.0, limits=LIMITS, today=None) -> Budget:
    return Budget(limits, llm, FakeLedger(spent_eur_24h), "s", lambda usd: usd, PRICES, today)


CAPPED = KeyStatus(100.0, 100.0, 0.0, False)


class SpendGuard(unittest.TestCase):
    def test_bounds_charge_web_search_to_research_only(self):
        # 2 runs × (12500*2e-6 + 1000*12e-6) = 0.074 ; research 1500*2e-6 + 1000*8e-6 + 0.005 web = 0.016
        self.assertAlmostEqual(question_bound_usd(PAID, PRICES), 0.09)
        # typical: 2 × (6500*2e-6 + 1500*12e-6) = 0.062 ; research 0.003 + 0.012 + 0.005 = 0.02
        self.assertAlmostEqual(question_typical_usd(PAID, PRICES), 0.082)

    def test_unknown_price_refused(self):
        with self.assertRaises(BudgetExceeded):
            question_bound_usd({**PAID, "models": ["unknown/model"]}, PRICES)

    def test_uncapped_key_refused_unless_allowed(self):
        llm = FakeKeyLLM(KeyStatus(None, None, 0.0, False))
        with self.assertRaisesRegex(BudgetExceeded, "no credit limit"):
            budget(llm).choose([PAID])
        self.assertEqual(budget(llm, limits={**LIMITS, "allow_uncapped_key": True}).choose([PAID])[0], PAID)

    def test_free_profile_never_needs_a_key(self):
        llm = FakeKeyLLM(KeyStatus(None, None, 0.0, True))
        self.assertEqual(budget(llm).choose([FREE])[:2], (FREE, 0.0))
        self.assertEqual(llm.calls, 0)

    def test_hard_stops_degrade_to_free_instead_of_stopping(self):
        llm = FakeKeyLLM(KeyStatus(1.0, 0.105, 0.895, False))  # credit barely above the 0.1 reserve
        config, bound, why = budget(llm).choose([PAID, CHEAP, FREE])
        self.assertEqual((config, bound), (FREE, 0.0))
        self.assertIn("key credit", why)
        with self.assertRaisesRegex(BudgetExceeded, "key credit"):  # no free fallback in the chain → stop
            budget(llm).choose([PAID, CHEAP])

    def test_run_and_day_caps(self):
        llm = FakeKeyLLM(CAPPED)
        llm.total_cost_usd = 0.95
        self.assertEqual(budget(llm).choose([PAID, FREE])[0], FREE)
        llm.total_cost_usd = 0.0
        self.assertEqual(budget(llm, spent_eur_24h=1.95).choose([PAID, FREE])[0], FREE)

    def test_pace_falls_back_to_cheaper_profile(self):
        guard = budget(FakeKeyLLM(CAPPED), spent_eur_24h=0.45)
        self.assertEqual(guard.choose([PAID, CHEAP, FREE])[0], CHEAP)  # paid's expected cost crosses the pace
        self.assertEqual(guard.choose([PAID])[0], PAID)  # last profile of the chain: only hard stops apply

    def test_lower_pace_share_downgrades_minibench_first(self):
        guard = budget(FakeKeyLLM(CAPPED), spent_eur_24h=0.3)
        self.assertEqual(guard.choose([PAID, CHEAP])[0], PAID)  # seasonal question: full pace
        self.assertEqual(guard.choose([PAID, CHEAP], pace_share=0.4)[0], CHEAP)  # MiniBench: 0.2 $/day pace

    def test_pace_adapts_to_remaining_credit_and_season_end(self):
        limits = {**LIMITS, "pace_usd_per_day": 6.0, "season_end": "2026-12-20"}
        guard = budget(FakeKeyLLM(KeyStatus(100.0, 73.1, 26.9, False)), limits=limits, today=date(2026, 10, 8))
        guard.choose([PAID, FREE])  # loads the key
        self.assertAlmostEqual(guard.pace_usd_per_day(), 73.0 / 74)  # (73.1 − 0.1 reserve) / 74 days incl. today
        no_end = budget(FakeKeyLLM(CAPPED), limits={**limits, "season_end": None})
        self.assertEqual(no_end.pace_usd_per_day(), 6.0)


class OpenRouterClient(unittest.TestCase):
    CATALOG = {"openai/x": {"pricing": {}, "supported_parameters": {"max_tokens", "reasoning"}},
               "google/y": {"pricing": {}, "supported_parameters": {"max_tokens", "temperature"}}}
    OK = {"choices": [{"message": {"content": "Probability: 5%"}}], "usage": {"cost": 0.02}}

    def test_unsupported_parameters_are_not_sent(self):
        with mock.patch.object(clients, "_request", return_value=self.OK) as request:
            llm = OpenRouter(api_key="k", catalog=self.CATALOG)
            llm.complete("openai/x", "p", max_tokens=123, temperature=0.5, reasoning="high", web="native")
            body = request.call_args.args[3]
            self.assertEqual((body["max_tokens"], body["reasoning"], body["plugins"]),
                             (123, {"effort": "high"}, [{"id": "web", "engine": "native"}]))
            self.assertNotIn("temperature", body)
            llm.complete("google/y", "p", temperature=0.5, reasoning="high")
            body = request.call_args.args[3]
            self.assertEqual(body["temperature"], 0.5)
            self.assertNotIn("reasoning", body)

    def test_meter_and_total_track_cost_even_on_empty_answer(self):
        empty = {"choices": [{"message": {"content": None}, "finish_reason": "length"}], "usage": {"cost": 0.04}}
        meter = Meter()
        with mock.patch.object(clients, "_request", side_effect=[self.OK, empty]):
            llm = OpenRouter(api_key="k")
            llm.complete("m", "p", meter=meter)
            with self.assertRaisesRegex(RuntimeError, "finish_reason=length"):
                llm.complete("m", "p", meter=meter)
        self.assertAlmostEqual(llm.total_cost_usd, 0.06)
        self.assertAlmostEqual(meter.cost_usd, 0.06)


class ForecasterPipeline(unittest.TestCase):
    POST = {"id": 1, "question": {"id": 11, "type": "binary", "title": "Will X happen?", "description": "",
                                  "resolution_criteria": "", "fine_print": ""}}

    class LLM:
        def __init__(self, answers):
            self.answers, self.total_cost_usd = list(answers), 0.0

        def complete(self, model, prompt, *, max_tokens, temperature=None, reasoning=None, web=None, meter=None):
            answer = self.answers.pop(0)
            meter.add(cost_usd=0.01)
            if isinstance(answer, Exception):
                raise answer
            return clients.Completion(answer, 0.01, model)

    def test_research_failure_is_best_effort_and_costs_are_metered(self):
        llm = self.LLM([RuntimeError("search down"), "Probability: 30%", "Probability: 40%",
                        "Probability: 20%", "Probability: 60%"])
        config = {**PAID, "models": ["a", "b"], "runs_per_model": 2, "binary_clamp": [0.01, 0.99],
                  "extremize": 1.0, "reasoning_effort": None}
        forecast = Forecaster(llm, config, news=None, fetcher=None).forecast(self.POST)
        self.assertAlmostEqual(forecast.cost_usd, 0.05)
        self.assertEqual((forecast.runs_ok, forecast.runs_total), (4, 4))
        self.assertIn("[unavailable: search down]", forecast.comment)
        # a: median 0.35, b: median 0.40 → log-odds mean ≈ 0.375
        self.assertAlmostEqual(forecast.payload["probability_yes"], 0.3748, places=3)

    def test_too_few_parsed_runs_fails_with_cost_attached(self):
        llm = self.LLM(["brief", "garbage", "Probability: 40%", "garbage", "garbage"])
        config = {**PAID, "models": ["a", "b"], "runs_per_model": 2, "reasoning_effort": None}
        with self.assertRaises(ForecastError) as caught:
            Forecaster(llm, config, news=None, fetcher=None).forecast(self.POST)
        self.assertAlmostEqual(caught.exception.cost_usd, 0.05)
        self.assertIn("only 1/4 runs parsed", str(caught.exception))

    def test_failure_reasons_are_summarised_in_the_error(self):
        llm = self.LLM(["brief", RuntimeError("POST u → HTTP 429: upstream"), RuntimeError("POST u → HTTP 429: x"),
                        "garbage", "Probability: 40%"])
        config = {**PAID, "models": ["a", "b"], "runs_per_model": 2, "reasoning_effort": None}
        with self.assertRaises(ForecastError) as caught:
            Forecaster(llm, config, news=None, fetcher=None).forecast(self.POST)
        self.assertIn("only 1/4 runs parsed (2× HTTP 429, 1× unparsable)", str(caught.exception))


class ResolutionSources(unittest.TestCase):
    QUESTION = {"title": "Will Brent close above $100 on 30 Nov 2026?",
                "resolution_criteria": "Per [EIA](https://www.eia.gov/dnav/pet/hist/rbrteD.htm), or "
                                       "https://fred.stlouisfed.org/series/DCOILBRENTEU.",
                "fine_print": "Ignore https://twitter.com/foo and https://www.metaculus.com/questions/1/. "
                              "Data: https://example.org/brent.pdf",
                "description": "Background: https://en.wikipedia.org/wiki/Brent_Crude, "
                               "https://www.eia.gov/dnav/pet/hist/rbrteD.htm"}

    def test_urls_criteria_first_deduped_and_filtered(self):
        self.assertEqual(extract_urls(self.QUESTION), ["https://www.eia.gov/dnav/pet/hist/rbrteD.htm",
                                                       "https://fred.stlouisfed.org/series/DCOILBRENTEU",
                                                       "https://en.wikipedia.org/wiki/Brent_Crude"])

    def test_table_excerpt_keeps_latest_rows(self):
        rows = "\n".join(f"<tr><td>day {i}</td><td>{50 + i * 0.01:.2f}</td></tr>" for i in range(5000))
        parser = _Text()
        parser.feed(f"<html><script>var x=1;</script><table>{rows}</table></html>")
        text = parser.text()
        self.assertTrue(text.startswith("day 0 | 50.00"))
        self.assertNotIn("var x", text)
        self.assertIn("day 4999 | 99.99", excerpt(text, self.QUESTION["title"]))

    def test_prose_excerpt_prefers_relevant_lines_over_page_chrome(self):
        text = "\n".join(["Intro."] * 200 + ["Brent closed at 112 USD on 29 Sep."] + ["Categories: footer"] * 200)
        cut = excerpt(text, self.QUESTION["title"])
        self.assertIn("Brent closed at 112", cut)
        self.assertNotIn("Categories", cut)

    def test_failed_or_slow_source_is_reported_not_raised(self):
        def fetcher(url):
            if "eia" in url:
                raise OSError("HTTP 403")
            if "fred" in url:
                import time
                time.sleep(0.5)
            return "ok"
        out = resolution_sources(self.QUESTION, fetcher, timeout=0.1)
        self.assertIn("[could not fetch: OSError HTTP 403]", out)
        self.assertIn("[could not fetch: TimeoutError", out)
        self.assertEqual(out.count("\nok"), 1)


class Coverage(unittest.TestCase):
    def test_counts_only_recently_closed_and_missed(self):
        now = datetime.now(timezone.utc)

        def post(post_id, hours_ago, forecast):
            return {"id": post_id, "question": {
                "scheduled_close_time": (now - timedelta(hours=hours_ago)).isoformat(),
                "my_forecasts": {"latest": {"forecast_values": [0.3, 0.7]} if forecast else None}}}

        posts = {1: post(1, 2, True), 2: post(2, 30, False), 3: post(3, 24 * 10, False), 4: post(4, -1, False)}

        class Client:
            def list_posts(self, tournament, status):
                return list(posts.values()) if status == "closed" else []

            def post(self, post_id):
                return posts[post_id]

        self.assertEqual(coverage(Client(), ["t"]), {"t": {"closed": 2, "missed": 1, "missed_posts": [2]}})

class MetaculusPagination(unittest.TestCase):
    """2026-10-08, bot-testing-area: /api/posts/ returns `next` on every page, even empty ones, and no `count`."""

    def test_empty_page_with_next_ends_listing(self):
        full = {"results": [{"id": i} for i in range(100)], "next": "https://x/?offset=100"}
        empty = {"results": [], "next": "https://x/?offset=200"}
        with mock.patch.object(clients, "_request", side_effect=[full, empty, empty, empty]) as request:
            posts = clients.MetaculusClient(token="t").open_posts("bot-testing-area")
        self.assertEqual(len(posts), 100)
        self.assertEqual(request.call_count, 2)

    def test_short_page_ends_listing_without_extra_call(self):
        short = {"results": [{"id": 1}, {"id": 2}], "next": "https://x/?offset=100"}
        with mock.patch.object(clients, "_request", side_effect=[short]) as request:
            posts = clients.MetaculusClient(token="t").list_posts("bot-testing-area", "open")
        self.assertEqual([p["id"] for p in posts], [1, 2])
        self.assertEqual(request.call_count, 1)

    def test_page_cap_guards_against_endless_full_pages(self):
        full = {"results": [{"id": 0}] * 100, "next": "https://x/?offset=100"}
        with mock.patch.object(clients, "_request", return_value=full) as request:
            posts = clients.MetaculusClient(token="t").list_posts("bot-testing-area", "open", max_pages=3)
        self.assertEqual(len(posts), 300)
        self.assertEqual(request.call_count, 3)




class RequestRetries(unittest.TestCase):
    """429 handling in clients._request: backoff 5/10/20/40 s or the Retry-After header, then give up."""

    @staticmethod
    def _http_error(code, headers=None):
        import email.message
        import io
        hdrs = email.message.Message()
        for key, value in (headers or {}).items():
            hdrs[key] = value
        return clients.urllib.error.HTTPError("https://x", code, "err", hdrs, io.BytesIO(b'{"error": "x"}'))

    def test_429_backs_off_then_succeeds(self):
        ok = mock.MagicMock()
        ok.__enter__.return_value.read.return_value = b'{"a": 1}'
        with mock.patch("urllib.request.urlopen", side_effect=[self._http_error(429), self._http_error(429), ok]), \
                mock.patch("time.sleep") as sleep:
            self.assertEqual(clients._request("https://x"), {"a": 1})
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 10])

    def test_429_honours_retry_after_then_gives_up(self):
        errors = [self._http_error(429, {"Retry-After": "3"}) for _ in range(5)]
        with mock.patch("urllib.request.urlopen", side_effect=errors), mock.patch("time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
                clients._request("https://x")
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [3, 3, 3, 3])

class ProductionTournaments(unittest.TestCase):
    def test_only_verified_tournaments_run_in_production(self):
        # Market Pulse 26Q4 and Animal Futures stay parked until their rules, a real submission and group posts
        # are verified (see the comment on TOURNAMENTS); their pacing settings remain configured.
        self.assertEqual(bot.TOURNAMENTS["tournament"], [33121, "minibench"])
        config = json.loads((Path(__file__).resolve().parent.parent / "bot_config.json").read_text())
        self.assertIn("market-pulse-26q4", config["tournaments"])
        self.assertIn("33016", config["tournaments"])


class ExitCode(unittest.TestCase):
    def test_red_only_when_nothing_worked(self):
        self.assertEqual(bot.exit_code(2, 3, 1), 0)  # partial failures: ledger + coverage report, not a red run
        self.assertEqual(bot.exit_code(0, 0, 1), 0)  # nothing to do
        self.assertEqual(bot.exit_code(0, 3, 1), 1)  # every question failed
        self.assertEqual(bot.exit_code(0, 0, 0), 1)  # no tournament could be listed


class GroupsAndRefresh(unittest.TestCase):
    GROUP = {"id": 7, "title": "Yield at month end?", "scheduled_close_time": "2026-12-31T00:00:00Z",
             "group_of_questions": {"description": "D", "resolution_criteria": "R", "fine_print": "F", "questions": [
                 {"id": 71, "type": "numeric", "label": "October", "title": "Yield at month end?",
                  "scheduled_close_time": "2026-10-31T00:00:00Z", "my_forecasts": {"latest": None}},
                 {"id": 72, "type": "numeric", "label": "", "my_forecasts": {"latest": None}}]}}

    def test_group_post_unpacks_with_group_text_and_labels(self):
        subs = expand(self.GROUP)
        self.assertEqual([s["question"]["id"] for s in subs], [71, 72])
        first = subs[0]["question"]
        self.assertEqual((first["description"], first["resolution_criteria"], first["fine_print"]), ("D", "R", "F"))
        self.assertEqual(first["title"], "Yield at month end? — October")
        self.assertEqual(subs[1]["question"]["title"], "Yield at month end?")  # empty label, post title
        self.assertEqual(expand({"id": 1, "question": {"id": 2}}), [{"id": 1, "question": {"id": 2}}])
        self.assertEqual(expand({"id": 1, "notice": "no question"}), [])

    def test_needs_forecast_one_shot_vs_periodic(self):
        now = datetime(2026, 10, 9, tzinfo=timezone.utc)
        never = {"question": {"my_forecasts": {"latest": None}}}
        recent = {"question": {"my_forecasts": {"latest": {"forecast_values": [0.5], "start_time": now.timestamp() - 86400}}}}
        old = {"question": {"my_forecasts": {"latest": {"forecast_values": [0.5], "start_time": now.timestamp() - 8 * 86400}}}}
        self.assertTrue(needs_forecast(never, None, now))
        self.assertFalse(needs_forecast(recent, None, now))
        self.assertFalse(needs_forecast(old, None, now))  # one-shot tournaments never refresh
        self.assertFalse(needs_forecast(recent, 7, now))
        self.assertTrue(needs_forecast(old, 7, now))

    def test_combined_note_single_vs_group(self):
        rows = [{"id": 1, "post_id": 7, "question_id": 71, "type": "numeric", "summary": "\"median≈3.1\"",
                 "comment": "Automated forecast ... Aggregate: median≈3.1\n\n## Research\nstuff\n\n## Run 1\nlong"},
                {"id": 2, "post_id": 7, "question_id": 72, "type": "numeric", "summary": "\"median≈3.3\"",
                 "comment": "second"}]
        self.assertEqual(combined_note(rows[:1]), rows[0]["comment"])
        note = combined_note(rows)
        self.assertIn("2 sub-questions", note)
        self.assertIn("- q71 (numeric): median≈3.1", note)
        self.assertIn("- q72 (numeric): median≈3.3", note)
        self.assertIn("## Research\nstuff", note)
        self.assertNotIn("## Run 1", note)


class LoopResilience(unittest.TestCase):
    def test_refused_post_is_logged_and_the_run_goes_on(self):
        original = bot.FixtureMetaculus.post

        def flaky(self, post_id):
            if post_id == 9002:
                raise RuntimeError("GET /posts/9002/ → HTTP 404: not found")
            return original(self, post_id)

        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.sqlite"
            with mock.patch.object(bot.FixtureMetaculus, "post", flaky), \
                    mock.patch.object(sys, "argv", ["bot.py", "--mode", "fixtures", "--ledger", str(ledger_path)]), \
                    mock.patch("sys.stdout", new_callable=io.StringIO), \
                    mock.patch("sys.stderr", new_callable=io.StringIO) as err:
                code = bot.main()
            self.assertEqual(code, 0)  # the other posts went through
            self.assertIn("[fail] post 9002 fetch: GET /posts/9002/", err.getvalue())
            connection = sqlite3.connect(ledger_path)
            self.assertEqual(connection.execute("SELECT ok FROM operation WHERE kind='post'").fetchall(), [(0,)])
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM forecast WHERE submitted=1").fetchone()[0], 6)


if __name__ == "__main__":
    unittest.main()
