# Metaculus FutureEval bot

Stdlib-only Python 3.10+ forecasting bot for the Metaculus bot tournaments (Fall 2026 FutureEval, MiniBench).
Pipeline per question: resolution sources + news + search-grounded brief → ensemble across two model families →
median per family, log-odds mean across families → Metaculus payload + one short private note per post.

    cd experiments/04_metaculus_bot
    python3 -m unittest discover -s tests -p 'test_*.py'   # 56 offline tests
    python3 bot.py --mode fixtures                        # offline self-test (fake models, bundled posts)
    python3 bot.py --mode test --dry-run --profile free   # real bot-testing-area questions, nothing submitted
    python3 bot.py --mode tournament                      # what the GitHub workflow runs

Secrets (`.env` locally, repository secrets on GitHub): `METACULUS_TOKEN`, `OPENROUTER_API_KEY`, optional
`ASKNEWS_API_KEY` or `ASKNEWS_CLIENT_ID` + `ASKNEWS_SECRET`. Lines are bare `KEY=value`.

## Modules

| File | Role |
|---|---|
| `bot.py` | One run: discover open questions (soonest close first, groups unpacked) → forecast several at once → submit → private notes → ledger. Degrades on every failure, never dies on one. |
| `forecaster.py` | Research (best effort, three parts in parallel), prompts, ensemble, aggregation, reformat fallback for answers that miss the output format. |
| `budget.py` | Paces the OpenRouter key's remaining credit until season end; picks a cheaper profile per question before any hard stop. Refuses uncapped keys unless `allow_uncapped_key`. |
| `clients.py` | Metaculus, OpenRouter (sends only the parameters a model declares; 429 backoff with `Retry-After`), AskNews, offline fakes. |
| `sources.py` | Fetches the pages cited in the resolution criteria; strips page chrome; FRED series become CSV. |
| `cdf.py` | Percentiles → Metaculus CDF (port of the official template's conversion). |
| `score.py` | `--coverage`: questions closed without our forecast in the last 7 days; local Brier/log scores of resolved forecasts. |
| `survey.py` | Measures a tournament before tuning: windows, openings per day, simultaneous questions, resolution sources. |
| `replay.py` | Re-aggregates stored model runs of resolved questions under other rules and scores them; changes nothing by itself. |

## Ledger (SQLite, `data/ledger.sqlite` locally, cached between GitHub runs)

- `forecast`: one row per submitted question (payload, summary, cost, profile, note, commented).
- `run`: one row per model run (model, parsed value, cost, tokens, duration, reformatted, error) — for `replay.py`.
- `operation`: discover / forecast / comment / skip / budget / cycle events with details (research availability,
  parse rate, failure reasons such as `3× HTTP 429`).

## Configuration (`bot_config.json`)

Profiles `full` → `standard` → `cheap` → `free` (`profile_chain`), each with research model, forecasting models,
runs per model; `reformat_model` (cheap model that extracts a final answer from a malformed one; `null` disables);
`tournaments.<id>.pace_share` (share of the daily allowance a question of that tournament may use) and
`reforecast_days` (periodic refresh, only for lifetime-scored tournaments; unused by FutureEval/MiniBench).

## GitHub Actions

`metaculus-bot.yaml`: every 20 minutes once the repository variable `BOT_LIVE` is `true`; manual runs take
bot.py arguments; optional variable `BOT_PROFILE` forces one profile. `tests.yaml`: unit tests and the offline
self-test on every push.

## Measured on 2026-10-09 (survey.py)

- Questions stay open 3 h; Fall FutureEval ≈ 1.5 questions/day, MiniBench ≈ 20/day in bursts, up to 10 open at once.
- Question text (description, criteria, fine print) is blank once a question has closed: no text backtests.
- About 260 forecasters per question; spot peer score at close; resolution 10 days (MiniBench) to 3 months later.
- Free Gemma models on OpenRouter answer HTTP 429 on most runs; the `free` profile is a last resort only.
