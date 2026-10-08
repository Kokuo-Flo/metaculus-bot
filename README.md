# Metaculus FutureEval bot

Stdlib-only Python 3.10+ forecasting bot. Research brief → multi-model ensemble → Metaculus payload + reasoning comment.

    cd experiments/04_metaculus_bot
    python3 bot.py --mode fixtures                      # offline self-test
    python3 bot.py --mode test --dry-run --profile free # real questions, $0 models, nothing submitted
    python3 bot.py --mode tournament                    # what the GitHub workflow runs

Spending is limited to a capped OpenRouter key (see budget.py). Secrets: METACULUS_TOKEN, OPENROUTER_API_KEY.
