"""Cost models and FX. Every constant cites its source; override via config.json."""
import json
import re
import urllib.request
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / "config.json"
DEFAULTS = {
    # TRV EDF option Base 6 kVA, 1 Aug 2026 (CRE délibération 2026-147) — fournisseurs-electricite.com
    "electricity_eur_per_kwh": 0.2001,
    "usd_eur_fallback": 0.86,
    # Laptop (i5-10310U, TDP 15 W): measured package power not available → spec-based
    "laptop_idle_w": 8,
    "laptop_load_w": 35,
}
_fx_cache: dict[str, float] = {}


def config() -> dict:
    values = dict(DEFAULTS)
    if CONFIG.exists():
        values.update(json.loads(CONFIG.read_text()))
    return values


def usd_to_eur(amount_usd: float) -> float:
    """ECB daily reference rate (public XML), cached per process; falls back to config."""
    if "usd" not in _fx_cache:
        try:
            with urllib.request.urlopen("https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml",
                                        timeout=15) as response:
                match = re.search(rb"currency='USD' rate='([\d.]+)'", response.read())
            _fx_cache["usd"] = 1 / float(match.group(1)) if match else config()["usd_eur_fallback"]
        except OSError:
            _fx_cache["usd"] = config()["usd_eur_fallback"]
    return amount_usd * _fx_cache["usd"]


def electricity_eur(watts: float, hours: float) -> float:
    return watts / 1000 * hours * config()["electricity_eur_per_kwh"]


def straight_line_depreciation_eur_month(purchase_eur: float, life_months: int, resale_eur: float = 0.0) -> float:
    return (purchase_eur - resale_eur) / life_months
