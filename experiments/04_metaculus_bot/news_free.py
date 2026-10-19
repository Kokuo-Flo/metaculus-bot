"""Free news and background with no API key: Google News RSS (an official public feed) and the Wikipedia API.

Replaces the paid search-grounded brief when the bot runs on free models only. Headlines are titles with dates
(no article bodies: paywalls and anti-bot pages are not fought), which is enough to know what happened and when.
One polite GET per source per question, honest User-Agent.
"""
import html
import json
import re
import urllib.parse
import urllib.request
from xml.etree import ElementTree

USER_AGENT = "revenue-lab-forecasting-bot/0.2 (Metaculus FutureEval participant)"
GOOGLE_NEWS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
STOPWORDS = {"will", "what", "when", "which", "who", "whom", "how", "many", "much", "with", "from", "that", "this",
             "than", "before", "after", "between", "more", "less", "least", "most", "there", "their", "have", "been",
             "according", "resolve", "resolves", "question", "the", "and", "for", "any", "end", "least", "above",
             "below", "least", "its", "his", "her", "per", "over", "under", "into", "onto", "does", "did", "has"}


def _get(url: str, timeout: int = 15) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(2_000_000)


def query_from_title(title: str, limit: int = 12) -> str:
    """Search words from a question title: drop question words and punctuation, keep names, numbers, dates."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'&.\-]*", title)
    kept = [word.strip(".") for word in words if word.lower() not in STOPWORDS and len(word.strip(".")) > 1]
    return " ".join(kept[:limit]) or title


def headlines(title: str, days: int = 14, limit: int = 12, fetch=_get) -> str:
    """Dated headlines from Google News for the question's keywords, newest first as the feed orders them."""
    query = urllib.parse.quote_plus(f"{query_from_title(title)} when:{days}d")
    root = ElementTree.fromstring(fetch(GOOGLE_NEWS.format(query=query)))
    lines = []
    for item in root.iter("item"):
        headline = html.unescape(item.findtext("title") or "").strip()
        date = (item.findtext("pubDate") or "").strip()[:16]  # "Mon, 19 Oct 2026"
        if headline:
            lines.append(f"- {date}: {headline}")
        if len(lines) >= limit:
            break
    if not lines:
        return ""
    return f"## Recent headlines (Google News, last {days} days; titles only)\n" + "\n".join(lines)


def wikipedia_background(title: str, limit_chars: int = 1500, fetch=_get) -> str:
    """Intro of the best-matching Wikipedia article: names, dates, the shape of the thing being asked about."""
    search = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": query_from_title(title),
                                     "srlimit": 1, "format": "json"})
    hits = json.loads(fetch(f"{WIKIPEDIA_API}?{search}")).get("query", {}).get("search") or []
    if not hits:
        return ""
    page = hits[0]["title"]
    extract_query = urllib.parse.urlencode({"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                                            "titles": page, "format": "json", "redirects": 1})
    pages = json.loads(fetch(f"{WIKIPEDIA_API}?{extract_query}")).get("query", {}).get("pages") or {}
    extract = (next(iter(pages.values()), {}) or {}).get("extract") or ""
    if not extract.strip():
        return ""
    return f"## Background (Wikipedia: {page})\n{extract.strip()[:limit_chars]}"


def free_research(title: str, fetch=_get) -> str:
    """Headlines + background, each best effort; an empty string when nothing came back."""
    parts = []
    for source in (headlines, wikipedia_background):
        try:
            text = source(title, fetch=fetch)
        except Exception as error:  # a dead feed must not sink the question
            text = f"## {source.__name__}\n[unavailable: {type(error).__name__}]"
        if text:
            parts.append(text)
    return "\n\n".join(parts)
