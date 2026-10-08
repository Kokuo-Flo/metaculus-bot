"""Resolution sources: fetch the pages a question says it resolves on, so the forecast starts from today's value.

Spring 2026's top bots read the resolution sources (some even screenshot them); the official template does not.
Short-horizon questions mostly resolve to the status quo, and the status quo is on that page: the latest data
point, the current count, the official list. Stdlib only, one polite GET per URL, honest User-Agent: a site that
blocks bots is skipped, never worked around.
"""
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser

URL = re.compile(r"https?://[^\s<>()\[\]\"'`*]+")
SKIP = re.compile(r"metaculus\.com|\.pdf(\?|$)|\.(png|jpe?g|gif|zip|xlsx?)(\?|$)|"
                  r"(twitter|x|facebook|instagram|linkedin|tiktok)\.com", re.I)
FRED_SERIES = re.compile(r"fred\.stlouisfed\.org/series/([A-Za-z0-9_]+)")
MAX_SOURCES, SOURCE_CHARS, MAX_BYTES = 3, 2_500, 3_000_000
USER_AGENT = "revenue-lab-forecasting-bot/0.2 (Metaculus FutureEval participant)"
STOPWORDS = {"will", "what", "when", "which", "with", "from", "that", "this", "than", "before", "after", "between",
             "more", "less", "least", "most", "there", "their", "have", "been", "according", "resolve", "question"}


class _Text(HTMLParser):
    """Visible text, one line per block element; drops scripts, styles and page chrome."""
    SKIPPED = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "iframe"}
    BLOCKS = {"p", "div", "tr", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "article"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIPPED:
            self.depth += 1
        elif tag in self.BLOCKS:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in self.SKIPPED and self.depth:
            self.depth -= 1

    def handle_data(self, data):
        if not self.depth:
            self.parts.append(data.replace("\n", " "))  # line breaks come from block tags only

    def text(self) -> str:
        lines = (re.sub(r"[ \t\xa0|]*\|[ \t\xa0|]*", " | ", re.sub(r"[ \t\xa0]+", " ", line)).strip(" |")
                 for line in "".join(self.parts).splitlines())
        return "\n".join(line for line in lines if line)


def extract_urls(question: dict) -> list[str]:
    """URLs from the resolution criteria first (that is where the source is named), then fine print, then background."""
    urls: list[str] = []
    for field in ("resolution_criteria", "fine_print", "description"):
        for url in URL.findall(question.get(field) or ""):
            url = url.rstrip(".,;:!?")
            if url not in urls and not SKIP.search(url):
                urls.append(url)
    return urls[:MAX_SOURCES]


def fetch(url: str, timeout: int = 15) -> str:
    """Page text; a FRED series page becomes its CSV, cut to the last 30 observations."""
    series = FRED_SERIES.search(url)
    target = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series.group(1)}" if series else url
    request = urllib.request.Request(target, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        kind = response.headers.get("Content-Type", "")
        body = response.read(MAX_BYTES).decode(response.headers.get_content_charset() or "utf-8", errors="replace")
    if series:
        lines = body.strip().splitlines()
        return "\n".join(lines[:1] + lines[-30:])
    if "html" in kind or body.lstrip()[:1] == "<":
        parser = _Text()
        parser.feed(body)
        return parser.text()
    return body


def excerpt(text: str, title: str, limit: int = SOURCE_CHARS) -> str:
    """Head + lines about the question + tail: long data tables keep their latest rows at either end."""
    if len(text) <= limit:
        return text
    lines = text.splitlines()
    keywords = {word for word in re.findall(r"[a-z0-9]{4,}", title.lower()) if word not in STOPWORDS}
    keep: set[int] = set()

    def take(indexes, budget: int) -> None:
        for index in indexes:
            if budget <= 0:
                return
            if index not in keep:
                keep.add(index)
                budget -= len(lines[index]) + 1

    numeric = sum(bool(re.search(r"\d[\d,.]*\s*(\||$)", line)) for line in lines) > 0.3 * len(lines)
    tail = limit * 35 // 100 if numeric else 0  # data tables: latest rows are often last; prose ends in chrome
    take(range(len(lines)), limit * 20 // 100)
    by_relevance = sorted(range(len(lines)), key=lambda i: -sum(word in lines[i].lower() for word in keywords))
    take([i for i in by_relevance if any(word in lines[i].lower() for word in keywords)], limit - tail - limit // 5)
    take(range(len(lines) - 1, -1, -1), tail)
    out, previous = [], -1
    for index in sorted(keep):
        if index != previous + 1:
            out.append("[…]")
        out.append(lines[index][:400])
        previous = index
    return "\n".join(out)[:limit + 200]


def resolution_sources(question: dict, fetcher=fetch, timeout: float = 25.0) -> str:
    """All sources fetched in parallel; a slow one is reported, not waited for (the question's window is short)."""
    urls = extract_urls(question)
    if not urls:
        return ""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    title = question.get("title") or ""
    executor = ThreadPoolExecutor(max_workers=len(urls))
    futures = {url: executor.submit(fetcher, url) for url in urls}
    blocks = []
    for url, future in futures.items():
        try:
            blocks.append(f"### {url}\n{excerpt(future.result(timeout=timeout), title)}")
        except Exception as error:  # a dead, slow or bot-blocking source must not sink the question
            blocks.append(f"### {url}\n[could not fetch: {type(error).__name__} {str(error)[:100]}]")
    executor.shutdown(wait=False)
    return f"## Resolution sources (fetched {today})\n" + "\n\n".join(blocks)
