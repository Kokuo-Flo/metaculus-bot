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


CHROME_CLASS = re.compile(r"\b(nav|navbar|menu|footer|sidebar|breadcrumbs?|cookie|consent|share|sharing|social|"
                          r"subscribe|newsletter|related|recommended|promo|advert|ads|banner|toolbar|comments?|popup|"
                          r"modal|skip-?link|login|signup|search-?form)\b", re.I)
CHROME_ROLES = {"navigation", "banner", "contentinfo", "complementary", "search", "dialog", "menu", "menubar", "toolbar"}
CONTAINERS = {"div", "section", "aside", "ul", "ol", "header", "footer", "nav", "form", "button", "select", "label",
              "figure", "details"}  # tags whose class/role can mark a chrome block (void tags cannot contain text)
CHROME_LABEL = re.compile(r"^(share|contact( us)?|menu|search|sign (in|up)|log ?(in|out)|subscribe|home|sections?|more|"
                          r"close|skip to [a-z]+|cookies?|privacy|terms|follow( us)?|facebook|twitter|x \(twitter\)|"
                          r"linkedin|reddit|email|print|open|next|previous|back|top|advertisement|loading)$", re.I)
MIN_AGGRESSIVE_SHARE = 0.4  # below this share of the plain text, the class-based skipping ate an article: drop it


class _Text(HTMLParser):
    """Visible text, one line per block element; drops scripts, styles and page chrome (by tag, and, when
    `aggressive`, by class/id/role: menus, share buttons, cookie banners, related links...)."""
    SKIPPED = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "iframe", "aside", "button",
               "select", "template"}
    BLOCKS = {"p", "div", "tr", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "article"}

    def __init__(self, aggressive: bool = True):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.aggressive = aggressive
        self.skip: list | None = None  # [tag, nesting] of the element being skipped, until its own end tag

    def _chrome(self, tag: str, attrs) -> bool:
        if tag in self.SKIPPED:
            return True
        if not self.aggressive or tag not in CONTAINERS:
            return False
        for name, value in attrs:
            if not value:
                continue
            if name == "role" and value.strip().lower() in CHROME_ROLES:
                return True
            if name in ("class", "id", "aria-label") and CHROME_CLASS.search(value):
                return True
        return False

    def handle_starttag(self, tag, attrs):
        if self.skip:
            if tag == self.skip[0]:
                self.skip[1] += 1
            return
        if self._chrome(tag, attrs):
            self.skip = [tag, 1]
        elif tag in self.BLOCKS:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if self.skip and tag == self.skip[0]:
            self.skip[1] -= 1
            if self.skip[1] == 0:
                self.skip = None
        elif not self.skip and tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data.replace("\n", " "))  # line breaks come from block tags only

    def text(self) -> str:
        """Lines, each once; button labels (Share, Contact us, Sections...) are chrome."""
        lines = (re.sub(r"[ \t\xa0|]*\|[ \t\xa0|]*", " | ", re.sub(r"[ \t\xa0]+", " ", line)).strip(" |")
                 for line in "".join(self.parts).splitlines())
        out, seen = [], set()
        for line in lines:
            if not line:
                continue
            key = line.lower()
            if key in seen or CHROME_LABEL.match(line):
                continue
            seen.add(key)
            out.append(line)
        return "\n".join(out)


def html_to_text(body: str) -> str:
    """Page text with chrome removed; falls back to plain tag skipping when the aggressive pass empties a malformed
    page, and says so when a large page holds almost no text (data loaded by JavaScript is not in the HTML)."""
    plain, aggressive = _Text(aggressive=False), _Text()
    plain.feed(body)
    aggressive.feed(body)
    text, lean = plain.text(), aggressive.text()
    if len(lean) >= MIN_AGGRESSIVE_SHARE * len(text):  # a utility class like "lg:pt-nav" must not eat an article
        text = lean
    if len(text) < 300 and len(body) > 50_000:
        text += "\n[page body is mostly script: its data is probably loaded by JavaScript and is not in this HTML]"
    return text


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
        return html_to_text(body)
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
