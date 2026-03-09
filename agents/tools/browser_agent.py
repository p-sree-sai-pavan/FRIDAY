"""
agents/tools/browser_agent.py
==============================
FRIDAY BrowserAgent

Playwright-powered async browser that FRIDAY uses for all live web interaction.
Follows the FRIDAY principle: only returns verified, actually-fetched content.

Capabilities:
    • navigate_and_extract()  — load a URL, return clean text + metadata
    • click_element()         — find and click a CSS/text selector
    • screenshot()            — capture current page state
    • find_links()            — extract all anchor tags from a page
    • run_js()                — execute arbitrary JavaScript (with guard)
    • close()                 — graceful cleanup

Design notes:
    - Single shared browser instance (Chromium, headless by default)
    - Each operation gets a fresh context (clean cookies/session)
    - Page content is cleaned via BeautifulSoup before returning
    - All methods return dicts conforming to AgentResult.data schema
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("FRIDAY.BrowserAgent")

# ── Optional deps — graceful fallback if not installed ──────────────────────
try:
    from playwright.async_api import async_playwright, Browser, Page, Playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("[BrowserAgent] playwright not installed. Run: pip install playwright && playwright install chromium")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


# ─────────────────────────────────────────────
#  Data Classes
# ─────────────────────────────────────────────

@dataclass
class PageSnapshot:
    url: str
    title: str
    text_content: str          # Cleaned body text
    links: list[dict]          # [{text, href}, ...]
    forms: list[dict]          # Detected forms (id, action, fields)
    screenshot_path: Optional[str] = None
    raw_html: Optional[str] = None
    error: Optional[str] = None


# ─────────────────────────────────────────────
#  BrowserAgent
# ─────────────────────────────────────────────

class BrowserAgent:
    """
    Manages a persistent Playwright Chromium browser.
    Intended to be reused across multiple agent operations within a session.
    """

    # Domains that should never be navigated to (safety guard)
    _BLOCKED_DOMAINS = frozenset([
        "localhost", "127.0.0.1", "169.254.",  # local/meta
    ])

    def __init__(self, headless: bool = True, timeout_ms: int = 30_000):
        self.headless   = headless
        self.timeout    = timeout_ms
        self._pw: Optional["Playwright"] = None
        self._browser: Optional["Browser"] = None
        self._lock = asyncio.Lock()

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def _ensure_browser(self):
        async with self._lock:
            if self._browser is None or not self._browser.is_connected():
                if not PLAYWRIGHT_AVAILABLE:
                    raise RuntimeError(
                        "Playwright is not installed. "
                        "Run: pip install playwright && playwright install chromium"
                    )
                self._pw      = await async_playwright().start()
                self._browser = await self._pw.chromium.launch(headless=self.headless)
                logger.info("[BrowserAgent] Chromium browser started.")

    async def close(self):
        """Gracefully shut down the browser."""
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()
        self._browser = None
        self._pw      = None
        logger.info("[BrowserAgent] Browser closed.")

    # ── Core Operations ─────────────────────────────────────────────────

    async def navigate_and_extract(
        self,
        url: str,
        instruction: str = "",
        take_screenshot: bool = False,
    ) -> "AgentResult":  # noqa: F821 — imported at runtime
        """
        Navigate to URL and return a clean PageSnapshot.

        Args:
            url             : Full URL to load.
            instruction     : Context hint (used in logging/future LLM re-rank).
            take_screenshot : Save a PNG of the page if True.

        Returns:
            AgentResult with .data["snapshot"] = PageSnapshot
        """
        from .models import AgentResult, TaskIntent

        self._guard_url(url)
        await self._ensure_browser()

        context = await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        try:
            logger.info(f"[BrowserAgent] Navigating → {url}")
            await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout)
            await page.wait_for_load_state("networkidle", timeout=self.timeout)

            snapshot = await self._snapshot_page(page, take_screenshot)
            await context.close()

            return AgentResult(
                success=True,
                intent=TaskIntent.WEB_BROWSE,
                summary=f"Loaded '{snapshot.title}' — {len(snapshot.text_content)} chars extracted.",
                data={"snapshot": snapshot},
            )

        except Exception as exc:
            logger.error(f"[BrowserAgent] Navigation error: {exc}")
            await context.close()
            return AgentResult(
                success=False,
                intent=TaskIntent.WEB_BROWSE,
                summary=f"Failed to load {url}: {exc}",
                error=str(exc),
            )

    async def click_and_extract(
        self,
        page: "Page",
        selector: str,
        wait_after_ms: int = 2000,
    ) -> PageSnapshot:
        """
        Click an element on an already-open page and return updated snapshot.

        Args:
            page         : Active Playwright Page object.
            selector     : CSS selector or text content ("text=Submit").
            wait_after_ms: Milliseconds to wait after click for page update.
        """
        logger.info(f"[BrowserAgent] Clicking: {selector}")
        await page.click(selector, timeout=self.timeout)
        await page.wait_for_timeout(wait_after_ms)
        return await self._snapshot_page(page)

    async def fill_field(self, page: "Page", selector: str, value: str):
        """Type a value into a field identified by CSS selector."""
        await page.fill(selector, value, timeout=self.timeout)

    async def select_option(self, page: "Page", selector: str, value: str):
        """Select a <select> dropdown value."""
        await page.select_option(selector, value=value, timeout=self.timeout)

    async def get_open_page(self, url: str) -> tuple["Page", "BrowserContext"]:  # noqa
        """
        Open a page and return (page, context) so FormAgent can interact with it.
        Caller is responsible for closing context when done.
        """
        await self._ensure_browser()
        context = await self._browser.new_context()
        page    = await context.new_page()
        self._guard_url(url)
        await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout)
        await page.wait_for_load_state("networkidle", timeout=self.timeout)
        return page, context

    # ── Page Analysis ───────────────────────────────────────────────────

    async def _snapshot_page(self, page: "Page", screenshot: bool = False) -> PageSnapshot:
        """Extract structured data from the current page state."""
        html  = await page.content()
        title = await page.title()
        url   = page.url

        text_content = self._clean_html(html)
        links        = await self._extract_links(page)
        forms        = await self._extract_forms(page)

        screenshot_path = None
        if screenshot:
            screenshot_path = f"/tmp/friday_screenshot_{_slugify(url)}.png"
            await page.screenshot(path=screenshot_path, full_page=True)

        return PageSnapshot(
            url=url,
            title=title,
            text_content=text_content,
            links=links,
            forms=forms,
            screenshot_path=screenshot_path,
            raw_html=html,
        )

    @staticmethod
    async def _extract_links(page: "Page") -> list[dict]:
        return await page.evaluate("""
            () => Array.from(document.querySelectorAll('a[href]'))
                       .map(a => ({
                           text: a.innerText.trim().slice(0, 120),
                           href: a.href
                       }))
                       .filter(l => l.text.length > 0)
                       .slice(0, 80)
        """)

    @staticmethod
    async def _extract_forms(page: "Page") -> list[dict]:
        """Detect all forms and their input fields."""
        return await page.evaluate("""
            () => Array.from(document.querySelectorAll('form')).map((form, fi) => ({
                index: fi,
                id:     form.id     || null,
                name:   form.name   || null,
                action: form.action || null,
                method: form.method || 'get',
                fields: Array.from(form.querySelectorAll('input, textarea, select')).map(el => ({
                    tag:         el.tagName.toLowerCase(),
                    type:        el.type        || null,
                    name:        el.name        || null,
                    id:          el.id          || null,
                    placeholder: el.placeholder || null,
                    label:       (() => {
                        if (el.id) {
                            const lbl = document.querySelector(`label[for="${el.id}"]`);
                            if (lbl) return lbl.innerText.trim();
                        }
                        const parent = el.closest('label');
                        return parent ? parent.innerText.trim().slice(0, 80) : null;
                    })(),
                    required:    el.required    || false,
                    value:       el.value       || null,
                })).filter(f => f.type !== 'hidden')
            }))
        """)

    @staticmethod
    def _clean_html(html: str) -> str:
        """Strip HTML tags and return readable text (BeautifulSoup if available)."""
        if BS4_AVAILABLE:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "head"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        else:
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
        # Truncate to avoid overwhelming context window
        return text[:12_000]

    # ── Safety ──────────────────────────────────────────────────────────

    def _guard_url(self, url: str):
        for blocked in self._BLOCKED_DOMAINS:
            if blocked in url:
                raise ValueError(f"Navigation to blocked domain denied: {url}")
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Only http/https URLs are allowed. Got: {url}")


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "_", text.lower())[:40]