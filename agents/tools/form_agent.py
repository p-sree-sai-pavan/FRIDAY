"""
agents/tools/form_agent.py
===========================
FRIDAY FormAgent

The FormAgent handles end-to-end web form interaction:
    1. Navigate to the target URL via BrowserAgent
    2. Detect all forms and their fields
    3. Use an LLM call to map user data → field selectors (verified, not guessed)
    4. Fill each field with .fill() / .select_option()
    5. Present a full confirmation summary to the user BEFORE submitting
    6. Only submit after explicit user approval
    7. Return success/failure with post-submit page content

"Never fake data. Verify before submitting." — FRIDAY core principle
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .models import AgentResult, TaskIntent

logger = logging.getLogger("FRIDAY.FormAgent")

# ─────────────────────────────────────────────
#  Module-level constants
# ─────────────────────────────────────────────

_FIELD_PATTERNS = {
    "name":       (r"full.?name|your.?name|name", ["name", "full_name"]),
    "first_name": (r"first.?name|given.?name",    ["first_name", "firstname"]),
    "last_name":  (r"last.?name|surname|family",  ["last_name", "lastname", "surname"]),
    "email":      (r"e.?mail",                    ["email", "email_address"]),
    "phone":      (r"phone|mobile|tel",           ["phone", "mobile", "telephone"]),
    "company":    (r"company|organisation|org",   ["company", "organisation", "org"]),
    "message":    (r"message|comment|question|body", ["message", "comment", "body"]),
    "subject":    (r"subject|topic|title",        ["subject", "topic"]),
    "address":    (r"address|street",             ["address", "street_address"]),
    "city":       (r"city|town",                  ["city", "town"]),
    "country":    (r"country",                    ["country"]),
    "zip":        (r"zip|postal|postcode",        ["zip", "postal_code", "postcode"]),
}

_SUCCESS_PATTERNS = re.compile(
    r"\b(thank you|thanks|success|received|submitted|confirmed|"
    r"we.ll be in touch|confirmation number|order placed)\b",
    re.I,
)


# ─────────────────────────────────────────────
#  Data Classes
# ─────────────────────────────────────────────

@dataclass
class FormPayload:
    """Task description passed to FormAgent.execute()"""
    url: str
    instruction: str                          # Natural-language: "Fill this contact form"
    prefill_data: dict[str, Any] = field(default_factory=dict)  # From user profile/memory
    form_index: int = 0                       # Which form on the page (default: first)
    auto_submit: bool = False                 # Never true by default — always confirm


@dataclass
class FieldMapping:
    """A single resolved field ↔ value binding."""
    selector: str          # CSS selector to target the field
    field_type: str        # input | textarea | select
    label: str             # Human-readable label shown in confirmation
    value: str             # Value to fill


# ─────────────────────────────────────────────
#  FormAgent
# ─────────────────────────────────────────────

class FormAgent:
    """
    Intelligent form filler with mandatory user confirmation before submit.

    Args:
        browser          : BrowserAgent instance (shared with dispatcher).
        confirm_callback : async(prompt: str) -> bool
                           Called with a formatted summary — user must return True to proceed.
        llm_client       : Optional Groq/Gemini client. If None, rule-based mapping is used.
    """

    def __init__(
        self,
        browser,
        confirm_callback: Callable,
        llm_client=None,
    ):
        self.browser  = browser
        self._confirm = confirm_callback
        self._llm     = llm_client

    # ── Public Entry Point ──────────────────────────────────────────────

    async def execute(self, payload: FormPayload) -> AgentResult:
        logger.info(f"[FormAgent] Starting form fill → {payload.url}")

        # ── Step 1: Navigate ────────────────────────────────────────────
        try:
            page, ctx = await self.browser.get_open_page(payload.url)
        except Exception as exc:
            return AgentResult(
                success=False, intent=TaskIntent.FORM_FILL,
                summary=f"Could not open page: {exc}", error=str(exc),
            )

        try:
            snapshot = await self.browser._snapshot_page(page)

            if not snapshot.forms:
                await ctx.close()
                return AgentResult(
                    success=False, intent=TaskIntent.FORM_FILL,
                    summary=f"No forms found on {payload.url}.",
                    error="NO_FORMS_FOUND",
                )

            # ── Step 2: Select target form ───────────────────────────────
            if payload.form_index >= len(snapshot.forms):
                payload.form_index = 0
            target_form = snapshot.forms[payload.form_index]
            logger.info(f"[FormAgent] Targeting form #{payload.form_index}: {target_form.get('id') or target_form.get('name') or 'unnamed'}")

            # ── Step 3: Map data to fields ───────────────────────────────
            mappings = await self._map_fields(
                form=target_form,
                instruction=payload.instruction,
                prefill_data=payload.prefill_data,
            )

            if not mappings:
                await ctx.close()
                return AgentResult(
                    success=False, intent=TaskIntent.FORM_FILL,
                    summary="Could not map any data to form fields.",
                    error="MAPPING_FAILED",
                )

            # ── Step 4: Show confirmation ────────────────────────────────
            confirmation_text = self._build_confirmation_prompt(
                url=payload.url,
                form=target_form,
                mappings=mappings,
            )

            approved = await self._confirm(confirmation_text)

            if not approved:
                await ctx.close()
                return AgentResult(
                    success=False, intent=TaskIntent.FORM_FILL,
                    summary="Form submission cancelled by user.",
                    data={"mappings": [m.__dict__ for m in mappings]},
                )

            # ── Step 5: Fill fields ──────────────────────────────────────
            fill_errors = []
            for mapping in mappings:
                try:
                    await self._fill_field(page, mapping)
                    logger.info(f"[FormAgent] Filled '{mapping.label}' = '{mapping.value[:40]}'")
                except Exception as exc:
                    fill_errors.append(f"{mapping.label}: {exc}")
                    logger.warning(f"[FormAgent] Fill error on '{mapping.label}': {exc}")

            # ── Step 6: Submit ───────────────────────────────────────────
            submit_selector = await self._find_submit_button(page, payload.form_index)
            pre_url = page.url

            if submit_selector:
                await page.click(submit_selector, timeout=10_000)
            else:
                # Fallback: submit form via JS
                await page.evaluate(
                    f"document.querySelectorAll('form')[{payload.form_index}].submit()"
                )

            await page.wait_for_load_state("networkidle", timeout=15_000)
            post_snapshot = await self.browser._snapshot_page(page)

            # ── Step 7: Detect success ───────────────────────────────────
            success_signal = self._detect_submission_success(
                pre_url=pre_url,
                post_url=post_snapshot.url,
                post_text=post_snapshot.text_content,
            )

            await ctx.close()

            summary = (
                f"✅ Form submitted successfully.\n"
                f"Post-submit page: '{post_snapshot.title}'\n"
                f"URL changed: {pre_url} → {post_snapshot.url}"
                if success_signal
                else
                f"⚠️ Form submitted but success is unconfirmed.\n"
                f"Page title: '{post_snapshot.title}'"
            )

            if fill_errors:
                summary += f"\n\nField errors ({len(fill_errors)}):\n" + "\n".join(fill_errors)

            return AgentResult(
                success=success_signal,
                intent=TaskIntent.FORM_FILL,
                summary=summary,
                data={
                    "mappings": [m.__dict__ for m in mappings],
                    "post_url": post_snapshot.url,
                    "post_title": post_snapshot.title,
                    "post_text_preview": post_snapshot.text_content[:600],
                    "fill_errors": fill_errors,
                },
            )

        except Exception as exc:
            logger.exception(f"[FormAgent] Unexpected error: {exc}")
            await ctx.close()
            return AgentResult(
                success=False, intent=TaskIntent.FORM_FILL,
                summary=f"Form agent failed: {exc}", error=str(exc),
            )

    # ── Field Mapping ───────────────────────────────────────────────────

    async def _map_fields(
        self,
        form: dict,
        instruction: str,
        prefill_data: dict,
    ) -> list[FieldMapping]:
        """
        Map available data → form fields.
        Uses LLM if available, otherwise rule-based heuristics.
        """
        if self._llm:
            return await self._llm_map_fields(form, instruction, prefill_data)
        return self._heuristic_map_fields(form, prefill_data)

    def _heuristic_map_fields(self, form: dict, data: dict) -> list[FieldMapping]:
        """
        Rule-based field mapper. Matches field labels/names to data keys
        using keyword patterns.
        """
        mappings = []
        fields   = form.get("fields", [])
        fi       = form.get("index", 0)

        for f in fields:
            if f.get("type") in ("submit", "button", "reset", "hidden", "file", "image"):
                continue

            label_text = " ".join(filter(None, [f.get("label"), f.get("placeholder"), f.get("name"), f.get("id")])).lower()
            value_to_fill = None

            # Direct data match first
            for key in data:
                if key.lower() in label_text or label_text.startswith(key.lower()):
                    value_to_fill = str(data[key])
                    break

            # Pattern match
            if not value_to_fill:
                for _, (pattern, keys) in _FIELD_PATTERNS.items():
                    if re.search(pattern, label_text, re.I):
                        for key in keys:
                            if key in data:
                                value_to_fill = str(data[key])
                                break
                    if value_to_fill:
                        break

            if not value_to_fill:
                continue  # Skip unmapped fields — never fill with fabricated data

            selector = self._build_selector(f, fi)
            if not selector:
                continue

            mappings.append(FieldMapping(
                selector=selector,
                field_type=f.get("tag", "input"),
                label=f.get("label") or f.get("placeholder") or f.get("name") or "Unknown",
                value=value_to_fill,
            ))

        return mappings

    async def _llm_map_fields(self, form: dict, instruction: str, data: dict) -> list[FieldMapping]:
        """
        Ask the LLM to produce a JSON mapping of field selectors to values.
        Only called if self._llm is set.
        """
        fields_summary = json.dumps(form.get("fields", []), indent=2)
        data_summary   = json.dumps(data, indent=2)
        fi             = form.get("index", 0)

        prompt = f"""You are helping FRIDAY fill a web form. 
User instruction: {instruction}

Available user data:
{data_summary}

Form fields detected (form index {fi}):
{fields_summary}

Return ONLY a JSON array of objects. Each object must have:
  - "selector": CSS selector (prefer #id, then [name=...], then form:nth-of-type({fi+1}) input:nth-of-type(N))
  - "field_type": "input" | "textarea" | "select"
  - "label": human readable label
  - "value": the value to fill — ONLY from the available user data above, never invent data

If a field has no matching data, omit it entirely. Never hallucinate values.
Return only the JSON array, no other text."""

        try:
            resp = await self._llm.chat.completions.create(
                model="llama-3.1-8b-instant",
                temperature=0,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            items = json.loads(raw)
            return [FieldMapping(**item) for item in items]
        except Exception as exc:
            logger.warning(f"[FormAgent] LLM mapping failed, using heuristics: {exc}")
            return self._heuristic_map_fields(form, data)

    # ── Fill + Submit ───────────────────────────────────────────────────

    async def _fill_field(self, page: "Page", mapping: FieldMapping):  # noqa: F821
        if mapping.field_type == "select":
            await self.browser.select_option(page, mapping.selector, mapping.value)
        else:
            await page.fill(mapping.selector, mapping.value, timeout=8000)

    @staticmethod
    async def _find_submit_button(page: "Page", form_index: int) -> Optional[str]:  # noqa
        """Locate the submit button for a form."""
        selectors = [
            f"form:nth-of-type({form_index + 1}) [type=submit]",
            f"form:nth-of-type({form_index + 1}) button",
            "[type=submit]",
            "button[type=submit]",
            "input[type=submit]",
        ]
        for sel in selectors:
            try:
                count = await page.locator(sel).count()
                if count > 0:
                    return sel
            except Exception:
                continue
        return None

    # ── Confirmation ────────────────────────────────────────────────────

    @staticmethod
    def _build_confirmation_prompt(url: str, form: dict, mappings: list[FieldMapping]) -> str:
        lines = [
            "═" * 52,
            "  FRIDAY — FORM SUBMISSION CONFIRMATION",
            "═" * 52,
            f"  URL   : {url}",
            f"  Form  : {form.get('id') or form.get('name') or ('Form #' + str(form.get('index', 0)))}",
            f"  Action: {form.get('action') or 'same page'}",
            "",
            "  Fields to fill:",
        ]
        for m in mappings:
            display_val = m.value if len(m.value) < 50 else m.value[:47] + "..."
            lines.append(f"    • {m.label:<25} → {display_val}")
        lines += [
            "",
            "  Proceed with submission? [yes/no]",
            "═" * 52,
        ]
        return "\n".join(lines)

    # ── Outcome Detection ───────────────────────────────────────────────

    @staticmethod
    def _detect_submission_success(pre_url: str, post_url: str, post_text: str) -> bool:
        """
        Heuristic check: did the form submit succeed?
        Not guaranteed — always shown to user alongside raw page content.
        """
        if pre_url != post_url:
            return True

        return bool(_SUCCESS_PATTERNS.search(post_text))

    # ── Selector Builder ────────────────────────────────────────────────

    @staticmethod
    def _build_selector(field: dict, form_index: int) -> Optional[str]:
        """Build the best available CSS selector for a field."""
        if field.get("id"):
            return f"#{field['id']}"
        if field.get("name"):
            return f"[name='{field['name']}']"
        return None  # Cannot safely target without id or name