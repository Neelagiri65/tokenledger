"""
Activity-type classification — split token spend by what the work actually was
(coding / admin / PR / outreach / RAG / general). This is the dimension that makes spend
actionable: "we pay $X/mo on coding vs $Y on outreach" → decide what to migrate.

Order of preference (cheapest + most trustworthy first), all LOCAL (no egress):
  1. Explicit signal the customer already has — a tag/metadata field, the virtual-key/app, the
     route. Best: deterministic and free. Use it if present.
  2. Heuristic on the request text (keywords / tool names). Cheap, transparent, no model call.
  3. (Optional, not default) a local classifier model for ambiguous calls — local only.

Taxonomy is configurable; defaults below cover the activities the user named plus common ones.
"""

from __future__ import annotations

import re
from typing import Optional

from .core import CallRecord

# activity -> signals (lowercased substrings / regex) checked against request text + route + tools
TAXONOMY: dict[str, list[str]] = {
    "coding": ["```", "def ", "function ", "import ", "stack trace", "traceback",
               "compile", "refactor", "unit test", "pull request diff", "code review",
               "git ", "bug", "exception", "npm ", "pip ", "kubectl", "sql ", "regex"],
    "pr": ["press release", "pr draft", "media pitch", "embargo", "newsroom", "journalist",
           "comms ", "announcement", "blog post", "spokesperson"],
    "outreach": ["cold email", "outreach", "follow up", "follow-up", "prospect", "linkedin message",
                 "sales email", "intro email", "reach out", "sequence", "icebreaker"],
    "admin": ["schedule", "calendar", "meeting notes", "summarise this email", "summarize this email",
              "expense", "invoice", "policy", "onboarding doc", "minutes", "agenda", "to-do", "todo"],
    "rag": ["based on the following context", "use the context", "retrieved document",
            "answer using the documents", "<context>", "knowledge base", "cite the source"],
    "summarisation": ["summarise", "summarize", "tl;dr", "key points", "executive summary",
                      "condense", "extract the main"],
}

# explicit metadata/tag keys an enterprise gateway may already carry
_TAG_KEYS = ("task_class", "activity", "use_case", "feature", "category", "tag")


def classify_from_tags(tags: Optional[dict | list]) -> Optional[str]:
    """Use an explicit tag/metadata signal if present (deterministic, preferred)."""
    if not tags:
        return None
    if isinstance(tags, dict):
        for k in _TAG_KEYS:
            v = tags.get(k)
            if isinstance(v, str) and v.strip():
                return _normalise(v)
        # litellm-style tags list under metadata["tags"]
        lst = tags.get("tags")
        if isinstance(lst, list):
            return classify_from_tags(lst)
        return None
    if isinstance(tags, list):
        for v in tags:
            if isinstance(v, str):
                n = _match_known(v)
                if n:
                    return n
    return None


def classify_text(text: str, route: str = "") -> str:
    """Heuristic classification from the request text + route. Returns a taxonomy key or
    'general'. Transparent and local — no model call."""
    hay = f"{route}\n{text}".lower()
    scores = {k: sum(1 for sig in sigs if sig in hay) for k, sigs in TAXONOMY.items()}
    best = max(scores, key=lambda k: scores[k]) if scores else "general"
    return best if scores.get(best, 0) > 0 else "general"


def classify_call(rec: CallRecord, tags: Optional[dict | list] = None) -> str:
    """Resolve a call's activity type: explicit tag first, else heuristic on text."""
    if rec.task_class and rec.task_class != "unclassified":
        return rec.task_class
    tagged = classify_from_tags(tags)
    if tagged:
        return tagged
    if rec.request_text:
        return classify_text(rec.request_text, rec.route)
    return "unclassified"


def _normalise(v: str) -> str:
    v = v.strip().lower()
    aliases = {"code": "coding", "engineering": "coding", "dev": "coding",
               "sales": "outreach", "bd": "outreach", "marketing": "pr",
               "comms": "pr", "support": "admin", "ops": "admin", "summary": "summarisation"}
    return aliases.get(v, v)


def _match_known(v: str) -> Optional[str]:
    n = _normalise(v)
    return n if n in TAXONOMY or n in ("general",) else None
