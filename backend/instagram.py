"""
Instagram Messaging API helpers (official API).

Notes:
- The official Instagram Messaging API supports 1:1 conversations between an Instagram
  professional account and a user; it does NOT support group conversations.
- This module focuses on webhook parsing + sending text replies.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any, Iterable

import httpx


def verify_x_hub_signature_256(body: bytes, signature_header: str | None, app_secret: str | None) -> bool:
    """
    Verify Meta Webhooks `X-Hub-Signature-256`.

    If `app_secret` is not provided, verification is skipped (returns True).
    """
    if not app_secret:
        return True
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(app_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature_header, expected)


def iter_instagram_text_messages(payload: dict[str, Any]) -> Iterable[tuple[str, str, str]]:
    """
    Yield (ig_user_id, sender_id, text) for incoming text messages.

    Expected webhook shape (simplified):
    {
      "object": "instagram",
      "entry": [{
        "id": "<IG_USER_ID>",
        "messaging": [{
          "sender": {"id": "<USER_ID>"},
          "recipient": {"id": "<IG_USER_ID>"},
          "message": {"text": "...", "is_echo": false}
        }]
      }]
    }
    """
    if not isinstance(payload, dict):
        return
    entries = payload.get("entry") or []
    if not isinstance(entries, list):
        return

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        ig_user_id = str(entry.get("id") or "")
        messaging = entry.get("messaging") or []
        if not isinstance(messaging, list):
            continue
        for ev in messaging:
            if not isinstance(ev, dict):
                continue
            msg = ev.get("message") or {}
            if not isinstance(msg, dict):
                continue
            if msg.get("is_echo"):
                continue
            text = msg.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            sender = ev.get("sender") or {}
            if not isinstance(sender, dict):
                continue
            sender_id = sender.get("id")
            if not isinstance(sender_id, str) or not sender_id.strip():
                continue
            yield (ig_user_id, sender_id.strip(), text.strip())


def _split_for_instagram(text: str, max_chars: int) -> list[str]:
    s = (text or "").strip()
    if not s:
        return []
    if max_chars <= 0 or len(s) <= max_chars:
        return [s]

    parts: list[str] = []
    rest = s
    while rest:
        chunk = rest[:max_chars]
        # Prefer splitting at a newline/space near the end, but fall back to hard cut.
        cut = max(chunk.rfind("\n"), chunk.rfind(" "))
        if cut >= int(max_chars * 0.7):
            chunk = rest[:cut]
        parts.append(chunk.strip())
        rest = rest[len(chunk) :].lstrip()
    return parts


async def send_instagram_text(
    *,
    ig_user_id: str,
    recipient_id: str,
    text: str,
    access_token: str | None = None,
) -> None:
    """
    Send a text message via the Instagram Messaging API.

    Env vars:
    - IG_ACCESS_TOKEN (required unless passed explicitly)
    - IG_GRAPH_BASE (default https://graph.instagram.com)
    - IG_API_VERSION (default v21.0)
    - IG_MESSAGE_MAX_CHARS (default 900)
    - IG_SEND_DELAY_MS (default 250)
    """
    token = access_token or os.environ.get("IG_ACCESS_TOKEN") or ""
    if not token:
        raise RuntimeError("Missing IG_ACCESS_TOKEN")
    if not ig_user_id:
        raise RuntimeError("Missing ig_user_id (Instagram professional account id)")
    if not recipient_id:
        raise RuntimeError("Missing recipient_id")

    base = (os.environ.get("IG_GRAPH_BASE") or "https://graph.instagram.com").rstrip("/")
    version = (os.environ.get("IG_API_VERSION") or "v21.0").strip()
    max_chars = int(os.getenv("IG_MESSAGE_MAX_CHARS", "900"))
    delay_ms = int(os.getenv("IG_SEND_DELAY_MS", "250"))

    url = f"{base}/{version}/{ig_user_id}/messages"
    headers = {"Authorization": f"Bearer {token}"}

    parts = _split_for_instagram(text, max_chars=max_chars)
    async with httpx.AsyncClient(timeout=20.0) as client:
        for i, part in enumerate(parts):
            payload = {"recipient": {"id": recipient_id}, "message": {"text": part}}
            r = await client.post(url, json=payload, headers=headers)
            if r.status_code >= 400:
                raise RuntimeError(f"Instagram send failed ({r.status_code}): {r.text}")
            if i != len(parts) - 1 and delay_ms > 0:
                import asyncio

                await asyncio.sleep(delay_ms / 1000.0)

