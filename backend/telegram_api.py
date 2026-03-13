"""
Telegram Bot API helpers.

Supports:
- Webhook updates (single Update JSON)
- Long polling updates (getUpdates result payload)

Design goal: make it easy to run the bot locally (polling) or behind a public URL (webhook),
and keep group chat costs under control by only responding to explicit triggers.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable

import httpx


TELEGRAM_MAX_MESSAGE_CHARS = 4096


def verify_telegram_webhook_secret(header_value: str | None, expected: str | None) -> bool:
    """
    Verify Telegram webhook secret token.

    Telegram sends it as header:
      X-Telegram-Bot-Api-Secret-Token: <secret>
    """
    expected = (expected or "").strip()
    if not expected:
        return True
    return (header_value or "").strip() == expected


def _sender_display(from_obj: dict[str, Any]) -> str:
    username = from_obj.get("username")
    if isinstance(username, str) and username.strip():
        return "@" + username.strip()
    first = from_obj.get("first_name")
    last = from_obj.get("last_name")
    if isinstance(first, str) and first.strip():
        name = first.strip()
        if isinstance(last, str) and last.strip():
            name += " " + last.strip()
        return name
    return "Someone"


def _is_bot_command(text: str, entities: Any) -> bool:
    if not isinstance(text, str) or not text.startswith("/"):
        return False
    if not isinstance(entities, list) or not entities:
        return True
    e0 = entities[0]
    if not isinstance(e0, dict):
        return True
    return e0.get("type") == "bot_command" and int(e0.get("offset") or 0) == 0


def _parse_command(text: str, entities: Any) -> tuple[str | None, str | None, str]:
    """
    Return (command, target_username, rest). Command is normalized without bot username.
    Example: "/council@MyBot hello" -> ("council", "mybot", "hello")
    """
    if not _is_bot_command(text, entities):
        return (None, None, text.strip())
    first = (text or "").strip().split(maxsplit=1)[0]
    if not first.startswith("/"):
        return (None, None, text.strip())
    cmd = first[1:]
    target: str | None = None
    if "@" in cmd:
        cmd, target = cmd.split("@", 1)
    rest = (text or "").strip()[len(first) :].strip()
    target_norm = target.strip().lower() if isinstance(target, str) and target.strip() else None
    return (cmd.lower(), target_norm, rest)


def _strip_bot_mention(text: str, bot_username: str) -> str:
    u = (bot_username or "").strip().lstrip("@")
    if not u:
        return (text or "").strip()
    # Remove @BotName mentions.
    pat = re.compile(rf"@{re.escape(u)}\\b", re.IGNORECASE)
    return pat.sub("", text or "").strip()


def iter_telegram_text_messages(payload: dict[str, Any]) -> Iterable[tuple[int, int, str, str, str, dict[str, Any]]]:
    """
    Yield (chat_id, message_id, chat_type, sender_display, text, message_obj) for each incoming text message.

    `payload` can be:
    - a single Update (webhook)
    - a getUpdates response: {"ok": true, "result": [Update, ...]}
    """
    updates: list[Any]
    if isinstance(payload, dict) and isinstance(payload.get("result"), list):
        updates = payload["result"]
    else:
        updates = [payload]

    for upd in updates:
        if not isinstance(upd, dict):
            continue
        msg = upd.get("message") or upd.get("edited_message")
        if not isinstance(msg, dict):
            continue
        text = msg.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        from_obj = msg.get("from") or {}
        if not isinstance(from_obj, dict):
            continue
        if from_obj.get("is_bot"):
            continue
        chat = msg.get("chat") or {}
        if not isinstance(chat, dict):
            continue
        chat_id = chat.get("id")
        if not isinstance(chat_id, int):
            continue
        chat_type = chat.get("type")
        if not isinstance(chat_type, str):
            chat_type = ""
        message_id = msg.get("message_id")
        if not isinstance(message_id, int):
            message_id = 0
        yield (chat_id, message_id, chat_type, _sender_display(from_obj), text.strip(), msg)


def should_respond(
    *,
    text: str,
    chat_type: str,
    message_obj: dict[str, Any],
    bot_username: str | None,
    trigger_mode: str,
) -> tuple[bool, str | None, str]:
    """
    Decide if the bot should respond. Returns (should, command, prompt_text).
    """
    entities = message_obj.get("entities")
    cmd, cmd_target, rest = _parse_command(text, entities)

    if (chat_type or "").lower() == "private":
        # Always respond in 1:1 chats.
        return (True, cmd, rest if cmd else text.strip())

    # For groups: respond only when explicitly triggered.
    mode = (trigger_mode or "command_or_mention").strip().lower()

    if cmd:
        bot_u = (bot_username or "").strip().lstrip("@").lower()
        if cmd_target and bot_u and cmd_target != bot_u:
            return (False, None, text.strip())
        return (True, cmd, rest)

    bot_u = (bot_username or "").strip().lstrip("@")
    mentioned = bool(bot_u) and (f"@{bot_u}".lower() in (text or "").lower())

    reply_to = message_obj.get("reply_to_message") or {}
    is_reply_to_bot = False
    if isinstance(reply_to, dict):
        r_from = reply_to.get("from") or {}
        if isinstance(r_from, dict) and r_from.get("is_bot"):
            # If we know our username, ensure it matches; otherwise assume any bot reply counts.
            if bot_u:
                is_reply_to_bot = (r_from.get("username") or "").lower() == bot_u.lower()
            else:
                is_reply_to_bot = True

    if mode in ("mention", "command_or_mention", "mention_or_reply"):
        if mentioned:
            return (True, None, _strip_bot_mention(text, bot_u))
    if mode in ("reply", "mention_or_reply"):
        if is_reply_to_bot:
            return (True, None, text.strip())

    return (False, None, text.strip())


def _split_for_telegram(text: str, max_chars: int = TELEGRAM_MAX_MESSAGE_CHARS) -> list[str]:
    s = (text or "").strip()
    if not s:
        return []
    if max_chars <= 0 or len(s) <= max_chars:
        return [s]

    parts: list[str] = []
    rest = s
    while rest:
        chunk = rest[:max_chars]
        cut = max(chunk.rfind("\n"), chunk.rfind(" "))
        if cut >= int(max_chars * 0.7):
            chunk = rest[:cut]
        parts.append(chunk.strip())
        rest = rest[len(chunk) :].lstrip()
    return parts


async def send_telegram_text(
    *,
    token: str,
    chat_id: int,
    text: str,
    reply_to_message_id: int | None = None,
) -> None:
    """
    Send a Telegram message via sendMessage. Splits long texts to fit Telegram limits.
    """
    if not token:
        raise RuntimeError("Missing Telegram bot token")
    base = f"https://api.telegram.org/bot{token}"
    url = f"{base}/sendMessage"

    delay_ms = int(os.getenv("TG_SEND_DELAY_MS", "250"))
    parts = _split_for_telegram(text, TELEGRAM_MAX_MESSAGE_CHARS)
    async with httpx.AsyncClient(timeout=20.0) as client:
        for i, part in enumerate(parts):
            payload: dict[str, Any] = {
                "chat_id": chat_id,
                "text": part,
                "disable_web_page_preview": True,
            }
            if reply_to_message_id:
                payload["reply_to_message_id"] = reply_to_message_id
                payload["allow_sending_without_reply"] = True
            r = await client.post(url, json=payload)
            if r.status_code >= 400:
                raise RuntimeError(f"Telegram send failed ({r.status_code}): {r.text}")
            if i != len(parts) - 1 and delay_ms > 0:
                import asyncio

                await asyncio.sleep(delay_ms / 1000.0)


async def get_updates(*, token: str, offset: int | None = None, timeout: int = 30) -> dict[str, Any]:
    """
    Long-poll updates via getUpdates.
    """
    if not token:
        raise RuntimeError("Missing Telegram bot token")
    base = f"https://api.telegram.org/bot{token}"
    url = f"{base}/getUpdates"
    params: dict[str, Any] = {"timeout": int(timeout)}
    if offset is not None:
        params["offset"] = int(offset)
    async with httpx.AsyncClient(timeout=float(timeout) + 10.0) as client:
        r = await client.get(url, params=params)
        if r.status_code >= 400:
            raise RuntimeError(f"Telegram getUpdates failed ({r.status_code}): {r.text}")
        return r.json()
