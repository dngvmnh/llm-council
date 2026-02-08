"""Shared request/response handling for Lambda and GCP."""
import json
from dataclasses import asdict
from providers import Message, ProviderResponse


def parse_body(body: str | bytes) -> list[Message]:
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    data = json.loads(body or "{}")
    messages = data.get("messages") or []
    return [Message(role=m.get("role", "user"), content=m.get("content", "")) for m in messages]


def serialize_responses(responses: list[ProviderResponse]) -> list[dict]:
    return [asdict(r) for r in responses]
