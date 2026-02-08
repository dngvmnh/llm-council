"""
Google Cloud Function (2nd gen) HTTP handler for the multi-LLM debate API.
Deploy with: gcloud functions deploy debate-api --gen2 --runtime python311 --trigger-http ...
Set environment variables (e.g. OPENAI_API_KEY, GEMINI_API_KEY, ...) in the function config.
"""
import json
import sys
from pathlib import Path

_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from core import run_debate_round
from handlers.shared import parse_body, serialize_responses


async def _handle(messages):
    responses = await run_debate_round(messages)
    return {"responses": serialize_responses(responses)}


def debate_http(request):
    """HTTP Cloud Function entrypoint. Expects POST with JSON body: {"messages": [{"role":"user","content":"..."}]}"""
    import asyncio

    if request.method != "POST":
        return (json.dumps({"error": "Method not allowed"}), 405, {"Content-Type": "application/json"})
    try:
        body = request.get_data(as_text=True) if hasattr(request, "get_data") else (request.data or b"").decode("utf-8")
        messages = parse_body(body or "{}")
        if not messages:
            return (
                json.dumps({"error": "messages array required"}),
                400,
                {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            )
        result = asyncio.run(_handle(messages))
        return (
            json.dumps(result),
            200,
            {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
        )
    except Exception as e:
        return (
            json.dumps({"error": str(e)}),
            500,
            {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
        )
