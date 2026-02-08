"""
AWS Lambda handler for the multi-LLM debate API.
Configure the Lambda to use Python 3.11+, set handler to handlers.aws_lambda.handler,
and set environment variables for each LLM API key you want to enable.
"""
import json
import sys
from pathlib import Path

# Ensure backend root is on path when running from Lambda (working dir is often the deployment package root)
_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from core import run_debate_round
from handlers.shared import parse_body, serialize_responses


async def _handle(messages):
    responses = await run_debate_round(messages)
    return {"responses": serialize_responses(responses)}


def handler(event, context):
    try:
        body = event.get("body") or event.get("messages")
        if body is None and "requestContext" in event:
            body = event.get("body", "{}")
        if isinstance(body, dict):
            body = json.dumps(body)
        messages = parse_body(body or "{}")
        if not messages:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"error": "messages array required"}),
            }
        import asyncio
        result = asyncio.run(_handle(messages))
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps(result),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)}),
        }
