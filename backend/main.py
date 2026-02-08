"""
GCP Cloud Functions entrypoint. Exposes debate_http for 2nd gen HTTP functions.
Set entry-point to main.debate_http when deploying.
"""
from handlers.gcp_function import debate_http

__all__ = ["debate_http"]
