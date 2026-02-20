import os
from openai import OpenAI

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Missing OPENAI_API_KEY. Set it in your shell (or load it from a.env) before running.")

client = OpenAI(api_key=api_key)

models = client.models.list()

for m in models.data:
    print(m.id)
