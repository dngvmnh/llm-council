# Multilple LLMs Council

A shared environment where **you** chat with multiple OpenAI models at once. Each message is sent to the selected models in parallel; responses appear side-by-side so you can compare and continue the discussion.

- **Backend**: Python, deployable to **AWS Lambda** or **Google Cloud Functions** (or run locally).
- **Frontend**: React + Vite; single-page chat UI with optional **Stream** toggle for live updates.
- **CI/CD**: GitHub Actions to deploy the backend to Lambda or GCP on push to `main`.

(Aegis framework is not used; this is a standalone app.)

## Quick start

You need **two terminals**: one for the backend (API) and one for the frontend (the web UI you open in a browser).

### 1. Backend (terminal 1)

```bash
cd backend
python -m venv .venv

# Activate the venv (use one for your OS):
#   Windows (cmd):  .venv\Scripts\activate
#   Windows (PowerShell):  .venv\Scripts\Activate.ps1
source .venv/bin/activate

pip install -r requirements.txt

# Secrets live in the project root `a.env` (ignored by git):
cp ../a.env.example ../a.env
# Edit ../a.env and set OPENAI_API_KEY

# Optional non-secret config (can be in backend/.env or the project root .env; both are ignored by git):
cp ../.env.example .env
python server.py
```

Backend runs at `http://localhost:8080` (API only; no web UI here).

### 2. Frontend (terminal 2)

In a **new terminal**, from the project root:

```bash
cd frontend
npm install
npm run dev
```

Vite will print a URL like **`http://localhost:5173`**. **Open that URL in your browser** to use the app. The frontend talks to the backend on port 8080; keep the backend running.

**Check that it runs:** With the backend up, open `http://localhost:8080` in a browser — you should see `{"ok":true,"service":"multi-llm-debate"}`. Then use the frontend URL to send a message; you should get responses from each configured provider. The backend loads env from `a.env`/`.env` in the project root or `backend/` automatically.

### 3. Environment variables

Copy `a.env.example` to `a.env` (project root) for secrets. Optionally copy `.env.example` to `.env` for non-secret config.

| Variable | Purpose |
|----------|--------|
| **`OPENAI_API_KEY`** | OpenAI API key (used for both model listing and chat). |
| **`OPENAI_DEBATE_MODELS`** | Optional. Comma-separated model IDs to use for debate fan-out. |
| **`MAX_MODELS_PER_ROUND`** | Optional. Hard cap on how many models can run per message (default `3`). |
| **`DEFAULT_MAX_TOKENS`** | Optional. Default max output tokens per model call (default `256`). |

### Instagram (optional)

This repo can also act as an **Instagram DM bot** via the official Instagram Messaging API.

Important limitation: the official API supports **1:1 conversations only** (no group DMs). The bot can still *simulate* a “group chat” by replying as multiple personas (Moderator/Pro/Con/Judge) in sequence.

Backend endpoints:
- `GET /webhooks/instagram` (verification)
- `POST /webhooks/instagram` (message receiver)

Required secrets (put them in `a.env`):
- `IG_ACCESS_TOKEN`
- `IG_USER_ID`
- `IG_VERIFY_TOKEN`
- `IG_APP_SECRET` (optional but recommended for webhook signature verification)

Optional config:
- `IG_MODEL_POOL` (comma-separated OpenAI model IDs used as the pool for role assignment)

### Telegram (optional)

This repo can also act as a **Telegram group chat bot** via the Telegram Bot API.

Backend endpoint:
- `POST /webhooks/telegram` (webhook receiver)

Secrets (put them in `a.env`):
- `TG_BOT_TOKEN`
- `TG_BOT_USERNAME` (optional; used for @mention detection in groups)
- `TG_WEBHOOK_SECRET_TOKEN` (optional; if set, the backend verifies `X-Telegram-Bot-Api-Secret-Token`)

Useful config (in `.env`):
- `TG_TRIGGER` (default `command_or_mention`)
- `TG_DEFAULT_PIPELINE` (default `debate`)
- `TG_MODEL_POOL` (comma-separated OpenAI model IDs used as the pool for role assignment)
- `TG_PANEL_COMBINE=1` (default; avoids spamming groups with one message per model)

For local dev without a public URL, you can enable long polling:
- set `TG_POLLING=1`
- do **not** set a Telegram webhook (polling conflicts with webhooks)

## Deploying

### AWS Lambda

1. Create a Lambda function (Python 3.11), handler `handlers.aws_lambda.handler`, with an execution role that allows `lambda:UpdateFunctionCode` (and create if you use the create path).
2. In the repo **Secrets**, set:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_REGION` (optional; default `us-east-1`)
   - `AWS_LAMBDA_ROLE_ARN` (only if the workflow will create the function)
3. Optionally set `LAMBDA_FUNCTION_NAME` (default `multi-llm-debate`) in the workflow or as a repo variable.
4. Push to `main` or run the **Deploy to AWS Lambda** workflow. Then set the Lambda’s environment variables (each LLM API key) in the AWS console.

After deploy, use the Lambda URL or API Gateway URL as `VITE_API_URL` for the frontend (e.g. `https://xxx.execute-api.region.amazonaws.com/prod`), and ensure the request body is `{"messages": [...]}`.

### Google Cloud Functions

1. Create a GCP project and enable Cloud Functions (Gen 2).
2. In the repo **Secrets**, set:
   - `GCP_SA_KEY`: JSON key for a service account with Cloud Functions Admin (and Storage if needed).
   - `GCP_PROJECT_ID`
   - `GCP_REGION` (optional; default `us-central1`)
3. Push to `main` or run the **Deploy to GCP Cloud Functions** workflow.
4. In GCP Console → Functions → **debate-api** → Edit → Environment variables, add each LLM API key.

The function URL will look like `https://REGION-PROJECT.cloudfunctions.net/debate-api`. Use this as `VITE_API_URL` (and call `POST /debate` from the frontend; you may need to adjust the frontend base path to the function path).

## Project layout

```
multi-llm-debate/
├── backend/
│   ├── providers/       # OpenAI adapters (one key, N models)
│   ├── handlers/       # AWS Lambda + GCP HTTP handlers
│   ├── core.py          # Debate orchestration (fan-out to all providers)
│   ├── server.py        # Local FastAPI dev server
│   ├── main.py          # GCP entrypoint
│   └── requirements.txt
├── frontend/            # Vite + React app
├── .github/workflows/
│   ├── deploy-aws.yml
│   └── deploy-gcp.yml
├── a.env.example
├── .env.example
└── README.md
```

## API

- **POST /debate**  
  Body: `{ "messages": [ { "role": "user"|"assistant", "content": "..." } ] }`  
  Response: `{ "responses": [ { "provider_id": "<model or slug>", "content": "...", "error": null|"..." } ] }`  
  All selected OpenAI models are called in parallel.

- **POST /debate/stream** (local server)  
  Same body. Response: `text/event-stream` with SSE events `data: {"model_id":"...","delta":"..."}` or `{"model_id":"...","done":true}` or `{"model_id":"...","error":"..."}`.  
  In the frontend, enable **Stream** to use this and see responses appear in real time.

## License

MIT.
