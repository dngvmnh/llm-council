import { useState, useRef, useEffect } from "react";
import { sendDebateRound, sendDebateRoundStream, getAvailableModels } from "./api";
import "./App.css";
import type { ChatMessage, ProviderResponse } from "./api";

type Turn = {
  user: string;
  responses: ProviderResponse[];
};

const SELECTED_MODELS_STORAGE_KEY = "llmCouncil.selectedModels";
const DEFAULT_PINNED_MODELS = ["gpt-5.2-chat-latest", "GPT-4o", "o3-mini"];
const DEFAULT_SELECTED_MODEL_COUNT = 100;
const QUICK_MODEL_COUNTS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50];
const DEBATE_FEED_PROMPTS = [
  "AI vs jobs: will automation create more roles than it destroys?",
  "Should governments require AI watermarking for generated media?",
  "Open-source frontier models: freedom or unacceptable risk?",
  "Is model reasoning (o-series) worth the cost vs fast GPT models?",
  "What should be banned in political deepfakes, if anything?",
  "Are we in an AI hype bubble or a real productivity shift?",
];

/** Display name for provider_id (OpenAI model IDs). */
function getProviderLabel(id: string): string {
  const known: Record<string, string> = {
    moderator: "Moderator",
    pro: "Pro",
    con: "Con",
    judge: "Judge",
    system: "System",
    "gpt-4o-mini": "GPT-4o mini",
    "gpt-4o": "GPT-4o",
    "o3-mini": "o3-mini",
  };
  return known[id] ?? id;
}

function loadStoredSelectedModels(): string[] | null {
  try {
    const raw = localStorage.getItem(SELECTED_MODELS_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return null;
    return parsed.filter((v) => typeof v === "string") as string[];
  } catch {
    return null;
  }
}

function pickDefaultModels(available: string[]): string[] {
  const out: string[] = [];
  const availableSet = new Set(available);
  for (const id of DEFAULT_PINNED_MODELS) {
    if (availableSet.has(id) && !out.includes(id)) out.push(id);
    if (out.length >= DEFAULT_SELECTED_MODEL_COUNT) return out;
  }
  for (const id of available) {
    if (!out.includes(id)) out.push(id);
    if (out.length >= DEFAULT_SELECTED_MODEL_COUNT) return out;
  }
  return out;
}

const PROVIDER_COLORS: Record<string, string> = {
  moderator: "var(--accent)",
  pro: "var(--chatgpt)",
  con: "#e07070",
  judge: "var(--claude)",
  system: "var(--muted)",
  "gpt-4o-mini": "var(--chatgpt)",
  "gpt-4o": "var(--chatgpt)",
  "o3-mini": "var(--chatgpt)",
};

export default function App() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(true);
  const [pipeline, setPipeline] = useState<"panel" | "debate">("debate");
  const [streamingTurn, setStreamingTurn] = useState<{
    user: string;
    content: Record<string, string>;
    errors: Record<string, string>;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [showModelSelector, setShowModelSelector] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, loading, streamingTurn]);

  useEffect(() => {
    getAvailableModels().then((models) => {
      setAvailableModels(models);

      const stored = loadStoredSelectedModels();
      if (stored && stored.length > 0) {
        const filtered = stored.filter((m) => models.includes(m));
        if (filtered.length > 0) {
          // If some stored models disappeared (e.g. filtered out as non-chat models),
          // top back up to the default count with pinned defaults.
          const padded = [...filtered];
          if (padded.length < DEFAULT_SELECTED_MODEL_COUNT) {
            for (const id of pickDefaultModels(models)) {
              if (!padded.includes(id)) padded.push(id);
              if (padded.length >= DEFAULT_SELECTED_MODEL_COUNT) break;
            }
          }
          setSelectedModels(new Set(padded));
          return;
        }
      }

      setSelectedModels(new Set(pickDefaultModels(models)));
    });
  }, []);

  useEffect(() => {
    try {
      if (availableModels.length === 0) return;
      localStorage.setItem(SELECTED_MODELS_STORAGE_KEY, JSON.stringify(Array.from(selectedModels)));
    } catch {
      // ignore storage failures (private mode, etc.)
    }
  }, [selectedModels, availableModels.length]);

  function truncateForPrompt(text: string, maxChars: number): string {
    const s = text.trim();
    if (s.length <= maxChars) return s;
    if (maxChars <= 3) return s.slice(0, maxChars);
    return `${s.slice(0, maxChars - 3)}...`;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;
    if (selectedModels.size === 0) {
      setError("Please select at least one model to participate.");
      return;
    }
    setInput("");
    setError(null);

    const includeFromTurn = Math.max(0, turns.length - 2);
    const messages: ChatMessage[] = [
      ...turns.flatMap((t, idx) => {
        const turnMessages: ChatMessage[] = [{ role: "user" as const, content: t.user }];
        // Keep full transcript behavior, but cap size to control cost:
        // - include model replies only for the last 2 turns
        // - truncate each model reply to 800 chars
        if (idx >= includeFromTurn) {
          const modelResponses = t.responses
            .filter((r) => r.content && !r.error)
            .map((r) => ({
              role: "assistant" as const,
              content: `Model ${getProviderLabel(r.provider_id)}: ${truncateForPrompt(r.content, 800)}`,
            }));
          turnMessages.push(...modelResponses);
        }
        return turnMessages;
      }),
      { role: "user", content: text },
    ];

    setLoading(true);
    if (streaming) {
      setStreamingTurn({ user: text, content: {}, errors: {} });
      try {
        await sendDebateRoundStream(messages, (ev) => {
          const id = ev.model_id;
          setStreamingTurn((prev) => {
            if (!prev) return prev;
            const next = { ...prev, content: { ...prev.content }, errors: { ...prev.errors } };
            if ("delta" in ev && ev.delta) next.content[id] = (next.content[id] ?? "") + ev.delta;
            if ("error" in ev && ev.error) next.errors[id] = ev.error;
            return next;
          });
        }, Array.from(selectedModels), pipeline);
        setStreamingTurn((prev) => {
          if (!prev) return null;
          const responses: ProviderResponse[] = Object.entries(prev.content).map(([provider_id, content]) => ({
            provider_id,
            content,
            error: prev.errors[provider_id] ?? null,
          }));
          Object.keys(prev.errors).forEach((id) => {
            if (!responses.some((r) => r.provider_id === id))
              responses.push({ provider_id: id, content: "", error: prev.errors[id] ?? null });
          });
          setTurns((t) => [...t, { user: prev.user, responses }]);
          return null;
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : "Request failed");
        setStreamingTurn(null);
      } finally {
        setLoading(false);
      }
      return;
    }

    try {
      const { responses } = await sendDebateRound(messages, Array.from(selectedModels), pipeline);
      setTurns((prev) => [...prev, { user: text, responses }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  function toggleModel(modelId: string) {
    setSelectedModels((prev) => {
      const next = new Set(prev);
      if (next.has(modelId)) {
        next.delete(modelId);
      } else {
        next.add(modelId);
      }
      return next;
    });
  }

  function selectAll() {
    setSelectedModels(new Set(availableModels));
  }

  function deselectAll() {
    setSelectedModels(new Set());
  }

  function selectCount(n: number) {
    const limit = Math.max(0, Math.min(n, availableModels.length));
    const out: string[] = [];
    const availableSet = new Set(availableModels);

    for (const id of DEFAULT_PINNED_MODELS) {
      if (availableSet.has(id) && !out.includes(id)) out.push(id);
      if (out.length >= limit) break;
    }
    for (const id of availableModels) {
      if (!out.includes(id)) out.push(id);
      if (out.length >= limit) break;
    }

    setSelectedModels(new Set(out));
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Multiple LLMs Council</h1>
        <p className="subtitle">Chat with multiple LLMs in one place</p>
        <button
          className="model-selector-toggle"
          onClick={() => setShowModelSelector(!showModelSelector)}
          type="button"
        >
          {showModelSelector ? "Hide" : "Select"} Models ({selectedModels.size}/{availableModels.length})
        </button>
      </header>

      {showModelSelector && (
        <div className="model-selector">
          <div className="model-selector-header">
            <span>
              {pipeline === "debate"
                ? "Select model pool (used for Moderator / Pro / Con / Judge):"
                : "Select models to include:"}
            </span>
            <div className="model-selector-actions">
              <button
                type="button"
                onClick={selectAll}
                className="count-button"
                disabled={loading || availableModels.length === 0}
              >
                All
              </button>
              <button
                type="button"
                onClick={deselectAll}
                className="count-button"
                disabled={loading || availableModels.length === 0}
              >
                None
              </button>
              {QUICK_MODEL_COUNTS.map((n) => (
                <button
                  key={n}
                  type="button"
                  onClick={() => selectCount(n)}
                  className="count-button"
                  disabled={loading || availableModels.length === 0}
                  title={`Select ${n} models`}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>
          <div className="model-checkboxes">
            {availableModels.map((modelId) => (
              <label key={modelId} className="model-checkbox">
                <input
                  type="checkbox"
                  checked={selectedModels.has(modelId)}
                  onChange={() => toggleModel(modelId)}
                  disabled={loading}
                />
                <span>{getProviderLabel(modelId)}</span>
              </label>
            ))}
          </div>
        </div>
      )}

      <main className="main">
        <div className="thread">
          {turns.length === 0 && !loading && (
            <div className="empty">
              <p className="feed-title">Debate feed</p>
              <p className="feed-subtitle">What should we debate today? Pick a prompt or write your own.</p>
              <div className="feed-prompts">
                {DEBATE_FEED_PROMPTS.map((p) => (
                  <button key={p} type="button" className="feed-prompt" onClick={() => setInput(p)} disabled={loading}>
                    {p}
                  </button>
                ))}
              </div>
              {availableModels.length === 0 ? (
                <p className="muted">No models detected. Set `OPENAI_API_KEY` in `a.env` and refresh.</p>
              ) : (
                <p className="muted">
                  {pipeline === "debate"
                    ? "Tip: Debate mode runs 4 roles; your selection is the pool."
                    : "Tip: keep your model selection small to save credits."}
                </p>
              )}
            </div>
          )}

          {turns.map((turn, i) => (
            <div key={i} className="turn">
              <div className="user-bubble">
                <span className="label">You</span>
                <div className="content">{turn.user}</div>
              </div>
              <div className="responses">
                {turn.responses.map((r) => (
                  <div
                    key={r.provider_id}
                    className="response-card"
                    style={{ "--provider-color": PROVIDER_COLORS[r.provider_id] || "var(--muted)" } as React.CSSProperties}
                  >
                    <span className="card-label">
                      {getProviderLabel(r.provider_id)}
                    </span>
                    {r.error ? (
                      <p className="error">{r.error}</p>
                    ) : (
                      <div className="content">{r.content || "—"}</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}

          {streamingTurn && (
            <div className="turn">
              <div className="user-bubble">
                <span className="label">You</span>
                <div className="content">{streamingTurn.user}</div>
              </div>
              <div className="responses">
                {[
                  ...Object.keys(streamingTurn.content),
                  ...Object.keys(streamingTurn.errors),
                ]
                  .filter((v, i, a) => a.indexOf(v) === i)
                  .map((modelId) => (
                    <div
                      key={modelId}
                      className="response-card streaming"
                      style={{ "--provider-color": PROVIDER_COLORS[modelId] || "var(--muted)" } as React.CSSProperties}
                    >
                      <span className="card-label">{getProviderLabel(modelId)}</span>
                      {streamingTurn.errors[modelId] ? (
                        <p className="error">{streamingTurn.errors[modelId]}</p>
                      ) : (
                        <div className="content">
                          {streamingTurn.content[modelId] || "…"}
                        </div>
                      )}
                    </div>
                  ))}
              </div>
            </div>
          )}

          {loading && !streamingTurn && (
            <div className="loading-row">
              <span className="loading-dots">
                {pipeline === "debate"
                  ? "Running debate (Moderator / Pro / Con / Judge)…"
                  : `Asking ${selectedModels.size} models…`}
              </span>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {error && (
          <div className="banner error">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="form">
          <label className="stream-toggle">
            <input
              type="checkbox"
              checked={streaming}
              onChange={(e) => setStreaming(e.target.checked)}
              disabled={loading}
            />
            <span>Stream</span>
          </label>
          <div className="mode-toggle" role="group" aria-label="Pipeline">
            <button
              type="button"
              className={`mode-button ${pipeline === "debate" ? "active" : ""}`}
              onClick={() => setPipeline("debate")}
              disabled={loading}
              title="Structured debate: Moderator -> Pro/Con -> Judge"
            >
              Debate
            </button>
            <button
              type="button"
              className={`mode-button ${pipeline === "panel" ? "active" : ""}`}
              onClick={() => setPipeline("panel")}
              disabled={loading}
              title="Panel: ask all selected models in parallel"
            >
              Panel
            </button>
          </div>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSubmit(e)}
            placeholder="Type your message…"
            rows={2}
            disabled={loading}
          />
          <button type="submit" disabled={loading || !input.trim() || selectedModels.size === 0}>
            Send
          </button>
        </form>
      </main>
    </div>
  );
}
