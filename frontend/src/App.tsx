import { useState, useRef, useEffect } from "react";
import { sendDebateRound, sendDebateRoundStream, getAvailableModels } from "./api";
import "./App.css";
import type { ChatMessage, ProviderResponse } from "./api";

type Turn = {
  user: string;
  responses: ProviderResponse[];
};

/** Display name for provider_id (native + OpenRouter/Groq model slugs). */
function getProviderLabel(id: string): string {
  const known: Record<string, string> = {
    chatgpt: "ChatGPT",
    gemini: "Gemini",
    grok: "Grok",
    kimi: "Kimi",
    claude: "Claude",
  };
  return known[id] ?? id;
}

const PROVIDER_COLORS: Record<string, string> = {
  chatgpt: "var(--chatgpt)",
  gemini: "var(--gemini)",
  grok: "var(--grok)",
  kimi: "var(--kimi)",
  claude: "var(--claude)",
};

export default function App() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
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
      setSelectedModels(new Set(models)); // All selected by default
    });
  }, []);

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

    const messages: ChatMessage[] = [
      ...turns.flatMap((t) => {
        const turnMessages: ChatMessage[] = [{ role: "user" as const, content: t.user }];
        // Include all model responses so each model sees what others said
        const modelResponses = t.responses
          .filter((r) => r.content && !r.error)
          .map((r) => ({
            role: "assistant" as const,
            content: `Model ${getProviderLabel(r.provider_id)}: ${r.content}`,
          }));
        turnMessages.push(...modelResponses);
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
        }, Array.from(selectedModels));
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
      const { responses } = await sendDebateRound(messages, Array.from(selectedModels));
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

  return (
    <div className="app">
      <header className="header">
        <h1>Multi-LLM Debate</h1>
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
            <span>Select models to include:</span>
            <div className="model-selector-actions">
              <button type="button" onClick={selectAll} className="link-button">
                All
              </button>
              <button type="button" onClick={deselectAll} className="link-button">
                None
              </button>
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
              <p>Send a message to get responses from all configured LLMs at once.</p>
              <p className="muted">Add your Groq and OpenRouter API keys to the backend to enable each provider.</p>
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
              <span className="loading-dots">Asking all models…</span>
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
