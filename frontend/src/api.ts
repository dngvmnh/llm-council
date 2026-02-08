const API_BASE = import.meta.env.VITE_API_URL || "/api";

export type MessageRole = "user" | "assistant";

export interface ChatMessage {
  role: MessageRole;
  content: string;
}

export interface ProviderResponse {
  provider_id: string;
  content: string;
  error: string | null;
}

export interface DebateResponse {
  responses: ProviderResponse[];
}

/** SSE stream event: { model_id, delta? } | { model_id, done? } | { model_id, error? } */
export type StreamEvent =
  | { model_id: string; delta?: string }
  | { model_id: string; done?: boolean }
  | { model_id: string; error?: string };

export async function getAvailableModels(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) {
    return [];
  }
  const data = await res.json();
  return data.models || [];
}

export async function sendDebateRound(messages: ChatMessage[], modelIds?: string[]): Promise<DebateResponse> {
  const res = await fetch(`${API_BASE}/debate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, model_ids: modelIds }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { error?: string }).error || res.statusText);
  }
  return res.json();
}

/** Stream debate via SSE (OpenRouter or Groq). Calls onEvent for each parsed event. */
export async function sendDebateRoundStream(
  messages: ChatMessage[],
  onEvent: (ev: StreamEvent) => void,
  modelIds?: string[]
): Promise<void> {
  const res = await fetch(`${API_BASE}/debate/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, model_ids: modelIds }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { error?: string }).error || res.statusText);
  }
  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (line.startsWith("data: ")) {
        try {
          const data = JSON.parse(line.slice(6)) as StreamEvent;
          onEvent(data);
        } catch {
          // skip invalid JSON
        }
      }
    }
  }
}
