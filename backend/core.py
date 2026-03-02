"""
Debate orchestration: fan-out to selected OpenAI models.
"""
import asyncio
import os
import time
import hashlib
import json
from typing import List

from providers import (
    Message,
    ProviderResponse,
)
from providers.openai_multi_provider import (
    filter_chat_model_ids,
    get_openai_responses,
    get_debate_models,
    list_available_openai_models,
    uses_max_completion_tokens,
)

DEFAULT_SYSTEM = (
    "You are participating in a shared discussion with the user and other AI assistants. "
    "You can see what other models have said in previous messages (they are labeled with their model names). "
    "Read their responses carefully, and feel free to agree, disagree, build upon, or challenge their viewpoints. "
    "Give a clear, concise response that engages with the conversation."
)

# Simple in-memory cache for recent debate requests (key -> (ts, responses_list))
_CACHE: dict = {}
_CACHE_LOCK = asyncio.Lock()
# TTL for cache entries (seconds)
_CACHE_TTL = int(os.getenv("DEFAULT_CACHE_TTL_SEC", "300"))

# Default max tokens (applies to provider requests when supported)
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "256"))


def _has_any_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


async def get_available_models() -> list[str]:
    """Return OpenAI chat-capable model IDs for frontend selection."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return []
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        models = await list_available_openai_models(client)
        # Ensure debate defaults are always present if the API listing filters them out.
        return filter_chat_model_ids(set(models).union(set(get_debate_models())))
    except Exception:
        # Fallback to configured debate models.
        return filter_chat_model_ids(get_debate_models())


async def run_debate_round(
    messages: List[Message],
    system_prompt: str | None = None,
    selected_models: List[str] | None = None,
) -> List[ProviderResponse]:
    """Send conversation to selected OpenAI models in parallel."""
    if not _has_any_key():
        return [
            ProviderResponse(
                provider_id="system",
                content="",
                error="No LLM API keys configured. Set OPENAI_API_KEY.",
            )
        ]
    system = system_prompt or DEFAULT_SYSTEM

    # Guardrail: cap number of models per round to avoid accidental credit burn.
    max_models_per_round = int(os.getenv("MAX_MODELS_PER_ROUND", "100"))
    effective_selected_models: list[str] | None = None
    truncated_notice: str | None = None
    if selected_models:
        # Preserve caller order, drop duplicates.
        deduped = list(dict.fromkeys(selected_models))
        if max_models_per_round > 0 and len(deduped) > max_models_per_round:
            truncated = deduped[:max_models_per_round]
            truncated_notice = (
                f"Selected {len(deduped)} models but MAX_MODELS_PER_ROUND={max_models_per_round}; "
                f"running only: {', '.join(truncated)}"
            )
            deduped = truncated
        effective_selected_models = deduped

    # Compute cache key (messages + system + selected_models)
    try:
        key_obj = {
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "selected_models": effective_selected_models or [],
        }
        key_raw = json.dumps(key_obj, sort_keys=True, ensure_ascii=False)
        key = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()
    except Exception:
        key = None

    # Return cached responses if fresh
    if key is not None:
        async with _CACHE_LOCK:
            item = _CACHE.get(key)
            if item:
                ts, cached = item
                if time.time() - ts < _CACHE_TTL:
                    # return a shallow copy to avoid mutation
                    return [ProviderResponse(r.provider_id, r.content, r.error) for r in cached]
                else:
                    del _CACHE[key]

    tasks = []
    tasks.append(get_openai_responses(messages, system, selected_models=effective_selected_models))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: List[ProviderResponse] = []
    idx = 0
    r = results[idx]
    if isinstance(r, Exception):
        out.append(ProviderResponse(provider_id="openai", content="", error=str(r)))
    else:
        out.extend(r)

    if truncated_notice:
        out.insert(0, ProviderResponse(provider_id="system", content="", error=truncated_notice))

    # store in cache
    if key is not None:
        async with _CACHE_LOCK:
            try:
                _CACHE[key] = (time.time(), [ProviderResponse(r.provider_id, r.content, r.error) for r in out])
            except Exception:
                pass

    return out


async def stream_debate_round(
    messages: List[Message],
    system_prompt: str | None = None,
    selected_models: List[str] | None = None,
):
    """
    Stream debate responses when using OpenAI.
    Yields dicts: {"model_id": str, "delta": str} | {"model_id": str, "done": True} | {"model_id": str, "error": str}.
    If OPENAI_API_KEY is not set, yields nothing.
    """
    system = system_prompt or DEFAULT_SYSTEM
    max_models_per_round = int(os.getenv("MAX_MODELS_PER_ROUND", "100"))
    effective_selected_models: list[str] | None = None
    truncated_notice: str | None = None
    if selected_models:
        deduped = list(dict.fromkeys(selected_models))
        if max_models_per_round > 0 and len(deduped) > max_models_per_round:
            truncated = deduped[:max_models_per_round]
            truncated_notice = (
                f"Selected {len(deduped)} models but MAX_MODELS_PER_ROUND={max_models_per_round}; "
                f"running only: {', '.join(truncated)}"
            )
            deduped = truncated
        effective_selected_models = deduped

    if truncated_notice:
        yield {"model_id": "system", "delta": truncated_notice}

    openai_messages = [{"role": "system", "content": system}] if system else []
    for m in messages:
        openai_messages.append({"role": m.role, "content": m.content})

    queue: asyncio.Queue = asyncio.Queue()
    expected_sentinels = 0

    if os.environ.get("OPENAI_API_KEY"):
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        models = effective_selected_models or get_debate_models()

        for model_id in models:
            expected_sentinels += 1

            async def stream_one(m_id: str):
                try:
                    params = {"model": m_id, "messages": openai_messages, "stream": True}
                    if uses_max_completion_tokens(m_id):
                        params["max_completion_tokens"] = DEFAULT_MAX_TOKENS
                    else:
                        params["max_tokens"] = DEFAULT_MAX_TOKENS

                    try:
                        stream = await client.chat.completions.create(**params)
                    except Exception as e:
                        msg = str(e)
                        if "Unsupported parameter: 'max_tokens'" in msg and "max_completion_tokens" in msg:
                            params.pop("max_tokens", None)
                            params["max_completion_tokens"] = DEFAULT_MAX_TOKENS
                            stream = await client.chat.completions.create(**params)
                        elif "Unsupported parameter: 'max_completion_tokens'" in msg and "max_tokens" in msg:
                            params.pop("max_completion_tokens", None)
                            params["max_tokens"] = DEFAULT_MAX_TOKENS
                            stream = await client.chat.completions.create(**params)
                        else:
                            raise
                    async for chunk in stream:
                        delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                        if delta:
                            await queue.put({"model_id": m_id, "delta": delta})
                    await queue.put({"model_id": m_id, "done": True})
                except Exception as e:
                    await queue.put({"model_id": m_id, "error": str(e)})
                finally:
                    await queue.put(None)

            asyncio.create_task(stream_one(model_id))

    if expected_sentinels == 0:
        return

    sentinels = 0
    while sentinels < expected_sentinels:
        ev = await queue.get()
        if ev is None:
            sentinels += 1
            continue
        yield ev


def _model_strength_score(model_id: str) -> int:
    s = (model_id or "").lower()
    if s.startswith("o3"):
        return 100
    if s.startswith("o1"):
        return 95
    if s.startswith("gpt-5"):
        return 90
    if "gpt-4o" in s:
        return 80
    if s.startswith("gpt-4"):
        return 70
    if s.startswith("gpt-3.5"):
        return 10
    return 50


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys([i for i in items if isinstance(i, str) and i.strip()]))


def _pick_debate_role_models(selected_models: list[str] | None) -> dict[str, str]:
    """
    Pick model IDs for each debate role.

    We prefer stronger models for moderator/judge, and preserve user selection order
    for pro/con when possible.
    """
    candidates = _dedupe_preserve_order(selected_models or []) or get_debate_models()
    if not candidates:
        return {"pro": "", "con": "", "moderator": "", "judge": ""}

    indexed = list(enumerate(candidates))
    ranked = sorted(indexed, key=lambda t: (-_model_strength_score(t[1]), t[0]))
    judge = ranked[0][1]
    remaining = [m for m in candidates if m != judge]
    moderator = remaining[0] if not remaining else sorted(
        list(enumerate(remaining)),
        key=lambda t: (-_model_strength_score(t[1]), t[0]),
    )[0][1]

    remaining2 = [m for m in candidates if m not in {judge, moderator}]
    pro = remaining2[0] if remaining2 else moderator
    con = remaining2[1] if len(remaining2) > 1 else judge
    return {"pro": pro, "con": con, "moderator": moderator, "judge": judge}


def _last_user_message(messages: list[Message]) -> str:
    for m in reversed(messages):
        if (m.role or "").lower() == "user" and (m.content or "").strip():
            return m.content.strip()
    return ""


def _recent_user_context(messages: list[Message], max_turns: int = 3) -> str:
    user_msgs = [m.content.strip() for m in messages if (m.role or "").lower() == "user" and (m.content or "").strip()]
    if not user_msgs:
        return ""
    take = user_msgs[-max(1, max_turns):]
    if len(take) == 1:
        return take[0]
    return "\n".join([f"{i + 1}. {t}" for i, t in enumerate(take)])


def _role_system(system: str, role: str) -> str:
    role = role.lower()
    if role == "moderator":
        extra = (
            "You are the Moderator of a formal debate. Stay neutral. "
            "Your job is to define the motion, set judging criteria, and ask sharp questions."
        )
    elif role == "pro":
        extra = (
            "You are the Pro side. Argue in favor of the motion. "
            "Be persuasive but honest; do not invent facts."
        )
    elif role == "con":
        extra = (
            "You are the Con side. Argue against the motion. "
            "Be persuasive but honest; do not invent facts."
        )
    elif role == "judge":
        extra = (
            "You are the Judge. Evaluate which side made the stronger case using the criteria. "
            "Call out fallacies, missing assumptions, and give a clear verdict."
        )
    else:
        extra = ""
    return f"{system}\n\n{extra}".strip()


async def run_debate_pro_con_moderator_judge(
    messages: List[Message],
    system_prompt: str | None = None,
    selected_models: List[str] | None = None,
) -> List[ProviderResponse]:
    """
    Structured debate pipeline:
    Moderator frames -> Pro/Con openings -> Moderator cross-exam -> Pro/Con rebuttals -> Judge verdict.

    Returns 4 responses with provider_id: moderator, pro, con, judge.
    """
    if not _has_any_key():
        return [
            ProviderResponse(
                provider_id="system",
                content="",
                error="No LLM API keys configured. Set OPENAI_API_KEY.",
            )
        ]

    system = system_prompt or DEFAULT_SYSTEM
    role_models = _pick_debate_role_models(selected_models)
    motion = _last_user_message(messages) or "(no user prompt provided)"
    user_ctx = _recent_user_context(messages, max_turns=int(os.getenv("DEBATE_USER_CONTEXT_TURNS", "3")))

    role_line = (
        f"Models: Pro={role_models['pro']}, Con={role_models['con']}, "
        f"Moderator={role_models['moderator']}, Judge={role_models['judge']}"
    )
    if selected_models and len(_dedupe_preserve_order(selected_models)) > 4:
        role_line += f" (selected {len(_dedupe_preserve_order(selected_models))}; using 4 roles)"

    base_messages: list[Message] = []
    if user_ctx:
        base_messages.append(Message(role="user", content=f"Conversation context (user only):\n{user_ctx}"))

    async def one(role: str, model_id: str, prompt: str) -> ProviderResponse:
        if not model_id:
            return ProviderResponse(provider_id=role, content="", error="No model assigned for this role.")
        role_msgs = base_messages + [Message(role="user", content=prompt)]
        res = await get_openai_responses(role_msgs, _role_system(system, role), selected_models=[model_id])
        if not res:
            return ProviderResponse(provider_id=role, content="", error="No response.")
        r0 = res[0]
        return ProviderResponse(provider_id=role, content=(r0.content or "").strip(), error=r0.error)

    moderator_framing_prompt = (
        f"{role_line}\n\n"
        f"Debate motion:\n{motion}\n\n"
        "Task:\n"
        "1) Restate the motion neutrally in 1 sentence.\n"
        "2) Define key terms/assumptions.\n"
        "3) Give 3 judging criteria.\n"
        "4) Ask 2 cross-exam questions for Pro and 2 for Con.\n"
        "Output Markdown with headings: Motion, Definitions, Criteria, Questions."
    )
    moderator_framing = await one("moderator", role_models["moderator"], moderator_framing_prompt)

    pro_opening_prompt = (
        f"{role_line}\n\n"
        f"Debate motion:\n{motion}\n\n"
        f"Moderator framing:\n{moderator_framing.content}\n\n"
        "Write the Pro opening statement.\n"
        "- 3-5 claims with reasoning\n"
        "- Anticipate the strongest Con objection\n"
        "- Keep it under 250 words\n"
        "Output Markdown."
    )
    con_opening_prompt = (
        f"{role_line}\n\n"
        f"Debate motion:\n{motion}\n\n"
        f"Moderator framing:\n{moderator_framing.content}\n\n"
        "Write the Con opening statement.\n"
        "- 3-5 counterclaims with reasoning\n"
        "- Identify Pro's weakest assumption\n"
        "- Keep it under 250 words\n"
        "Output Markdown."
    )
    pro_opening, con_opening = await asyncio.gather(
        one("pro", role_models["pro"], pro_opening_prompt),
        one("con", role_models["con"], con_opening_prompt),
    )

    moderator_cross_prompt = (
        f"{role_line}\n\n"
        f"Debate motion:\n{motion}\n\n"
        f"Pro opening:\n{pro_opening.content}\n\n"
        f"Con opening:\n{con_opening.content}\n\n"
        "Task:\n"
        "1) Point out 1 strong and 1 weak point for each side.\n"
        "2) Ask 1 follow-up question to Pro and 1 to Con.\n"
        "Output Markdown with headings: Pro, Con, Follow-ups."
    )
    moderator_cross = await one("moderator", role_models["moderator"], moderator_cross_prompt)

    pro_rebut_prompt = (
        f"{role_line}\n\n"
        f"Debate motion:\n{motion}\n\n"
        f"Con opening:\n{con_opening.content}\n\n"
        f"Moderator follow-ups:\n{moderator_cross.content}\n\n"
        "Write the Pro rebuttal.\n"
        "- Rebut Con's best point\n"
        "- Answer the moderator follow-up\n"
        "- Keep it under 200 words\n"
        "Output Markdown."
    )
    con_rebut_prompt = (
        f"{role_line}\n\n"
        f"Debate motion:\n{motion}\n\n"
        f"Pro opening:\n{pro_opening.content}\n\n"
        f"Moderator follow-ups:\n{moderator_cross.content}\n\n"
        "Write the Con rebuttal.\n"
        "- Rebut Pro's best point\n"
        "- Answer the moderator follow-up\n"
        "- Keep it under 200 words\n"
        "Output Markdown."
    )
    pro_rebut, con_rebut = await asyncio.gather(
        one("pro", role_models["pro"], pro_rebut_prompt),
        one("con", role_models["con"], con_rebut_prompt),
    )

    judge_prompt = (
        f"{role_line}\n\n"
        f"Debate motion:\n{motion}\n\n"
        f"Moderator framing:\n{moderator_framing.content}\n\n"
        f"Pro opening:\n{pro_opening.content}\n\n"
        f"Con opening:\n{con_opening.content}\n\n"
        f"Moderator cross-exam:\n{moderator_cross.content}\n\n"
        f"Pro rebuttal:\n{pro_rebut.content}\n\n"
        f"Con rebuttal:\n{con_rebut.content}\n\n"
        "Task:\n"
        "1) Pick a winner (Pro/Con) and justify using the judging criteria.\n"
        "2) Give a score out of 10 for each side.\n"
        "3) Provide a short 'best answer' to the user.\n"
        "Output Markdown with headings: Verdict, Scores, Best Answer."
    )
    judge = await one("judge", role_models["judge"], judge_prompt)

    moderator_content = f"{role_line}\n\n### Framing\n{moderator_framing.content}\n\n### Cross-exam\n{moderator_cross.content}".strip()
    pro_content = f"{role_line}\n\n### Opening\n{pro_opening.content}\n\n### Rebuttal\n{pro_rebut.content}".strip()
    con_content = f"{role_line}\n\n### Opening\n{con_opening.content}\n\n### Rebuttal\n{con_rebut.content}".strip()
    judge_content = f"{role_line}\n\n{judge.content}".strip()

    return [
        ProviderResponse(provider_id="moderator", content=moderator_content, error=moderator_framing.error or moderator_cross.error),
        ProviderResponse(provider_id="pro", content=pro_content, error=pro_opening.error or pro_rebut.error),
        ProviderResponse(provider_id="con", content=con_content, error=con_opening.error or con_rebut.error),
        ProviderResponse(provider_id="judge", content=judge_content, error=judge.error),
    ]


async def stream_debate_pro_con_moderator_judge(
    messages: List[Message],
    system_prompt: str | None = None,
    selected_models: List[str] | None = None,
):
    """
    SSE streaming version of the structured debate pipeline.

    Emits events keyed by model_id = role (moderator/pro/con/judge).
    """
    system = system_prompt or DEFAULT_SYSTEM
    if not os.environ.get("OPENAI_API_KEY"):
        return

    role_models = _pick_debate_role_models(selected_models)
    motion = _last_user_message(messages) or "(no user prompt provided)"
    user_ctx = _recent_user_context(messages, max_turns=int(os.getenv("DEBATE_USER_CONTEXT_TURNS", "3")))

    role_line = (
        f"Models: Pro={role_models['pro']}, Con={role_models['con']}, "
        f"Moderator={role_models['moderator']}, Judge={role_models['judge']}"
    )
    if selected_models and len(_dedupe_preserve_order(selected_models)) > 4:
        role_line += f" (selected {len(_dedupe_preserve_order(selected_models))}; using 4 roles)"

    base_user_block = f"Conversation context (user only):\n{user_ctx}\n\n" if user_ctx else ""

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    async def stream_one(role: str, model_id: str, header: str, prompt: str):
        if not model_id:
            yield {"model_id": role, "error": "No model assigned for this role."}
            return

        # Emit a header so the UI shows phase boundaries even if the model is slow to start.
        yield {"model_id": role, "delta": header}

        openai_messages = [{"role": "system", "content": _role_system(system, role)}]
        if base_user_block:
            openai_messages.append({"role": "user", "content": base_user_block.strip()})
        openai_messages.append({"role": "user", "content": prompt})

        params = {"model": model_id, "messages": openai_messages, "stream": True}
        if uses_max_completion_tokens(model_id):
            params["max_completion_tokens"] = DEFAULT_MAX_TOKENS
        else:
            params["max_tokens"] = DEFAULT_MAX_TOKENS

        try:
            try:
                stream = await client.chat.completions.create(**params)
            except Exception as e:
                msg = str(e)
                if "Unsupported parameter: 'max_tokens'" in msg and "max_completion_tokens" in msg:
                    params.pop("max_tokens", None)
                    params["max_completion_tokens"] = DEFAULT_MAX_TOKENS
                    stream = await client.chat.completions.create(**params)
                elif "Unsupported parameter: 'max_completion_tokens'" in msg and "max_tokens" in msg:
                    params.pop("max_completion_tokens", None)
                    params["max_tokens"] = DEFAULT_MAX_TOKENS
                    stream = await client.chat.completions.create(**params)
                else:
                    raise
            async for chunk in stream:
                delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                if not delta:
                    continue
                yield {"model_id": role, "delta": delta}
            yield {"model_id": role, "done": True}
        except Exception as e:
            yield {"model_id": role, "error": str(e)}
        return

    # Moderator framing
    moderator_framing_prompt = (
        f"{role_line}\n\n"
        f"Debate motion:\n{motion}\n\n"
        "Task:\n"
        "1) Restate the motion neutrally in 1 sentence.\n"
        "2) Define key terms/assumptions.\n"
        "3) Give 3 judging criteria.\n"
        "4) Ask 2 cross-exam questions for Pro and 2 for Con.\n"
        "Output Markdown with headings: Motion, Definitions, Criteria, Questions."
    )
    moderator_framing = ""
    async for ev in stream_one("moderator", role_models["moderator"], f"{role_line}\n\n### Framing\n\n", moderator_framing_prompt):
        if ev.get("delta"):
            moderator_framing += ev["delta"]
        yield ev

    # Pro opening
    pro_opening_prompt = (
        f"{role_line}\n\n"
        f"Debate motion:\n{motion}\n\n"
        f"Moderator framing:\n{moderator_framing}\n\n"
        "Write the Pro opening statement.\n"
        "- 3-5 claims with reasoning\n"
        "- Anticipate the strongest Con objection\n"
        "- Keep it under 250 words\n"
        "Output Markdown."
    )
    pro_opening = ""
    async for ev in stream_one("pro", role_models["pro"], f"{role_line}\n\n### Opening\n\n", pro_opening_prompt):
        if ev.get("delta"):
            pro_opening += ev["delta"]
        yield ev

    # Con opening
    con_opening_prompt = (
        f"{role_line}\n\n"
        f"Debate motion:\n{motion}\n\n"
        f"Moderator framing:\n{moderator_framing}\n\n"
        "Write the Con opening statement.\n"
        "- 3-5 counterclaims with reasoning\n"
        "- Identify Pro's weakest assumption\n"
        "- Keep it under 250 words\n"
        "Output Markdown."
    )
    con_opening = ""
    async for ev in stream_one("con", role_models["con"], f"{role_line}\n\n### Opening\n\n", con_opening_prompt):
        if ev.get("delta"):
            con_opening += ev["delta"]
        yield ev

    # Moderator cross-exam
    moderator_cross_prompt = (
        f"{role_line}\n\n"
        f"Debate motion:\n{motion}\n\n"
        f"Pro opening:\n{pro_opening}\n\n"
        f"Con opening:\n{con_opening}\n\n"
        "Task:\n"
        "1) Point out 1 strong and 1 weak point for each side.\n"
        "2) Ask 1 follow-up question to Pro and 1 to Con.\n"
        "Output Markdown with headings: Pro, Con, Follow-ups."
    )
    moderator_cross = ""
    async for ev in stream_one("moderator", role_models["moderator"], "\n\n### Cross-exam\n\n", moderator_cross_prompt):
        if ev.get("delta"):
            moderator_cross += ev["delta"]
        yield ev

    # Pro rebuttal
    pro_rebut_prompt = (
        f"{role_line}\n\n"
        f"Debate motion:\n{motion}\n\n"
        f"Con opening:\n{con_opening}\n\n"
        f"Moderator follow-ups:\n{moderator_cross}\n\n"
        "Write the Pro rebuttal.\n"
        "- Rebut Con's best point\n"
        "- Answer the moderator follow-up\n"
        "- Keep it under 200 words\n"
        "Output Markdown."
    )
    pro_rebut = ""
    async for ev in stream_one("pro", role_models["pro"], "\n\n### Rebuttal\n\n", pro_rebut_prompt):
        if ev.get("delta"):
            pro_rebut += ev["delta"]
        yield ev

    # Con rebuttal
    con_rebut_prompt = (
        f"{role_line}\n\n"
        f"Debate motion:\n{motion}\n\n"
        f"Pro opening:\n{pro_opening}\n\n"
        f"Moderator follow-ups:\n{moderator_cross}\n\n"
        "Write the Con rebuttal.\n"
        "- Rebut Pro's best point\n"
        "- Answer the moderator follow-up\n"
        "- Keep it under 200 words\n"
        "Output Markdown."
    )
    con_rebut = ""
    async for ev in stream_one("con", role_models["con"], "\n\n### Rebuttal\n\n", con_rebut_prompt):
        if ev.get("delta"):
            con_rebut += ev["delta"]
        yield ev

    # Judge verdict
    judge_prompt = (
        f"{role_line}\n\n"
        f"Debate motion:\n{motion}\n\n"
        f"Moderator framing:\n{moderator_framing}\n\n"
        f"Pro opening:\n{pro_opening}\n\n"
        f"Con opening:\n{con_opening}\n\n"
        f"Moderator cross-exam:\n{moderator_cross}\n\n"
        f"Pro rebuttal:\n{pro_rebut}\n\n"
        f"Con rebuttal:\n{con_rebut}\n\n"
        "Task:\n"
        "1) Pick a winner (Pro/Con) and justify using the judging criteria.\n"
        "2) Give a score out of 10 for each side.\n"
        "3) Provide a short 'best answer' to the user.\n"
        "Output Markdown with headings: Verdict, Scores, Best Answer."
    )
    async for ev in stream_one("judge", role_models["judge"], f"{role_line}\n\n### Verdict\n\n", judge_prompt):
        yield ev
