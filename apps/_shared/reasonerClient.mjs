// Shared server-side client for vLLM/OpenAI-compatible Cosmos Reasoner APIs.
//
// Env var priority:
//   ALPAMAYO_BASE_URL when INFERENCE_BACKEND=alpamayo,
//   otherwise VLLM_BASE_URL > NIM_BASE_URL > REASONER_BASE_URL > http://localhost:8000/v1
//
// The BYO-video standing order is to send media as data URLs in the OpenAI
// multimodal content array. Do not use file:// or HTML-tag-in-text shapes.

function resolveBaseUrl() {
  if (String(process.env.INFERENCE_BACKEND || "").toLowerCase() === "alpamayo") {
    const alpamayoUrl =
      process.env.ALPAMAYO_BASE_URL ||
      process.env.REASONER_BASE_URL ||
      "http://localhost:8001/v1";
    const strippedAlpamayo = alpamayoUrl.replace(/\/$/, "");
    return strippedAlpamayo.endsWith("/v1") ? strippedAlpamayo : `${strippedAlpamayo}/v1`;
  }
  const raw =
    process.env.VLLM_BASE_URL ||
    process.env.NIM_BASE_URL ||
    process.env.REASONER_BASE_URL ||
    "http://localhost:8000/v1";
  const stripped = raw.replace(/\/$/, "");
  return stripped.endsWith("/v1") ? stripped : `${stripped}/v1`;
}

function configuredModel() {
  return (
    process.env.MODEL_NAME ||
    process.env.MODEL_ID ||
    process.env.ALPAMAYO_MODEL_ID ||
    process.env.VLLM_MODEL ||
    (String(process.env.INFERENCE_BACKEND || "").toLowerCase() === "alpamayo"
      ? "nvidia/Alpamayo-1.5-10B"
      : "nvidia/Cosmos3-Nano-Reasoner")
  );
}

function mediaContent({ mediaDataUrl, mediaKind, mediaFrames, framesPerSecond, prompt }) {
  const promptText = prompt || "Describe the provided media.";
  if (Array.isArray(mediaFrames) && mediaFrames.length > 0) {
    const fps = Number.isFinite(Number(framesPerSecond)) ? Number(framesPerSecond) : 1;
    return [
      { type: "text", text: `[Video — ${mediaFrames.length} frames at ${fps}fps]\n${promptText}` },
      ...mediaFrames.map((url) => ({ type: "image_url", image_url: { url } }))
    ];
  }
  if (!mediaDataUrl) return [{ type: "text", text: promptText }];
  if (mediaKind === "image") {
    return [
      { type: "image_url", image_url: { url: mediaDataUrl } },
      { type: "text", text: promptText }
    ];
  }
  return [
    { type: "video_url", video_url: { url: mediaDataUrl } },
    { type: "text", text: promptText }
  ];
}

export function redactPayload(payload) {
  if (!payload) return payload;
  return {
    ...payload,
    messages: payload.messages?.map((message) => {
      if (!Array.isArray(message.content)) return message;
      return {
        ...message,
        content: message.content.map((entry) => {
          if (entry?.image_url?.url) return { ...entry, image_url: { url: "<data-url>" } };
          if (entry?.video_url?.url) return { ...entry, video_url: { url: "<data-url>" } };
          return entry;
        })
      };
    })
  };
}

function stripAnswerTags(text) {
  return (text || "").replace(/^<answer>\s*/i, "").replace(/\s*<\/answer>$/i, "").trim();
}

function splitInlineReasoning(text) {
  const content = text || "";
  const match = content.match(/<think>([\s\S]*?)<\/think>/i);
  if (!match) {
    return { reasoning: "", answer: stripAnswerTags(content.trim()), combined: stripAnswerTags(content.trim()) };
  }

  const answer = stripAnswerTags(content.slice((match.index || 0) + match[0].length).trim());
  const reasoning = (match[1] || "").trim();
  const combined = reasoning ? `<think>\n${reasoning}\n</think>\n\n${answer}`.trim() : answer;
  return { reasoning, answer, combined };
}

function splitAnswerAfterCloseTag(text) {
  const content = text || "";
  const closeTag = "</think>";
  const closeIndex = content.toLowerCase().lastIndexOf(closeTag);
  if (closeIndex === -1) return stripAnswerTags(content.trim());
  return stripAnswerTags(content.slice(closeIndex + closeTag.length).trim());
}

export function normalizeReasonerMessage(data) {
  const choice = data?.choices?.[0] || {};
  const message = choice.message || {};
  const rawContent = message.content || "";
  const inline = splitInlineReasoning(rawContent);
  const explicitReasoning = (message.reasoning_content || message.reasoning || "").trim();
  const reasoning = explicitReasoning || inline.reasoning;
  const answer = explicitReasoning ? splitAnswerAfterCloseTag(rawContent) : inline.answer;
  const combined = reasoning ? `<think>\n${reasoning}\n</think>\n\n${answer}`.trim() : answer;

  const normalized = {
    id: data?.id || `chatcmpl-byo-${Date.now()}`,
    object: data?.object || "chat.completion",
    created: data?.created || Math.floor(Date.now() / 1000),
    model: data?.model || configuredModel(),
    choices: [
      {
        index: choice.index ?? 0,
        message: {
          role: message.role || "assistant",
          content: combined
        },
        finish_reason: choice.finish_reason || null
      }
    ],
    usage: data?.usage || null
  };

  Object.keys(normalized).forEach((key) => normalized[key] === null && delete normalized[key]);

  return {
    answer,
    reasoning,
    combined,
    normalized,
    schema: explicitReasoning
      ? "vllm_reasoning_fields"
      : inline.reasoning
        ? "nim_inline_think"
        : "plain_content"
  };
}

export async function listReasonerModels() {
  const baseUrl = resolveBaseUrl();
  try {
    const response = await fetch(`${baseUrl}/models`, {
      headers: { Authorization: `Bearer ${process.env.VLLM_API_KEY || process.env.NIM_API_KEY || "EMPTY"}` }
    });
    if (!response.ok) throw new Error(`Model probe failed with HTTP ${response.status}`);
    const data = await response.json();
    const models = Array.isArray(data?.data) ? data.data.map((model) => model.id).filter(Boolean) : [];
    return { baseUrl, models: models.length > 0 ? models : [configuredModel()] };
  } catch (error) {
    return {
      baseUrl,
      models: [configuredModel()],
      warning: error instanceof Error ? error.message : "Unable to reach model endpoint"
    };
  }
}

export function buildReasoningPayload({
  model,
  prompt,
  systemPrompt,
  mediaDataUrl,
  mediaKind,
  mediaFrames,
  framesPerSecond,
  params
} = {}) {
  const baseUrl = resolveBaseUrl();
  const p = params || {};
  const selectedModel = model || configuredModel();
  const inferenceBackend = String(process.env.INFERENCE_BACKEND || "").toLowerCase();
  const content = mediaContent({ mediaDataUrl, mediaKind, mediaFrames, framesPerSecond, prompt });

  const messages = [];
  if (systemPrompt) messages.push({ role: "system", content: systemPrompt });
  messages.push({ role: "user", content });

  const payload = {
    model: selectedModel,
    messages,
    temperature: Number.isFinite(Number(p.temperature)) ? Number(p.temperature) : 0.6,
    top_p: Number.isFinite(Number(p.top_p)) ? Number(p.top_p) : 0.95,
    repetition_penalty: Number.isFinite(Number(p.repetition_penalty)) ? Number(p.repetition_penalty) : undefined,
    presence_penalty: Number.isFinite(Number(p.presence_penalty)) ? Number(p.presence_penalty) : undefined
  };

  // Keep parity with the Gradio C3/vLLM path by default: it exposes this
  // control in the UI, but most OpenAI-compatible services can govern the cap
  // themselves. Alpamayo is local generation, so honoring max_tokens is the
  // fastest user-visible way to reduce full-response latency.
  if (
    (inferenceBackend === "alpamayo" || process.env.REASONER_SEND_MAX_TOKENS === "1") &&
    Number.isFinite(Number(p.max_tokens))
  ) {
    payload.max_tokens = Number(p.max_tokens);
  }
  if (process.env.REASONER_SEND_TOP_K === "1" && Number.isFinite(Number(p.top_k))) {
    payload.top_k = Number(p.top_k);
  }
  if (process.env.REASONER_SEND_SEED === "1" && Number.isFinite(Number(p.seed))) {
    payload.seed = Number(p.seed);
  }
  if (process.env.REASONER_SEND_MM_PROCESSOR_KWARGS === "1" && Number.isFinite(Number(p.frames_per_second))) {
    payload.mm_processor_kwargs = { fps: Number(p.frames_per_second) };
  }

  Object.keys(payload).forEach((key) => payload[key] === undefined && delete payload[key]);

  return { baseUrl, payload, redactedPayload: redactPayload(payload) };
}

function suffixOverlapLength(value, token) {
  const lowerValue = String(value || "").toLowerCase();
  const lowerToken = String(token || "").toLowerCase();
  const max = Math.min(lowerValue.length, lowerToken.length - 1);
  for (let length = max; length > 0; length -= 1) {
    if (lowerValue.endsWith(lowerToken.slice(0, length))) return length;
  }
  return 0;
}

function createInlineContentParser() {
  let mode = "unknown";
  let buffer = "";

  function emit(channel, text) {
    return text ? [{ channel, text }] : [];
  }

  function process() {
    const events = [];
    while (buffer) {
      if (mode === "answer") {
        events.push(...emit("answer", buffer));
        buffer = "";
        break;
      }

      if (mode === "unknown") {
        const openTag = "<think>";
        const index = buffer.toLowerCase().indexOf(openTag);
        if (index >= 0) {
          events.push(...emit("answer", stripAnswerTags(buffer.slice(0, index))));
          buffer = buffer.slice(index + openTag.length);
          mode = "reasoning";
          continue;
        }
        const keep = suffixOverlapLength(buffer, openTag);
        if (buffer.length > keep) {
          const text = buffer.slice(0, buffer.length - keep);
          buffer = buffer.slice(buffer.length - keep);
          events.push(...emit("answer", text));
        }
        break;
      }

      const closeTag = "</think>";
      const index = buffer.toLowerCase().indexOf(closeTag);
      if (index >= 0) {
        events.push(...emit("reasoning", buffer.slice(0, index)));
        buffer = buffer.slice(index + closeTag.length);
        mode = "answer";
        continue;
      }
      const keep = suffixOverlapLength(buffer, closeTag);
      if (buffer.length > keep) {
        const text = buffer.slice(0, buffer.length - keep);
        buffer = buffer.slice(buffer.length - keep);
        events.push(...emit("reasoning", text));
      }
      break;
    }
    return events;
  }

  return {
    push(text) {
      buffer += text || "";
      return process();
    },
    finish() {
      if (!buffer) return [];
      const channel = mode === "reasoning" ? "reasoning" : "answer";
      const text = channel === "answer" ? stripAnswerTags(buffer) : buffer;
      buffer = "";
      return emit(channel, text);
    },
    sawInlineReasoning() {
      return mode === "reasoning" || mode === "answer";
    }
  };
}

export function createReasoningStreamNormalizer() {
  const inline = createInlineContentParser();
  let schema = "plain_content";

  function withSchema(events) {
    return events
      .filter((event) => event.text)
      .map((event) => ({
        ...event,
        schema
      }));
  }

  return {
    accept(data) {
      const choice = data?.choices?.[0] || {};
      const delta = choice.delta || choice.message || {};
      const events = [];
      const explicitReasoning = delta.reasoning_content ?? delta.reasoning;
      const content = delta.content;

      if (explicitReasoning) {
        schema = "vllm_reasoning_fields";
        events.push({ channel: "reasoning", text: String(explicitReasoning), schema });
      }

      if (content) {
        if (schema === "vllm_reasoning_fields") {
          events.push({ channel: "answer", text: String(content), schema });
        } else {
          const inlineEvents = inline.push(String(content));
          if (inline.sawInlineReasoning() || inlineEvents.some((event) => event.channel === "reasoning")) {
            schema = "nim_inline_think";
          }
          events.push(...withSchema(inlineEvents));
        }
      }

      return {
        events,
        finishReason: choice.finish_reason || null,
        usage: data?.usage || null,
        schema
      };
    },
    finish() {
      const events = inline.finish();
      if (inline.sawInlineReasoning() || events.some((event) => event.channel === "reasoning")) {
        schema = "nim_inline_think";
      }
      return { events: withSchema(events), schema };
    },
    get schema() {
      return schema;
    }
  };
}

export async function submitReasoning(options = {}) {
  const { baseUrl, payload, redactedPayload } = buildReasoningPayload(options);

  let response;
  try {
    response = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.VLLM_API_KEY || process.env.NIM_API_KEY || "EMPTY"}`
      },
      body: JSON.stringify(payload)
    });
  } catch (error) {
    return {
      status: "error",
      message: error instanceof Error ? error.message : "Reasoner request failed",
      content: "",
      payload: redactedPayload
    };
  }

  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    return {
      status: "error",
      message:
        data?.error?.message ||
        data?.message ||
        `Reasoner returned HTTP ${response.status}`,
      content: "",
      payload: redactedPayload,
      raw: data
    };
  }

  const message = data?.choices?.[0]?.message || {};
  const contentText = message.content || "";
  const normalized = normalizeReasonerMessage(data);
  const mockBackend =
    data?.id === "chatcmpl-byo-mock" ||
    /mock cosmos reasoner response/i.test(contentText);
  if (mockBackend) {
    return {
      status: "error",
      message: "Mock backend is active on the configured Reasoner endpoint; start a real vLLM/NIM server on /v1.",
      content: "",
      reasoning: "",
      payload: redactedPayload,
      raw: data
    };
  }

  return {
    status: "success",
    message: data?.usage ? `prompt ${data.usage.prompt_tokens || 0} / completion ${data.usage.completion_tokens || 0}` : "",
    content: normalized.answer,
    reasoning: normalized.reasoning,
    combined_content: normalized.combined,
    schema: normalized.schema,
    openai: normalized.normalized,
    payload: redactedPayload,
    raw: data
  };
}

export default submitReasoning;
