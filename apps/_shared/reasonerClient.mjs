// Shared server-side client for vLLM/OpenAI-compatible Cosmos Reasoner APIs.
//
// Env var priority:
//   VLLM_BASE_URL > NIM_BASE_URL > REASONER_BASE_URL > http://localhost:8000/v1
//
// The BYO-video standing order is to send media as data URLs in the OpenAI
// multimodal content array. Do not use file:// or HTML-tag-in-text shapes.

function resolveBaseUrl() {
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
    process.env.VLLM_MODEL ||
    "nvidia/Cosmos3-Nano-Reasoner"
  );
}

function mediaContent(mediaDataUrl, mediaKind) {
  if (!mediaDataUrl) return [];
  if (mediaKind === "image") {
    return [{ type: "image_url", image_url: { url: mediaDataUrl } }];
  }
  return [{ type: "video_url", video_url: { url: mediaDataUrl } }];
}

function redactPayload(payload) {
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

export async function submitReasoning({
  model,
  prompt,
  systemPrompt,
  mediaDataUrl,
  mediaKind,
  params
} = {}) {
  const baseUrl = resolveBaseUrl();
  const p = params || {};
  const selectedModel = model || configuredModel();
  const content = [
    ...mediaContent(mediaDataUrl, mediaKind),
    { type: "text", text: prompt || "Describe the provided media." }
  ];

  const messages = [];
  if (systemPrompt) messages.push({ role: "system", content: systemPrompt });
  messages.push({ role: "user", content });

  const payload = {
    model: selectedModel,
    messages,
    temperature: Number.isFinite(Number(p.temperature)) ? Number(p.temperature) : 0.6,
    top_p: Number.isFinite(Number(p.top_p)) ? Number(p.top_p) : 0.9,
    max_tokens: Number.isFinite(Number(p.max_tokens)) ? Number(p.max_tokens) : 4096,
    seed: Number.isFinite(Number(p.seed)) ? Number(p.seed) : undefined,
    repetition_penalty: Number.isFinite(Number(p.repetition_penalty)) ? Number(p.repetition_penalty) : undefined,
    top_k: Number.isFinite(Number(p.top_k)) ? Number(p.top_k) : undefined,
    mm_processor_kwargs: Number.isFinite(Number(p.frames_per_second))
      ? { fps: Number(p.frames_per_second) }
      : undefined
  };

  Object.keys(payload).forEach((key) => payload[key] === undefined && delete payload[key]);

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
      payload: redactPayload(payload)
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
      payload: redactPayload(payload),
      raw: data
    };
  }

  const message = data?.choices?.[0]?.message || {};
  const contentText = message.content || "";
  const reasoningText = message.reasoning_content || message.reasoning || "";
  const mockBackend =
    data?.id === "chatcmpl-byo-mock" ||
    /mock cosmos reasoner response/i.test(contentText);
  if (mockBackend) {
    return {
      status: "error",
      message: "Mock backend is active on the configured Reasoner endpoint; start a real vLLM/NIM server on /v1.",
      content: "",
      reasoning: "",
      payload: redactPayload(payload),
      raw: data
    };
  }

  return {
    status: "success",
    message: data?.usage ? `prompt ${data.usage.prompt_tokens || 0} / completion ${data.usage.completion_tokens || 0}` : "",
    content: contentText,
    reasoning: reasoningText,
    payload: redactPayload(payload),
    raw: data
  };
}

export default submitReasoning;
