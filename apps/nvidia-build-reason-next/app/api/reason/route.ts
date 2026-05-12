const DEFAULT_BASE_URL = "http://localhost:8000/v1";
const DEFAULT_MODEL = "nvidia/Cosmos3-Nano-Reasoner";

type ReasonRequest = {
  mediaDataUrl?: string;
  mediaKind?: "video" | "image";
  userPrompt: string;
  systemPrompt: string;
  model?: string;
  temperature: number;
  topP: number;
  maxTokens: number;
  repetitionPenalty: number;
  seed: number;
  framesPerSecond: number;
};

export async function POST(request: Request) {
  const body = (await request.json()) as ReasonRequest;
  const baseUrl = process.env.VLLM_BASE_URL ?? DEFAULT_BASE_URL;
  const apiKey = process.env.VLLM_API_KEY ?? "EMPTY";
  const model = body.model || process.env.MODEL_NAME || DEFAULT_MODEL;

  if (!body.mediaDataUrl) {
    return Response.json({ error: "Upload media or load the sample before running." }, { status: 400 });
  }

  const mediaPart =
    body.mediaKind === "image"
      ? { type: "image_url", image_url: { url: body.mediaDataUrl } }
      : { type: "video_url", video_url: { url: body.mediaDataUrl } };

  const payload = {
    model,
    messages: [
      {
        role: "system",
        content: body.systemPrompt
      },
      {
        role: "user",
        content: [mediaPart, { type: "text", text: body.userPrompt }]
      }
    ],
    media_io_kwargs: body.mediaKind === "video" ? { video: { fps: body.framesPerSecond } } : undefined,
    temperature: body.temperature,
    top_p: body.topP,
    max_tokens: body.maxTokens,
    repetition_penalty: body.repetitionPenalty,
    seed: body.seed,
    stream: false
  };

  const response = await fetch(`${baseUrl.replace(/\/$/, "")}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`
    },
    body: JSON.stringify(payload)
  });

  const data = await response.json().catch(() => null);

  if (!response.ok) {
    return Response.json(
      {
        error: data?.error?.message ?? data?.message ?? `Backend returned HTTP ${response.status}`,
        payload
      },
      { status: response.status }
    );
  }

  return Response.json({
    content: data?.choices?.[0]?.message?.content ?? "",
    raw: data,
    payload
  });
}
