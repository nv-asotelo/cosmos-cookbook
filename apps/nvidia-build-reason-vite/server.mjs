import express from "express";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const isProduction = process.env.NODE_ENV === "production";
const port = Number(process.env.PORT || 5173);
const baseUrl = process.env.VLLM_BASE_URL || "http://localhost:8000/v1";
const apiKey = process.env.VLLM_API_KEY || "EMPTY";
const defaultModel = process.env.MODEL_NAME || "nvidia/Cosmos3-Nano-Reasoner";

const app = express();
app.use(express.json({ limit: "128mb" }));

app.get("/api/models", async (_request, response) => {
  try {
    const upstream = await fetch(`${baseUrl.replace(/\/$/, "")}/models`, {
      headers: { Authorization: `Bearer ${apiKey}` }
    });
    if (!upstream.ok) throw new Error(`Model probe failed with HTTP ${upstream.status}`);
    const data = await upstream.json();
    const models = Array.isArray(data?.data) ? data.data.map((model) => model.id).filter(Boolean) : [];
    response.json({ baseUrl, models: models.length > 0 ? models : [defaultModel] });
  } catch (error) {
    response.json({
      baseUrl,
      models: [defaultModel],
      warning: error instanceof Error ? error.message : "Unable to reach model endpoint"
    });
  }
});

app.post("/api/reason", async (request, response) => {
  const body = request.body;

  if (!body.mediaDataUrl) {
    response.status(400).json({ error: "Upload media or load the sample before running." });
    return;
  }

  const mediaPart =
    body.mediaKind === "image"
      ? { type: "image_url", image_url: { url: body.mediaDataUrl } }
      : { type: "video_url", video_url: { url: body.mediaDataUrl } };

  const payload = {
    model: body.model || defaultModel,
    messages: [
      { role: "system", content: body.systemPrompt },
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

  try {
    const upstream = await fetch(`${baseUrl.replace(/\/$/, "")}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`
      },
      body: JSON.stringify(payload)
    });
    const data = await upstream.json().catch(() => null);
    if (!upstream.ok) {
      response.status(upstream.status).json({
        error: data?.error?.message || data?.message || `Backend returned HTTP ${upstream.status}`,
        payload
      });
      return;
    }
    response.json({ content: data?.choices?.[0]?.message?.content || "", raw: data, payload });
  } catch (error) {
    response.status(502).json({ error: error instanceof Error ? error.message : "Backend request failed", payload });
  }
});

if (isProduction) {
  app.use(express.static(path.join(__dirname, "dist")));
  app.get("*", (_request, response) => {
    response.sendFile(path.join(__dirname, "dist", "index.html"));
  });
} else {
  const { createServer } = await import("vite");
  const vite = await createServer({
    server: { middlewareMode: true },
    appType: "spa"
  });
  app.use(vite.middlewares);
}

app.listen(port, "0.0.0.0", () => {
  console.log(`[vite-build-reason] http://localhost:${port}`);
});
