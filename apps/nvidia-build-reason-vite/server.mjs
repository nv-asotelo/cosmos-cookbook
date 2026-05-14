import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { listReasonerModels, submitReasoning } from "../_shared/reasonerClient.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const isProduction = process.env.NODE_ENV === "production";
const port = Number(process.env.PORT || 5173);

const defaultModel = process.env.MODEL_NAME || process.env.MODEL_ID || "nvidia/Cosmos3-Nano-Reasoner";

const app = express();
app.use(express.json({ limit: "128mb" }));

const EXAMPLE_MEDIA_HOSTS = new Set(["assets.ngc.nvidia.com"]);

function mediaMime(name, fallback = "application/octet-stream") {
  const lower = name.toLowerCase();
  if (lower.endsWith(".mp4")) return "video/mp4";
  if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return "image/jpeg";
  if (lower.endsWith(".png")) return "image/png";
  return fallback.split(";")[0] || "application/octet-stream";
}

app.get("/api/models", async (_request, response) => {
  response.json(await listReasonerModels());
});

app.get("/api/active-model", async (_request, response) => {
  const info = await listReasonerModels();
  response.json({
    checkpoint: info.models?.[0] || defaultModel,
    display_name: info.models?.[0] || defaultModel,
    backend: "vllm",
    base_url: info.baseUrl,
    warning: info.warning
  });
});

app.get("/api/example-media", async (request, response) => {
  const rawUrl = String(request.query.url || "");
  const requestedName = String(request.query.name || "");

  try {
    const parsed = new URL(rawUrl);
    if (parsed.protocol !== "https:" || !EXAMPLE_MEDIA_HOSTS.has(parsed.hostname)) {
      response.status(400).json({ message: "Example media URL is not allowed" });
      return;
    }

    const name = requestedName || path.basename(parsed.pathname) || "example-media";
    const upstream = await fetch(parsed);
    if (!upstream.ok) {
      response.status(502).json({ message: `Example media request failed with ${upstream.status}` });
      return;
    }

    const fallbackMime = upstream.headers.get("content-type") || undefined;
    const mime = mediaMime(name, fallbackMime);
    const bytes = Buffer.from(await upstream.arrayBuffer());
    response.json({
      name,
      mime,
      dataUrl: `data:${mime};base64,${bytes.toString("base64")}`
    });
  } catch (error) {
    response.status(502).json({
      message: error instanceof Error ? error.message : "Example media request failed"
    });
  }
});

app.post("/api/reason", async (request, response) => {
  const body = request.body || {};
  const prompt = body.prompt || body.userPrompt || "";
  const systemPrompt = body.systemPrompt || body.system_prompt || "";

  let mediaDataUrl;
  let mediaKind = null;
  if (body.video) {
    mediaDataUrl = body.video;
    mediaKind = "video";
  } else if (body.image) {
    mediaDataUrl = body.image;
    mediaKind = "image";
  } else if (body.mediaDataUrl) {
    mediaDataUrl = body.mediaDataUrl;
    mediaKind = body.mediaKind || null;
  }

  try {
    const result = await submitReasoning({
      model: body.model || defaultModel,
      prompt,
      systemPrompt,
      mediaDataUrl,
      mediaKind,
      params: body.params || {}
    });
    const httpStatus = result.status === "error" ? 502 : 200;
    response.status(httpStatus).json(result);
  } catch (error) {
    response.status(502).json({
      status: "error",
      message: error instanceof Error ? error.message : "Backend request failed",
      files: []
    });
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
