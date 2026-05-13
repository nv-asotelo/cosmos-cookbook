import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { submitGeneration } from "../_shared/cosmos3Client.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const isProduction = process.env.NODE_ENV === "production";
const port = Number(process.env.PORT || 5173);

// Env var priority: COSMOS3_BASE_URL > RAY_SERVE_BASE_URL > VLLM_BASE_URL > default.
const baseUrl =
  process.env.COSMOS3_BASE_URL ||
  process.env.RAY_SERVE_BASE_URL ||
  process.env.VLLM_BASE_URL ||
  "http://localhost:8000";
const defaultModel = process.env.MODEL_NAME || "Cosmos3-Nano";

const app = express();
app.use(express.json({ limit: "128mb" }));

app.get("/api/models", (_request, response) => {
  // Ray Serve does not expose an OpenAI /models listing — surface the
  // configured model name as the single available option.
  response.json({ baseUrl, models: [defaultModel] });
});

app.post("/api/reason", async (request, response) => {
  const body = request.body || {};
  const prompt = body.prompt || body.userPrompt || "";

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
    const result = await submitGeneration({
      prompt,
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
