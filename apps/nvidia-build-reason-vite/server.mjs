import express from "express";
import { execFile, spawn } from "node:child_process";
import { existsSync } from "node:fs";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  buildReasoningPayload,
  createReasoningStreamNormalizer,
  listReasonerModels,
  submitReasoning
} from "../_shared/reasonerClient.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const isProduction = process.env.NODE_ENV === "production";
const port = Number(process.env.PORT || 5173);

const defaultModel =
  process.env.MODEL_NAME ||
  process.env.MODEL_ID ||
  process.env.ALPAMAYO_MODEL_ID ||
  (process.env.INFERENCE_BACKEND === "alpamayo" ? "nvidia/Alpamayo-1.5-10B" : "nvidia/Cosmos3-Nano-Reasoner");
const backend =
  process.env.INFERENCE_BACKEND ||
  (process.env.ALPAMAYO_BASE_URL ? "alpamayo" : process.env.NIM_BASE_URL ? "nim_local" : "vllm");
const isAlpamayoBackend = String(backend || "").toLowerCase() === "alpamayo";
const KNOWN_HF_MODEL_COMMITS = {
  "nvidia/Cosmos3-Nano-Reasoner": "6406357cdc32fbf8db5f51ff7992343803b06961"
};

const app = express();
app.use(express.json({ limit: "128mb" }));

const EXAMPLE_MEDIA_HOSTS = new Set(["assets.ngc.nvidia.com"]);
const DEFAULT_NIM_FRAME_FALLBACK_IMAGES = 5;
const FRAME_EXTRACTOR_PY = String.raw`
import json
import os
import sys

import av

video_path, output_dir, fps_raw = sys.argv[1], sys.argv[2], sys.argv[3]
max_frames = int(float(sys.argv[4])) if len(sys.argv) > 4 and sys.argv[4] else 0
fps = float(fps_raw)
container = av.open(video_path)
stream = container.streams.video[0]
frame_rate = float(stream.average_rate) if stream.average_rate else 25.0
total_frames = int(stream.frames or 0)
if total_frames <= 0:
    duration_s = float(stream.duration * stream.time_base) if stream.duration else 0.0
    total_frames = max(1, int(duration_s * frame_rate))
interval = max(1, int(round(frame_rate / fps)))
target_indices = set(range(0, total_frames, interval))
if max_frames > 0 and len(target_indices) > max_frames:
    ordered = sorted(target_indices)
    if max_frames == 1:
        target_indices = {ordered[0]}
    else:
        target_indices = {
            ordered[int(round(i * (len(ordered) - 1) / (max_frames - 1)))]
            for i in range(max_frames)
        }
paths = []
for index, frame in enumerate(container.decode(stream)):
    if index in target_indices:
        image = frame.to_image().convert("RGB")
        frame_path = os.path.join(output_dir, f"frame_{len(paths):05d}.jpg")
        image.save(frame_path, format="JPEG", quality=80)
        paths.append(frame_path)
        if len(paths) >= len(target_indices):
            break
container.close()
print(json.dumps({"count": len(paths), "fps": fps, "frames": paths}))
`;

function dataUrlToBuffer(dataUrl) {
  const match = String(dataUrl || "").match(/^data:([^;,]+)?(;base64)?,(.*)$/s);
  if (!match) throw new Error("Media payload is not a data URL");
  const isBase64 = Boolean(match[2]);
  return {
    mime: match[1] || "application/octet-stream",
    buffer: isBase64 ? Buffer.from(match[3], "base64") : Buffer.from(decodeURIComponent(match[3]), "utf8")
  };
}

function pythonForFrames() {
  if (process.env.FRAME_EXTRACTOR_PYTHON) return process.env.FRAME_EXTRACTOR_PYTHON;
  if (process.env.PYTHON) return process.env.PYTHON;
  const hordePython = "/home/horde/cosmos-reason2/.venv/bin/python3";
  return existsSync(hordePython) ? hordePython : "python3";
}

function usesNativeVideoUrl(model) {
  const mid = String(model || "").toLowerCase();
  const mode = String(process.env.REASONER_MEDIA_MODE || "auto").toLowerCase();
  if (mode === "video_url") return true;
  if (mode === "frames") return false;
  if (isAlpamayoBackend) return true;
  if (backend === "nim_local" && !mid.includes("cosmos-reason1")) return true;
  return (
    mid.includes("qwen3-vl") ||
    mid.includes("qwen3vl") ||
    mid.includes("qwen") ||
    mid.includes("cosmos3") ||
    mid.includes("cosmos-3") ||
    mid.includes("cosmos-reason2") ||
    mid.includes("cosmos-reason-2") ||
    mid.includes("alpamayo") ||
    mid.includes("nemotron")
  );
}

function positiveIntegerEnv(name, fallback) {
  const parsed = Number.parseInt(process.env[name] || "", 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function isVlaModel(model) {
  const mid = String(model || "").toLowerCase();
  return isAlpamayoBackend || mid.includes("alpamayo") || mid.includes("reasoningvla");
}

function vlaRuntimeInfo(selectedModel, modelInfo) {
  if (!isVlaModel(selectedModel) && !isVlaModel(modelInfo?.id)) return null;
  const framesPerVideo = positiveIntegerEnv("ALPAMAYO_FRAMES", 4);
  const maxDecodedVideoFrames = positiveIntegerEnv("ALPAMAYO_MAX_DECODED_VIDEO_FRAMES", 96);
  return {
    type: "VLA",
    family: "Alpamayo",
    adapter: "alpamayo_openai",
    current_mode: "vqa_text_generation",
    model_id: selectedModel || modelInfo?.id || defaultModel,
    backbone: "Cosmos Reason2-8B VLM",
    input_shape: "OpenAI chat/completions with video_url or image_url plus text prompt",
    frames_per_video: framesPerVideo,
    max_decoded_video_frames: maxDecodedVideoFrames,
    prompt_role: "VQA/caption question over sampled frames",
    full_trajectory_mode: false,
    clip_guidance: `Use short, front-loaded clips. The adapter samples ${framesPerVideo} frame(s) from the first ${maxDecodedVideoFrames} decoded video frame(s) unless ALPAMAYO_FRAMES or ALPAMAYO_MAX_DECODED_VIDEO_FRAMES is changed.`
  };
}

function frameFallbackLimit() {
  const raw = process.env.REASONER_FRAME_FALLBACK_MAX_IMAGES || process.env.NIM_MAX_IMAGES_PER_PROMPT || "";
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_NIM_FRAME_FALLBACK_IMAGES;
}

function nativeVideoFallbackMessage(message) {
  return /video resolution or format not supported|cuvid|getdecodercaps|reconfiguredecoder|handlevideosequence|pynvvideocodec|at most \d+ image/i.test(
    String(message || "")
  );
}

function shouldRetryWithFrameFallback(prepared, resultOrError) {
  const message =
    resultOrError?.message ||
    resultOrError?.raw?.error?.message ||
    (resultOrError instanceof Error ? resultOrError.message : "");
  return prepared?.media?.mode === "video_url" && backend === "nim_local" && nativeVideoFallbackMessage(message);
}

function runFrameExtractor(videoPath, outputDir, framesPerSecond, maxFrames) {
  const fps = Number.isFinite(Number(framesPerSecond)) ? Number(framesPerSecond) : 2;
  const cap = Number.isFinite(Number(maxFrames)) && Number(maxFrames) > 0 ? String(Math.floor(Number(maxFrames))) : "";
  return new Promise((resolve, reject) => {
    const child = spawn(pythonForFrames(), ["-", videoPath, outputDir, String(fps), cap], {
      stdio: ["pipe", "pipe", "pipe"]
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr || `Frame extractor exited with ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        reject(error);
      }
    });
    child.stdin.end(FRAME_EXTRACTOR_PY);
  });
}

async function extractFrameDataUrls(mediaDataUrl, framesPerSecond, maxFrames) {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "reason-vite-frames-"));
  try {
    const { buffer } = dataUrlToBuffer(mediaDataUrl);
    const videoPath = path.join(tempDir, "input.mp4");
    const frameDir = path.join(tempDir, "frames");
    await fs.mkdir(frameDir);
    await fs.writeFile(videoPath, buffer);
    const info = await runFrameExtractor(videoPath, frameDir, framesPerSecond, maxFrames);
    const frames = [];
    for (const framePath of info.frames || []) {
      const frame = await fs.readFile(framePath);
      frames.push(`data:image/jpeg;base64,${frame.toString("base64")}`);
    }
    if (frames.length === 0) throw new Error("Could not extract frames from video");
    return { tempDir, frames, frameCount: frames.length, fps: info.fps || framesPerSecond };
  } catch (error) {
    await fs.rm(tempDir, { recursive: true, force: true });
    throw error;
  }
}

function mediaMime(name, fallback = "application/octet-stream") {
  const lower = name.toLowerCase();
  if (lower.endsWith(".mp4")) return "video/mp4";
  if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return "image/jpeg";
  if (lower.endsWith(".png")) return "image/png";
  return fallback.split(";")[0] || "application/octet-stream";
}

function commandValue(command, flag) {
  const index = command.indexOf(flag);
  if (index === -1) return null;
  const rest = command.slice(index + flag.length).trimStart();
  const nextFlag = rest.search(/\s--[A-Za-z0-9-]+/);
  return (nextFlag === -1 ? rest : rest.slice(0, nextFlag)).trim() || null;
}

function splitCommandWords(command) {
  const words = [];
  const pattern = /"((?:\\.|[^"\\])*)"|'([^']*)'|(\S+)/g;
  let match;
  while ((match = pattern.exec(command))) {
    words.push(match[1] || match[2] || match[3]);
  }
  return words;
}

function vllmServeModel(command) {
  const words = splitCommandWords(command);
  const serveIndex = words.findIndex((word) => word === "serve");
  if (serveIndex === -1) return null;
  for (let index = serveIndex + 1; index < words.length; index += 1) {
    const word = words[index];
    if (!word || word.startsWith("-")) {
      if (word?.startsWith("--") && index + 1 < words.length && !words[index + 1].startsWith("-")) index += 1;
      continue;
    }
    return word;
  }
  return null;
}

function inferQuantization(model, flags = {}) {
  const mid = String(model || "").toLowerCase();
  const explicit = flags.quantization || process.env.MODEL_QUANTIZATION || process.env.VLLM_QUANTIZATION || "";
  let inferred = "";
  if (mid.includes("nvfp4") || mid.includes("fp4")) inferred = "nvfp4/fp4";
  else if (mid.includes("fp8")) inferred = "fp8";
  else if (mid.includes("int8")) inferred = "int8";
  else if (mid.includes("awq")) inferred = "awq";
  else if (mid.includes("gptq")) inferred = "gptq";

  const method = explicit || inferred;
  return {
    applied: Boolean(method),
    method: method || null,
    dtype: flags.dtype || process.env.MODEL_DTYPE || null,
    source: explicit
      ? "vLLM launch flag"
      : inferred
        ? "model identifier"
        : "no --quantization flag or quantized model suffix detected"
  };
}

function parseVllmCommand(command) {
  const flags = {
    model: commandValue(command, "--model") || vllmServeModel(command),
    revision: commandValue(command, "--revision"),
    served_model_name: commandValue(command, "--served-model-name"),
    max_model_len: commandValue(command, "--max-model-len"),
    media_io_kwargs: commandValue(command, "--media-io-kwargs"),
    reasoning_parser: commandValue(command, "--reasoning-parser"),
    dtype: commandValue(command, "--dtype"),
    quantization: commandValue(command, "--quantization"),
    gpu_memory_utilization: commandValue(command, "--gpu-memory-utilization"),
    host: commandValue(command, "--host"),
    port: commandValue(command, "--port"),
    allowed_local_media_path: commandValue(command, "--allowed-local-media-path"),
    trust_remote_code: command.includes("--trust-remote-code")
  };
  Object.keys(flags).forEach((key) => flags[key] === null && delete flags[key]);
  return flags;
}

function execFileText(command, args) {
  return new Promise((resolve, reject) => {
    execFile(command, args, { timeout: 3000, maxBuffer: 1024 * 1024 }, (error, stdout) => {
      if (error) {
        reject(error);
        return;
      }
      resolve(stdout);
    });
  });
}

async function findVllmProcess(selectedModel) {
  try {
    const stdout = await execFileText("ps", ["-eo", "pid=,cmd="]);
    const lines = stdout
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.includes("vllm serve"));
    const modelNeedle = String(selectedModel || "").toLowerCase();
    const line =
      lines.find((item) => item.toLowerCase().includes(modelNeedle)) ||
      lines.find((item) => item.includes("--port 8000")) ||
      lines[0];
    if (!line) return null;
    const match = line.match(/^(\d+)\s+(.*)$/);
    if (!match) return null;
    return { pid: Number(match[1]), command: match[2], flags: parseVllmCommand(match[2]) };
  } catch {
    return null;
  }
}

async function fetchBackendModelInfo(baseUrl, selectedModel) {
  try {
    const response = await fetch(`${baseUrl}/models`, {
      headers: { Authorization: `Bearer ${process.env.VLLM_API_KEY || process.env.NIM_API_KEY || "EMPTY"}` }
    });
    if (!response.ok) return null;
    const data = await response.json();
    const models = Array.isArray(data?.data) ? data.data : [];
    return models.find((item) => item.id === selectedModel) || models[0] || null;
  } catch {
    return null;
  }
}

function appSourceInfo() {
  return {
    sha: process.env.APP_GIT_SHA || null,
    timestamp: process.env.APP_GIT_TIMESTAMP || null,
    branch: process.env.APP_GIT_BRANCH || null,
    dirty: process.env.APP_GIT_DIRTY === "1" ? true : process.env.APP_GIT_DIRTY === "0" ? false : null
  };
}

function huggingFaceToken() {
  return process.env.HF_TOKEN || process.env.HUGGINGFACE_HUB_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN || null;
}

function huggingFaceCacheRoot() {
  if (process.env.HUGGINGFACE_HUB_CACHE) return process.env.HUGGINGFACE_HUB_CACHE;
  if (process.env.TRANSFORMERS_CACHE) return process.env.TRANSFORMERS_CACHE;
  const hfHome =
    process.env.HF_HOME ||
    (process.env.XDG_CACHE_HOME ? path.join(process.env.XDG_CACHE_HOME, "huggingface") : path.join(os.homedir(), ".cache", "huggingface"));
  return path.join(hfHome, "hub");
}

function normalizeHfRepoId(value) {
  const raw = String(value || "").trim();
  if (!raw || raw.startsWith("/") || raw.startsWith(".") || raw.includes("\\") || raw.includes("://")) return null;
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/.test(raw)) return null;
  return raw;
}

function hfRepoCacheDir(repoId) {
  return path.join(huggingFaceCacheRoot(), `models--${repoId.replace(/\//g, "--")}`);
}

function hfCommitUrl(repoId, sha) {
  if (!repoId || !sha) return null;
  return `https://huggingface.co/${repoId}/commit/${sha}`;
}

async function readTextIfExists(filePath) {
  try {
    return (await fs.readFile(filePath, "utf8")).trim() || null;
  } catch {
    return null;
  }
}

async function statIfExists(filePath) {
  try {
    return await fs.stat(filePath);
  } catch {
    return null;
  }
}

async function snapshotShaFromPath(modelPath) {
  if (!modelPath) return null;
  const candidates = [modelPath];
  try {
    candidates.push(await fs.realpath(modelPath));
  } catch {
    // The backend can expose repo ids or stale paths; missing paths just fall through.
  }
  for (const candidate of candidates) {
    const match = String(candidate).match(/(?:^|\/)snapshots\/([a-f0-9]{40})(?:\/|$)/i);
    if (match) return { sha: match[1], cache_path: candidate };
  }
  return null;
}

async function newestSnapshotSha(repoCacheDir) {
  try {
    const snapshotDir = path.join(repoCacheDir, "snapshots");
    const entries = await fs.readdir(snapshotDir, { withFileTypes: true });
    const snapshots = [];
    for (const entry of entries) {
      if (!entry.isDirectory() || !/^[a-f0-9]{40}$/i.test(entry.name)) continue;
      const snapshotPath = path.join(snapshotDir, entry.name);
      const stats = await statIfExists(snapshotPath);
      snapshots.push({ sha: entry.name, path: snapshotPath, mtimeMs: stats?.mtimeMs || 0 });
    }
    snapshots.sort((left, right) => right.mtimeMs - left.mtimeMs);
    return snapshots[0] || null;
  } catch {
    return null;
  }
}

async function localHfModelSource({ modelPath, repoId, revision }) {
  const pathSnapshot = await snapshotShaFromPath(modelPath);
  if (pathSnapshot) {
    return {
      sha: pathSnapshot.sha,
      timestamp: null,
      branch: revision,
      dirty: null,
      repo_id: repoId,
      revision,
      cache_path: pathSnapshot.cache_path,
      commit_url: hfCommitUrl(repoId, pathSnapshot.sha),
      source: "hf_snapshot_path"
    };
  }

  if (!repoId) return null;
  const repoCacheDir = hfRepoCacheDir(repoId);
  const refSha = await readTextIfExists(path.join(repoCacheDir, "refs", revision || "main"));
  if (refSha) {
    const snapshotPath = path.join(repoCacheDir, "snapshots", refSha);
    return {
      sha: refSha,
      timestamp: null,
      branch: revision || "main",
      dirty: null,
      repo_id: repoId,
      revision: revision || "main",
      cache_path: (await statIfExists(snapshotPath)) ? snapshotPath : repoCacheDir,
      commit_url: hfCommitUrl(repoId, refSha),
      source: "hf_cache_ref"
    };
  }

  const newest = await newestSnapshotSha(repoCacheDir);
  if (!newest) return null;
  return {
    sha: newest.sha,
    timestamp: null,
    branch: revision || "main",
    dirty: null,
    repo_id: repoId,
    revision: revision || "main",
    cache_path: newest.path,
    commit_url: hfCommitUrl(repoId, newest.sha),
    source: "hf_cache_snapshot"
  };
}

async function remoteHfModelSource(repoId, revision) {
  if (!repoId) return null;
  const suffix =
    revision && revision !== "main"
      ? `/revision/${encodeURIComponent(revision)}`
      : "";
  const url = `https://huggingface.co/api/models/${repoId.split("/").map(encodeURIComponent).join("/")}${suffix}`;
  const token = huggingFaceToken();
  try {
    const response = await fetch(url, {
      headers: token ? { Authorization: `Bearer ${token}` } : {}
    });
    if (!response.ok) return null;
    const data = await response.json();
    const sha = data?.sha || data?._commit_hash || null;
    if (!sha) return null;
    return {
      sha,
      timestamp: data?.lastModified || null,
      branch: revision || "main",
      dirty: null,
      repo_id: repoId,
      revision: revision || "main",
      cache_path: null,
      commit_url: hfCommitUrl(repoId, sha),
      source: "hf_api"
    };
  } catch {
    return null;
  }
}

function knownHfModelSource(repoId, revision) {
  const sha = repoId ? KNOWN_HF_MODEL_COMMITS[repoId] : null;
  if (!sha) return null;
  return {
    sha,
    timestamp: null,
    branch: revision || "main",
    dirty: null,
    repo_id: repoId,
    revision: revision || "main",
    cache_path: null,
    commit_url: hfCommitUrl(repoId, sha),
    source: "known_hf_commit_hint"
  };
}

async function resolveModelSource({ selectedModel, vllmProcess, modelInfo }) {
  const flags = vllmProcess?.flags || {};
  const modelPath =
    process.env.HF_MODEL_PATH ||
    process.env.MODEL_PATH ||
    flags.model ||
    modelInfo?.root ||
    modelInfo?.id ||
    selectedModel;
  const repoId =
    normalizeHfRepoId(process.env.HF_MODEL_ID || process.env.MODEL_REPO_ID) ||
    normalizeHfRepoId(flags.model) ||
    normalizeHfRepoId(modelInfo?.id) ||
    normalizeHfRepoId(selectedModel);
  const revision = process.env.HF_MODEL_REVISION || process.env.MODEL_REVISION || flags.revision || "main";
  const envSha = process.env.HF_MODEL_SHA || process.env.MODEL_HF_SHA || process.env.MODEL_COMMIT_SHA || null;

  if (envSha) {
    return {
      sha: envSha,
      timestamp: process.env.HF_MODEL_TIMESTAMP || null,
      branch: revision,
      dirty: null,
      repo_id: repoId,
      revision,
      cache_path: modelPath || null,
      commit_url: hfCommitUrl(repoId, envSha),
      source: "environment"
    };
  }

  const local = await localHfModelSource({ modelPath, repoId, revision });
  const remote = await remoteHfModelSource(repoId, revision);
  if (local && remote && local.sha === remote.sha) {
    return { ...local, timestamp: remote.timestamp, source: `${local.source}+hf_api` };
  }
  if (local) return local;
  if (remote) return remote;
  const known = knownHfModelSource(repoId, revision);
  if (known) return known;
  return {
    sha: null,
    timestamp: null,
    branch: revision,
    dirty: null,
    repo_id: repoId,
    revision,
    cache_path: modelPath || null,
    commit_url: null,
    source: "unresolved"
  };
}

async function buildRuntimeInfo(info) {
  const selectedModel = info.models?.[0] || defaultModel;
  const vllmProcess = await findVllmProcess(selectedModel);
  const modelInfo = await fetchBackendModelInfo(info.baseUrl, selectedModel);
  const flags = vllmProcess?.flags || {};
  const quantization = inferQuantization(selectedModel, flags);
  const source = await resolveModelSource({ selectedModel, vllmProcess, modelInfo });
  return {
    source,
    app_source: appSourceInfo(),
    quantization,
    vla: vlaRuntimeInfo(selectedModel, modelInfo),
    vllm: {
      base_url: info.baseUrl,
      model: modelInfo,
      process: vllmProcess
        ? {
            pid: vllmProcess.pid,
            command: vllmProcess.command,
            flags
          }
        : null
    }
  };
}

app.get("/api/models", async (_request, response) => {
  response.json(await listReasonerModels());
});

app.get("/api/active-model", async (_request, response) => {
  const info = await listReasonerModels();
  const runtime = await buildRuntimeInfo(info);
  response.json({
    checkpoint: info.models?.[0] || defaultModel,
    display_name: info.models?.[0] || defaultModel,
    backend,
    base_url: info.baseUrl,
    source: runtime.source,
    app_source: runtime.app_source,
    quantization: runtime.quantization,
    vla: runtime.vla,
    vllm: runtime.vllm,
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

async function prepareReasonRequest(body = {}, options = {}) {
  const prompt = body.prompt || body.userPrompt || "";
  const systemPrompt = body.systemPrompt || body.system_prompt || "";
  const selectedModel = body.model || defaultModel;
  const params = body.params || {};
  const forceFrameFallback = Boolean(options.forceFrameFallback);

  let mediaDataUrl;
  let mediaKind = null;
  let mediaFrames;
  let frameTempDir;
  let mediaMode = "none";
  let frameCount = 0;
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

  if (mediaKind === "video" && mediaDataUrl && (forceFrameFallback || !usesNativeVideoUrl(selectedModel))) {
    const maxFrames = backend === "nim_local" || forceFrameFallback ? frameFallbackLimit() : undefined;
    const extracted = await extractFrameDataUrls(mediaDataUrl, params.frames_per_second, maxFrames);
    mediaFrames = extracted.frames;
    frameTempDir = extracted.tempDir;
    frameCount = extracted.frameCount;
    mediaMode = "image-frame-fallback";
    mediaDataUrl = undefined;
  } else if (mediaKind === "video" && mediaDataUrl) {
    mediaMode = "video_url";
  } else if (mediaKind === "image" && mediaDataUrl) {
    mediaMode = "image_url";
  }

  return {
    prompt,
    systemPrompt,
    selectedModel,
    params,
    mediaDataUrl,
    mediaKind,
    mediaFrames,
    frameTempDir,
    media: {
      mode: mediaMode,
      frame_count: frameCount || undefined,
      fps: mediaFrames?.length ? params.frames_per_second : undefined,
      fallback_from: forceFrameFallback ? "video_url" : undefined,
      fallback_error: forceFrameFallback ? options.fallbackError : undefined,
      max_frames: forceFrameFallback || backend === "nim_local" ? frameFallbackLimit() : undefined
    }
  };
}

function preparedReasoningOptions(prepared) {
  return {
    model: prepared.selectedModel,
    prompt: prepared.prompt,
    systemPrompt: prepared.systemPrompt,
    mediaDataUrl: prepared.mediaDataUrl,
    mediaKind: prepared.mediaKind,
    mediaFrames: prepared.mediaFrames,
    framesPerSecond: prepared.params.frames_per_second,
    params: prepared.params
  };
}

async function submitPreparedReasoning(prepared) {
  const result = await submitReasoning(preparedReasoningOptions(prepared));
  result.media = prepared.media;
  return result;
}

function sse(response, event, data) {
  if (response.writableEnded) return;
  response.write(`event: ${event}\n`);
  response.write(`data: ${JSON.stringify(data)}\n\n`);
  response.flush?.();
}

function readSseBlocks(buffer) {
  const blocks = buffer.split(/\r?\n\r?\n/);
  return { complete: blocks.slice(0, -1), rest: blocks.at(-1) || "" };
}

function sseData(block) {
  return block
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart())
    .join("\n")
    .trim();
}

function combinedContent(reasoning, answer) {
  return reasoning ? `<think>\n${reasoning.trim()}\n</think>\n\n${answer || ""}`.trim() : answer || "";
}

function streamingOpenAiResult({ answer, created, finishReason, id, model, object, reasoning, schema, usage }) {
  return {
    status: "success",
    message: usage ? `prompt ${usage.prompt_tokens || 0} / completion ${usage.completion_tokens || 0}` : "",
    content: answer,
    reasoning,
    combined_content: combinedContent(reasoning, answer),
    schema,
    openai: {
      id: id || `chatcmpl-byo-stream-${Date.now()}`,
      object: object || "chat.completion",
      created: created || Math.floor(Date.now() / 1000),
      model: model || defaultModel,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: combinedContent(reasoning, answer)
          },
          finish_reason: finishReason || "stop"
        }
      ],
      usage: usage || null
    }
  };
}

async function readOpenAiStream({ baseUrl, payload, signal, onChunk }) {
  const streamPayload = { ...payload, stream: true };
  const upstream = await fetch(`${baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.VLLM_API_KEY || process.env.NIM_API_KEY || "EMPTY"}`
    },
    body: JSON.stringify(streamPayload),
    signal
  });

  if (!upstream.ok) {
    let detail = "";
    try {
      const data = await upstream.json();
      detail = data?.error?.message || data?.message || "";
    } catch {
      detail = await upstream.text().catch(() => "");
    }
    throw new Error(detail || `Reasoner stream returned HTTP ${upstream.status}`);
  }

  if (!upstream.body) throw new Error("Reasoner stream did not include a response body");

  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let sawSse = false;

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const blocks = readSseBlocks(buffer);
    buffer = blocks.rest;

    for (const block of blocks.complete) {
      const data = sseData(block);
      if (!data) continue;
      sawSse = true;
      if (data === "[DONE]") return;
      onChunk(JSON.parse(data));
    }

    if (done) break;
  }

  if (!sawSse) throw new Error("Reasoner response was not an OpenAI streaming SSE response");
}

function emitFallback(response, result) {
  if (result.status === "error" || result.error) {
    sse(response, "error", { message: result.message || result.error || "Backend request failed" });
    return;
  }
  if (result.reasoning) {
    sse(response, "state", { phase: "reasoning" });
    sse(response, "delta", { channel: "reasoning", text: result.reasoning, schema: result.schema || "plain_content" });
  }
  if (result.content) {
    sse(response, "state", { phase: "answer" });
    sse(response, "delta", { channel: "answer", text: result.content, schema: result.schema || "plain_content" });
  }
  if (result.raw?.usage || result.openai?.usage) sse(response, "usage", result.raw?.usage || result.openai?.usage);
  sse(response, "raw", result);
  sse(response, "state", { phase: "complete" });
}

app.post("/api/reason", async (request, response) => {
  let prepared;
  let fallbackPrepared;
  try {
    prepared = await prepareReasonRequest(request.body || {});
    let result = await submitPreparedReasoning(prepared);
    if (result.status === "error" && shouldRetryWithFrameFallback(prepared, result)) {
      console.warn(
        `[vite-build-reason] native video_url rejected by NIM; retrying with ${frameFallbackLimit()} image frames`
      );
      fallbackPrepared = await prepareReasonRequest(request.body || {}, {
        forceFrameFallback: true,
        fallbackError: result.message
      });
      result = await submitPreparedReasoning(fallbackPrepared);
    }
    const httpStatus = result.status === "error" ? 502 : 200;
    response.status(httpStatus).json(result);
  } catch (error) {
    response.status(502).json({
      status: "error",
      message: error instanceof Error ? error.message : "Backend request failed",
      files: []
    });
  } finally {
    if (prepared?.frameTempDir) await fs.rm(prepared.frameTempDir, { recursive: true, force: true });
    if (fallbackPrepared?.frameTempDir) await fs.rm(fallbackPrepared.frameTempDir, { recursive: true, force: true });
  }
});

app.post("/api/reason/stream", async (request, response) => {
  let prepared;
  let fallbackPrepared;
  const upstreamAbort = new AbortController();
  let clientClosed = false;
  let startedStreaming = false;

  const abortForClosedClient = () => {
    if (response.writableEnded) return;
    clientClosed = true;
    upstreamAbort.abort();
  };
  request.on("aborted", abortForClosedClient);
  response.on("close", abortForClosedClient);

  response.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no"
  });
  response.flushHeaders?.();
  sse(response, "state", { phase: "waiting_first_token" });

  try {
    prepared = await prepareReasonRequest(request.body || {});
    const { baseUrl, payload, redactedPayload } = buildReasoningPayload({
      model: prepared.selectedModel,
      prompt: prepared.prompt,
      systemPrompt: prepared.systemPrompt,
      mediaDataUrl: prepared.mediaDataUrl,
      mediaKind: prepared.mediaKind,
      mediaFrames: prepared.mediaFrames,
      framesPerSecond: prepared.params.frames_per_second,
      params: prepared.params
    });
    const normalizer = createReasoningStreamNormalizer();
    let answer = "";
    let reasoning = "";
    let schema = "plain_content";
    let usage = null;
    let finishReason = null;
    let meta = {};

    await readOpenAiStream({
      baseUrl,
      payload,
      signal: upstreamAbort.signal,
      onChunk(data) {
        meta = {
          id: data?.id || meta.id,
          object: data?.object || meta.object,
          created: data?.created || meta.created,
          model: data?.model || meta.model
        };
        const normalized = normalizer.accept(data);
        schema = normalized.schema || schema;
        usage = normalized.usage || usage;
        finishReason = normalized.finishReason || finishReason;
        for (const event of normalized.events) {
          if (!event.text) continue;
          startedStreaming = true;
          if (event.channel === "reasoning") {
            reasoning += event.text;
            sse(response, "state", { phase: "reasoning" });
          } else {
            answer += event.text;
            sse(response, "state", { phase: "answer" });
          }
          sse(response, "delta", event);
        }
      }
    });

    const finished = normalizer.finish();
    schema = finished.schema || schema;
    for (const event of finished.events) {
      if (!event.text) continue;
      startedStreaming = true;
      if (event.channel === "reasoning") {
        reasoning += event.text;
        sse(response, "state", { phase: "reasoning" });
      } else {
        answer += event.text;
        sse(response, "state", { phase: "answer" });
      }
      sse(response, "delta", event);
    }

    if (usage) sse(response, "usage", usage);
    sse(
      response,
      "raw",
      {
        ...streamingOpenAiResult({
          answer,
          created: meta.created,
          finishReason,
          id: meta.id,
          model: meta.model || prepared.selectedModel,
          object: meta.object,
          reasoning,
          schema,
          usage
        }),
        media: prepared.media,
        payload: redactedPayload
      }
    );
    sse(response, "state", { phase: "complete" });
  } catch (error) {
    if (clientClosed || response.writableEnded) return;
    if (!startedStreaming && prepared) {
      if (shouldRetryWithFrameFallback(prepared, error)) {
        console.warn(
          `[vite-build-reason] native video_url stream rejected by NIM; retrying with ${frameFallbackLimit()} image frames`
        );
        fallbackPrepared = await prepareReasonRequest(request.body || {}, {
          forceFrameFallback: true,
          fallbackError: error instanceof Error ? error.message : "Native video_url rejected"
        });
      }
      const fallback = await submitPreparedReasoning(fallbackPrepared || prepared);
      emitFallback(response, fallback);
    } else {
      sse(response, "error", { message: error instanceof Error ? error.message : "Backend stream failed" });
    }
  } finally {
    if (prepared?.frameTempDir) await fs.rm(prepared.frameTempDir, { recursive: true, force: true });
    if (fallbackPrepared?.frameTempDir) await fs.rm(fallbackPrepared.frameTempDir, { recursive: true, force: true });
    if (!response.writableEnded) response.end();
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
