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
  normalizeReasonerMessage,
  submitReasoning
} from "../_shared/reasonerClient.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const isProduction = process.env.NODE_ENV === "production";
const port = Number(process.env.PORT || 5173);

const backend =
  process.env.INFERENCE_BACKEND ||
  (process.env.ALPAMAYO_BASE_URL ? "alpamayo" : process.env.NIM_BASE_URL ? "nim_local" : "vllm");
const isAlpamayoBackend = String(backend || "").toLowerCase() === "alpamayo";
const defaultModel =
  process.env.NIM_SERVED_MODEL_NAME ||
  process.env.MODEL_NAME ||
  process.env.MODEL_ID ||
  process.env.ALPAMAYO_MODEL_ID ||
  (isAlpamayoBackend ? "nvidia/Alpamayo-1.5-10B" : "Detecting model...");
const KNOWN_HF_MODEL_COMMITS = {
  "nvidia/Cosmos3-Nano-Reasoner": "6406357cdc32fbf8db5f51ff7992343803b06961"
};

const app = express();
app.use(express.json({ limit: "512mb" }));

const EXAMPLE_MEDIA_HOSTS = new Set(["assets.ngc.nvidia.com"]);
const DEFAULT_NIM_FRAME_FALLBACK_IMAGES = 5;
const LONG_VIDEO_MAX_IMAGES_PER_CHUNK = 5;
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
const LONG_FRAME_EXTRACTOR_PY = String.raw`
import json
import math
import os
import sys

import av

video_path, output_dir, fps_raw, max_raw, width_raw = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
fps = max(0.1, float(fps_raw))
max_frames = int(float(max_raw)) if max_raw else 0
target_width = int(float(width_raw)) if width_raw else 768

def emit_progress(phase, **payload):
    payload["phase"] = phase
    print("PROGRESS " + json.dumps(payload), file=sys.stderr, flush=True)

def fmt(seconds):
    seconds = max(0.0, float(seconds or 0.0))
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes:02d}:{rem:05.2f}"

container = av.open(video_path)
stream = container.streams.video[0]
frame_rate = float(stream.average_rate) if stream.average_rate else 25.0
duration_s = 0.0
if container.duration:
    duration_s = float(container.duration / av.time_base)
elif stream.duration:
    duration_s = float(stream.duration * stream.time_base)
elif stream.frames and frame_rate:
    duration_s = float(stream.frames) / frame_rate

target_count = max(1, int(math.ceil(duration_s * fps))) if duration_s > 0 else 1
target_times = [index / fps for index in range(target_count)]
if duration_s > 0 and (not target_times or target_times[-1] < duration_s - (0.5 / fps)):
    target_times.append(duration_s)
target_times = [min(duration_s, value) if duration_s > 0 else value for value in target_times]
if max_frames > 0 and len(target_times) > max_frames:
    if max_frames == 1:
        target_times = [target_times[0]]
    else:
        target_times = [
            target_times[int(round(index * (len(target_times) - 1) / (max_frames - 1)))]
            for index in range(max_frames)
        ]

emit_progress(
    "opened",
    duration_s=duration_s,
    source_fps=frame_rate,
    source_width=getattr(stream, "width", 0) or 0,
    source_height=getattr(stream, "height", 0) or 0,
    target_frames=len(target_times),
)

frames = []
next_target = 0
source_width = 0
source_height = 0
last_ts = 0.0
progress_stride = max(1, len(target_times) // 20)
for index, frame in enumerate(container.decode(stream)):
    if next_target >= len(target_times):
        break
    ts = float(frame.pts * frame.time_base) if frame.pts is not None else (float(index) / frame_rate if frame_rate else 0.0)
    last_ts = ts
    if ts + (0.5 / max(frame_rate, 1.0)) < target_times[next_target]:
        continue
    image = frame.to_image().convert("RGB")
    source_width, source_height = image.size
    if target_width > 0 and image.width > target_width:
        ratio = target_width / float(image.width)
        image = image.resize((target_width, max(1, int(round(image.height * ratio)))))
    frame_path = os.path.join(output_dir, f"frame_{len(frames):05d}.jpg")
    image.save(frame_path, format="JPEG", quality=78, optimize=True)
    frames.append({
        "path": frame_path,
        "index": index,
        "timestamp": ts,
        "timestamp_text": fmt(ts),
        "target_timestamp": target_times[next_target],
        "target_timestamp_text": fmt(target_times[next_target])
    })
    next_target += 1
    if len(frames) == 1 or len(frames) == len(target_times) or len(frames) % progress_stride == 0:
        emit_progress(
            "decoded",
            extracted=len(frames),
            target_frames=len(target_times),
            timestamp=ts,
            timestamp_text=fmt(ts),
            percent=(len(frames) / max(1, len(target_times))) * 100.0,
        )

container.close()
if duration_s <= 0:
    duration_s = last_ts
emit_progress("complete", extracted=len(frames), target_frames=len(target_times), duration_s=duration_s)
print(json.dumps({
    "duration_s": duration_s,
    "source_fps": frame_rate,
    "source_width": source_width,
    "source_height": source_height,
    "sample_fps": fps,
    "count": len(frames),
    "frames": frames
}))
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

async function mediaSourceToBuffer(mediaSource) {
  const source = String(mediaSource || "");
  if (/^https?:\/\//i.test(source)) {
    const upstream = await fetch(source);
    if (!upstream.ok) {
      throw new Error(`Video URL fetch failed with HTTP ${upstream.status}`);
    }
    return {
      mime: upstream.headers.get("content-type")?.split(";")[0] || "application/octet-stream",
      buffer: Buffer.from(await upstream.arrayBuffer())
    };
  }
  return dataUrlToBuffer(source);
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

function frameFallbackAllowed() {
  if (process.env.REASONER_ENABLE_FRAME_FALLBACK === "1") return true;
  if (process.env.REASONER_DISABLE_FRAME_FALLBACK === "1") return false;
  return false;
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
  return (
    prepared?.media?.mode === "video_url" &&
    backend === "nim_local" &&
    frameFallbackAllowed() &&
    nativeVideoFallbackMessage(message)
  );
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
    const { buffer } = await mediaSourceToBuffer(mediaDataUrl);
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

function formatTimestamp(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(value / 60);
  const remainder = value - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(2).padStart(5, "0")}`;
}

function positiveNumber(value, fallback = null) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function longVideoFrameLimit() {
  const parsed = Number.parseInt(process.env.REASONER_LONG_MAX_FRAMES || "", 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 720;
}

function longVideoMaxConcurrency() {
  const parsed = Number.parseInt(process.env.REASONER_LONG_MAX_CONCURRENCY || "", 10);
  return Number.isFinite(parsed) && parsed > 0 ? Math.min(32, parsed) : 16;
}

function longVideoPresetConfig(presetRaw, durationSeconds = 0, requestedConcurrency, requestedFramesPerSecond) {
  const preset = String(presetRaw || "balanced").toLowerCase();
  const duration = Number(durationSeconds) || 0;
  const requestedFps = positiveNumber(requestedFramesPerSecond);
  const defaults =
    preset === "fast"
      ? {
          preset: "fast",
          sampleFps: 6,
          maxFrames: duration > 0 ? Math.ceil(duration * 6) : 720,
          chunkSize: 5,
          overlap: 0,
          concurrency: 16,
          chunkMaxTokens: 512,
          finalMaxTokens: 1200,
          chunkTimeoutMs: 45000,
          targetWidth: 640
        }
      : preset === "detailed"
        ? {
            preset: "detailed",
            sampleFps: 10,
            maxFrames: duration > 0 ? Math.ceil(duration * 10) : 1200,
            chunkSize: 5,
            overlap: 1,
            concurrency: 16,
            chunkMaxTokens: 900,
            finalMaxTokens: 2200,
            chunkTimeoutMs: 90000,
            targetWidth: 768
          }
        : {
            preset: "balanced",
            sampleFps: 8,
            maxFrames: duration > 0 ? Math.ceil(duration * 8) : 960,
            chunkSize: 5,
            overlap: 1,
            concurrency: 16,
            chunkMaxTokens: 560,
            finalMaxTokens: 900,
            chunkTimeoutMs: 65000,
            targetWidth: 704
        };
  const maxFrameLimit = longVideoFrameLimit();
  const sampleFps = requestedFps ? Math.min(30, requestedFps) : defaults.sampleFps;
  const requestedFrameBudget =
    duration > 0
      ? Math.ceil(duration * sampleFps)
      : Math.ceil(sampleFps * 120);
  const maxFrames = Math.min(maxFrameLimit, Math.max(defaults.maxFrames, requestedFrameBudget));
  const concurrency = Number.parseInt(String(requestedConcurrency || ""), 10);
  return {
    ...defaults,
    sampleFps,
    maxFrames,
    requestedFps,
    frameLimit: maxFrameLimit,
    maxConcurrency: longVideoMaxConcurrency(),
    concurrency: Number.isFinite(concurrency) && concurrency > 0 ? Math.min(longVideoMaxConcurrency(), concurrency) : defaults.concurrency,
    maxImagesPerChunk: LONG_VIDEO_MAX_IMAGES_PER_CHUNK
  };
}

function runLongFrameExtractor(videoPath, outputDir, config, onProgress) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      pythonForFrames(),
      [
        "-",
        videoPath,
        outputDir,
        String(config.sampleFps),
        String(config.maxFrames || ""),
        String(config.targetWidth || 704)
      ],
      { stdio: ["pipe", "pipe", "pipe"] }
    );
    let stdout = "";
    let stderr = "";
    let progressBuffer = "";
    const drainProgress = (text, flush = false) => {
      progressBuffer += text;
      const lines = progressBuffer.split(/\r?\n/);
      progressBuffer = flush ? "" : lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        if (line.startsWith("PROGRESS ")) {
          try {
            onProgress?.(JSON.parse(line.slice("PROGRESS ".length)));
          } catch {
            // Keep malformed progress out of the user-facing error.
          }
        } else {
          stderr += `${line}\n`;
        }
      }
      if (flush && progressBuffer.trim()) {
        stderr += `${progressBuffer}\n`;
        progressBuffer = "";
      }
    };
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      drainProgress(String(chunk));
    });
    child.on("error", reject);
    child.on("close", (code) => {
      drainProgress("\n", true);
      if (code !== 0) {
        reject(new Error(stderr || `Long video frame extractor exited with ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        reject(error);
      }
    });
    child.stdin.end(LONG_FRAME_EXTRACTOR_PY);
  });
}

async function extractLongVideoFrames(mediaSource, config, onProgress) {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "reason-vite-long-"));
  try {
    onProgress?.({ phase: "loading", percent: 0 });
    const { buffer } = await mediaSourceToBuffer(mediaSource);
    const videoPath = path.join(tempDir, "input.mp4");
    const frameDir = path.join(tempDir, "frames");
    await fs.mkdir(frameDir);
    await fs.writeFile(videoPath, buffer);
    onProgress?.({ phase: "decode_start", percent: 0 });
    const info = await runLongFrameExtractor(videoPath, frameDir, config, onProgress);
    const frames = [];
    for (const frame of info.frames || []) {
      const bytes = await fs.readFile(frame.path);
      frames.push({
        ...frame,
        dataUrl: `data:image/jpeg;base64,${bytes.toString("base64")}`,
        timestamp: Number(frame.timestamp) || 0,
        timestamp_text: frame.timestamp_text || formatTimestamp(frame.timestamp)
      });
    }
    if (frames.length === 0) throw new Error("Could not extract timestamped frames from video");
    return {
      tempDir,
      durationSeconds: Number(info.duration_s) || frames.at(-1)?.timestamp || 0,
      sourceFps: Number(info.source_fps) || null,
      sourceWidth: Number(info.source_width) || null,
      sourceHeight: Number(info.source_height) || null,
      sampleFps: Number(info.sample_fps) || config.sampleFps,
      frames
    };
  } catch (error) {
    await fs.rm(tempDir, { recursive: true, force: true });
    throw error;
  }
}

function chunkLongFrames(frames, config) {
  const size = Math.min(config.chunkSize || 5, LONG_VIDEO_MAX_IMAGES_PER_CHUNK);
  const overlap = Math.max(0, Math.min(config.overlap || 0, size - 1));
  const step = Math.max(1, size - overlap);
  const chunks = [];
  for (let start = 0; start < frames.length; start += step) {
    const slice = frames.slice(start, start + size);
    if (slice.length === 0) continue;
    chunks.push({
      index: chunks.length,
      frames: slice,
      startSeconds: slice[0].timestamp,
      endSeconds: slice.at(-1).timestamp,
      timeRange: `${slice[0].timestamp_text} - ${slice.at(-1).timestamp_text}`
    });
    if (start + size >= frames.length) break;
  }
  return chunks;
}

function stripReasoningFormatInstruction(prompt) {
  return String(prompt || "")
    .replace(/Answer the question using the following format:\s*<think>\s*Your reasoning\.\s*<\/think>\s*Write your final answer immediately after the <\/think> tag\./gi, "")
    .replace(/<think>\s*Your reasoning\.\s*<\/think>/gi, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function maybeJsonText(value) {
  const text = String(value || "").trim();
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = fenced ? fenced[1].trim() : text;
  const first = candidate.search(/[\[{]/);
  if (first < 0) return null;
  const trimmed = candidate.slice(first);
  for (let end = trimmed.length; end > Math.max(1, trimmed.length - 400); end -= 1) {
    try {
      return JSON.parse(trimmed.slice(0, end));
    } catch {
      // Keep searching for a valid JSON tail.
    }
  }
  return null;
}

function cleanModelText(value) {
  return String(value || "")
    .replace(/```(?:json)?/gi, "")
    .replace(/```/g, "")
    .trim();
}

function compactChunkSummary(parsed, content) {
  if (parsed?.summary) return String(parsed.summary);
  const cleaned = cleanModelText(content);
  const summaryMatch = cleaned.match(/"summary"\s*:\s*"((?:\\.|[^"\\])*)"/i);
  if (summaryMatch) {
    try {
      return JSON.parse(`"${summaryMatch[1]}"`);
    } catch {
      return summaryMatch[1].replace(/\\"/g, '"');
    }
  }
  const firstLine = cleaned
    .split(/\n+/)
    .map((line) => line.trim())
    .find((line) => line && !/^[{}\[\],]+$/.test(line));
  const fallback = firstLine || cleaned || "";
  return fallback.length > 320 ? `${fallback.slice(0, 317)}...` : fallback;
}

function normalizeBaseUrl(url) {
  const stripped = String(url || "").replace(/\/$/, "");
  return stripped.endsWith("/v1") ? stripped : `${stripped}/v1`;
}

function reasonerEndpointPool(defaultBaseUrl) {
  const raw = process.env.REASONER_ENDPOINTS || "";
  const values = raw
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  return (values.length ? values : [defaultBaseUrl]).map(normalizeBaseUrl);
}

async function postOpenAiJson(baseUrl, payload, timeoutMs, parentSignal) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const abort = () => controller.abort();
  parentSignal?.addEventListener?.("abort", abort, { once: true });
  try {
    const upstream = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.VLLM_API_KEY || process.env.NIM_API_KEY || "EMPTY"}`
      },
      body: JSON.stringify(payload),
      signal: controller.signal
    });
    let data = null;
    try {
      data = await upstream.json();
    } catch {
      data = null;
    }
    if (!upstream.ok) {
      throw new Error(data?.error?.message || data?.message || `Reasoner returned HTTP ${upstream.status}`);
    }
    return data;
  } finally {
    clearTimeout(timer);
    parentSignal?.removeEventListener?.("abort", abort);
  }
}

async function runConcurrent(items, concurrency, worker) {
  const results = new Array(items.length);
  let nextIndex = 0;
  const count = Math.max(1, Math.min(Number(concurrency) || 1, items.length || 1));
  async function loop() {
    while (nextIndex < items.length) {
      const index = nextIndex;
      nextIndex += 1;
      results[index] = await worker(items[index], index);
    }
  }
  await Promise.all(Array.from({ length: count }, loop));
  return results;
}

function chunkPrompt({ chunk, durationSeconds, prompt, preset }) {
  const task = stripReasoningFormatInstruction(prompt) || "Summarize what happens in this video segment.";
  const frames = chunk.frames
    .map((frame, index) => `Frame ${index + 1}: ${frame.timestamp_text} (${frame.timestamp.toFixed(2)}s)`)
    .join("\n");
  return `You are analyzing one timestamped window from a longer video.

Full clip duration: ${formatTimestamp(durationSeconds)}.
Window: ${chunk.timeRange}.
Preset: ${preset}.

Frame timestamp map:
${frames}

Original user task:
${task}

Return concise valid JSON only with this shape:
{
  "time_range": {"start": "${chunk.frames[0].timestamp_text}", "end": "${chunk.frames.at(-1).timestamp_text}"},
  "summary": "one-sentence visible summary",
  "events": [
    {"start": "mm:ss.ff", "end": "mm:ss.ff", "event_type": "visible event type", "caption": "what is visibly happening", "confidence": 0.0}
  ],
  "uncertainty": "what might be missed because only these frames were sampled",
  "evidence_frames": ["Frame 1", "Frame 2"]
}

Use only visible evidence from these frames. Do not invent events between frames. Keep output short.`;
}

function reducerPrompt({ chunks, durationSeconds, failedChunks, prompt, preset, warnings }) {
  const task = stripReasoningFormatInstruction(prompt) || "Summarize the full video.";
  const chunkText = chunks
    .map((chunk) => {
      const parsed = chunk.parsed ? JSON.stringify(chunk.parsed) : chunk.content;
      return `Chunk ${chunk.index + 1} (${chunk.timeRange}, ${chunk.status}): ${parsed || chunk.error || "No output"}`;
    })
    .join("\n\n");
  const failures = failedChunks.length
    ? `\nFailed chunks: ${failedChunks.map((chunk) => `${chunk.index + 1} ${chunk.timeRange}: ${chunk.error}`).join("; ")}`
    : "";
  return `Stitch these timestamped chunk analyses into one answer for the original user task.

Original user task:
${task}

Clip duration: ${formatTimestamp(durationSeconds)}
Analysis preset: ${preset}
Warnings: ${warnings.join("; ") || "none"}${failures}

Chunk analyses:
${chunkText}

Instructions:
- Preserve timestamp evidence across the full clip.
- Deduplicate overlapping events with the same event type and similar timestamp.
- Mention failed or sparse sections as limitations instead of hiding them.
- Follow the user's requested output format when possible. If no exact format is requested, return no more than 12 concise timeline bullets plus key limitations.
- Do not include <think> tags.`;
}

function timestampSeconds(value) {
  const text = String(value || "");
  const match = text.match(/^(\d+):(\d+(?:\.\d+)?)$/);
  if (!match) return Number.NaN;
  return Number(match[1]) * 60 + Number(match[2]);
}

function localLongVideoStitch({ chunks, durationSeconds, failedChunks, prompt, preset, warnings }) {
  const wantsJson = /\bjson\b/i.test(String(prompt || ""));
  const events = [];
  const chunkSummaries = [];
  for (const chunk of chunks) {
    const summary = chunk.summary || compactChunkSummary(chunk.parsed, chunk.content);
    if (summary) {
      chunkSummaries.push({
        index: chunk.index + 1,
        time_range: chunk.timeRange,
        summary: String(summary).replace(/```(?:json)?|```/gi, "").trim()
      });
    }
    const parsedEvents = Array.isArray(chunk.parsed?.events) ? chunk.parsed.events : [];
    for (const event of parsedEvents) {
      const start = String(event.start || chunk.timeRange?.split(" - ")[0] || "");
      const type = String(event.event_type || event.type || "visible_event");
      const seconds = timestampSeconds(start);
      const duplicate = events.some((existing) => {
        if (String(existing.event_type).toLowerCase() !== type.toLowerCase()) return false;
        const existingSeconds = timestampSeconds(existing.start);
        return Number.isFinite(seconds) && Number.isFinite(existingSeconds) && Math.abs(existingSeconds - seconds) < 1.5;
      });
      if (duplicate) continue;
      events.push({
        start,
        end: String(event.end || start),
        event_type: type,
        caption: String(event.caption || summary || "Visible event"),
        confidence: Number.isFinite(Number(event.confidence)) ? Number(event.confidence) : undefined,
        source_chunk: chunk.index + 1
      });
    }
  }
  events.sort((left, right) => timestampSeconds(left.start) - timestampSeconds(right.start));
  const failed = failedChunks.map((chunk) => ({
    index: chunk.index + 1,
    time_range: chunk.timeRange,
    error: chunk.error
  }));
  const samplingLimits = {
    clip_duration: formatTimestamp(durationSeconds),
    preset,
    frame_chunking: `Timestamped image chunks, max ${LONG_VIDEO_MAX_IMAGES_PER_CHUNK} images per NIM request`,
    warnings,
    failed_chunks: failed
  };

  if (wantsJson) {
    return JSON.stringify(
      {
        events,
        chunk_summaries: chunkSummaries,
        sampling_limits: samplingLimits
      },
      null,
      2
    );
  }

  const timeline =
    events.length > 0
      ? events.slice(0, 24).map((event) => `- ${event.start}${event.end && event.end !== event.start ? `-${event.end}` : ""}: ${event.event_type} — ${event.caption}`)
      : chunkSummaries.slice(0, 18).map((item) => `- ${item.time_range}: ${item.summary}`);
  const limitLines = warnings.map((warning) => `- ${warning}`);
  if (failed.length > 0) {
    limitLines.push(`- Failed chunks: ${failed.map((chunk) => `${chunk.index} (${chunk.time_range})`).join(", ")}`);
  }
  return [
    `Long video analysis (${preset})`,
    "",
    `Coverage: ${chunks.length - failed.length}/${chunks.length} chunks across ${formatTimestamp(durationSeconds)}.`,
    "",
    "Timeline",
    ...timeline,
    "",
    "Sampling limits",
    ...limitLines
  ].join("\n");
}

function longVideoWarnings(durationSeconds, frameCount, preset, config = {}) {
  const warnings = [];
  const coverage = durationSeconds > 0 ? frameCount / durationSeconds : 0;
  if (coverage > 0 && coverage < 2) {
    warnings.push(`${coverage.toFixed(2)} fps equivalent coverage; fast contacts or brief state changes may be missed.`);
  }
  if (config.requestedFps) {
    warnings.push(
      `Long Video is using the FPS slider at ${config.requestedFps} fps. Higher FPS creates more chunks because each NIM request can include only ${LONG_VIDEO_MAX_IMAGES_PER_CHUNK} frames.`
    );
  }
  if (config.requestedFps && coverage > 0 && coverage + 0.01 < config.requestedFps) {
    warnings.push(
      `Requested ${config.requestedFps} fps was capped to ${coverage.toFixed(2)} fps effective coverage by the ${config.frameLimit} frame budget.`
    );
  }
  if (config.concurrency > 8) {
    warnings.push(
      `Chunk concurrency is set to ${config.concurrency}. This horde RTX PRO 6000 handled 16 comfortably in probes; lower it if other users are sharing the NIM or TTFT spikes.`
    );
  }
  if (preset === "fast") warnings.push("Fast preset uses 6 fps for dense captioning, but very brief actions or contacts may still be missed.");
  if (preset === "detailed") warnings.push("Detailed preset may run longer than the clip duration.");
  warnings.push("NIM native video decode is bypassed here; Vite sends timestamped image chunks with at most 5 frames per request.");
  return warnings;
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

function safeNimEnv(rawEnv = []) {
  const allowPrefixes = [
    "NIM_",
    "MODEL_",
    "VLLM_",
    "COSMOS_",
    "INFERENCE_BACKEND",
    "REASONER_MEDIA_MODE"
  ];
  const blocked = /(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH)/i;
  const safe = {};
  for (const item of rawEnv || []) {
    const index = String(item).indexOf("=");
    if (index <= 0) continue;
    const key = item.slice(0, index);
    const value = item.slice(index + 1);
    if (blocked.test(key)) continue;
    if (!allowPrefixes.some((prefix) => key.startsWith(prefix))) continue;
    safe[key] = value;
  }
  return safe;
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 3000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function findNimContainer() {
  const containerName = process.env.CONTAINER_NAME || "cosmos-nim";
  try {
    const stdout = await execFileText("docker", ["inspect", containerName]);
    const data = JSON.parse(stdout);
    const container = Array.isArray(data) ? data[0] : null;
    if (!container) return null;
    return {
      name: container.Name ? String(container.Name).replace(/^\//, "") : containerName,
      image: container.Config?.Image || null,
      image_id: container.Image || null,
      status: container.State?.Status || null,
      started_at: container.State?.StartedAt || null,
      env: safeNimEnv(container.Config?.Env || [])
    };
  } catch {
    return null;
  }
}

async function fetchBackendModelInfo(baseUrl, selectedModel) {
  try {
    const response = await fetchWithTimeout(`${baseUrl}/models`, {
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
  const nim = backend === "nim_local" ? await findNimContainer() : null;
  const flags = vllmProcess?.flags || {};
  const quantization = inferQuantization(selectedModel, flags);
  const source = await resolveModelSource({ selectedModel, vllmProcess, modelInfo });
  return {
    source,
    app_source: appSourceInfo(),
    nim,
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
    nim: runtime.nim,
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

  if (
    mediaKind === "video" &&
    mediaDataUrl &&
    (forceFrameFallback || (!usesNativeVideoUrl(selectedModel) && frameFallbackAllowed()))
  ) {
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
      max_frames: mediaMode === "image-frame-fallback" ? frameFallbackLimit() : undefined
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

async function submitLongChunk({ chunk, config, durationSeconds, endpoints, model, params, prompt, signal, systemPrompt }) {
  const text = chunkPrompt({ chunk, durationSeconds, prompt, preset: config.preset });
  const { baseUrl, payload, redactedPayload } = buildReasoningPayload({
    model,
    prompt: text,
    systemPrompt: systemPrompt
      ? `${systemPrompt}\n\nFor this long-video chunk pass, return concise JSON only. Do not include hidden reasoning.`
      : "Return concise JSON only for this long-video chunk pass. Do not include hidden reasoning.",
    mediaFrames: chunk.frames.map((frame) => frame.dataUrl),
    framesPerSecond: config.sampleFps,
    params: {
      ...params,
      temperature: Math.min(Number(params.temperature) || 0.6, 0.6),
      max_tokens: config.chunkMaxTokens
    }
  });
  delete payload.mm_processor_kwargs;
  payload.max_tokens = config.chunkMaxTokens;

  const endpoint = endpoints[chunk.index % endpoints.length] || baseUrl;
  const startedAt = Date.now();
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const attemptPayload = attempt === 0 ? payload : { ...payload, max_tokens: Math.min(384, config.chunkMaxTokens) };
      const data = await postOpenAiJson(endpoint, attemptPayload, config.chunkTimeoutMs, signal);
      const normalized = normalizeReasonerMessage(data);
      const content = normalized.answer || normalized.combined;
      const parsed = maybeJsonText(content);
      return {
        index: chunk.index,
        status: "done",
        timeRange: chunk.timeRange,
        startSeconds: chunk.startSeconds,
        endSeconds: chunk.endSeconds,
        content,
        reasoning: normalized.reasoning,
        parsed,
        summary: compactChunkSummary(parsed, content),
        elapsedSeconds: (Date.now() - startedAt) / 1000,
        endpoint,
        payload: redactedPayload
      };
    } catch (error) {
      if (attempt === 1 || signal?.aborted) {
        return {
          index: chunk.index,
          status: "error",
          timeRange: chunk.timeRange,
          startSeconds: chunk.startSeconds,
          endSeconds: chunk.endSeconds,
          error: error instanceof Error ? error.message : "Chunk inference failed",
          elapsedSeconds: (Date.now() - startedAt) / 1000,
          endpoint,
          payload: redactedPayload
        };
      }
    }
  }
  return {
    index: chunk.index,
    status: "error",
    timeRange: chunk.timeRange,
    error: "Chunk inference failed"
  };
}

async function stitchLongVideo({ chunks, config, durationSeconds, endpoints, failedChunks, model, params, prompt, signal, warnings }) {
  if (config.preset !== "detailed" && process.env.REASONER_LONG_USE_MODEL_REDUCER !== "1") {
    const content = localLongVideoStitch({ chunks, durationSeconds, failedChunks, prompt, preset: config.preset, warnings });
    return {
      status: "success",
      content,
      reasoning: "",
      combined_content: content,
      schema: /\bjson\b/i.test(String(prompt || "")) ? "local_long_video_json" : "local_long_video_timeline",
      openai: {
        id: `chatcmpl-byo-long-local-${Date.now()}`,
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model,
        choices: [
          {
            index: 0,
            message: { role: "assistant", content },
            finish_reason: "stop"
          }
        ],
        usage: null
      }
    };
  }

  const text = reducerPrompt({ chunks, durationSeconds, failedChunks, prompt, preset: config.preset, warnings });
  const { baseUrl, payload, redactedPayload } = buildReasoningPayload({
    model,
    prompt: text,
    systemPrompt: "You are a timeline reducer for long video analysis. Return the final answer only.",
    params: {
      ...params,
      temperature: Math.min(Number(params.temperature) || 0.6, 0.6),
      max_tokens: config.finalMaxTokens
    }
  });
  delete payload.mm_processor_kwargs;
  payload.max_tokens = config.finalMaxTokens;
  const endpoint = endpoints[0] || baseUrl;
  try {
    const data = await postOpenAiJson(endpoint, payload, Math.max(config.chunkTimeoutMs, 90000), signal);
    const normalized = normalizeReasonerMessage(data);
    return {
      status: "success",
      content: normalized.answer || normalized.combined,
      reasoning: normalized.reasoning,
      combined_content: normalized.combined,
      schema: normalized.schema,
      openai: normalized.normalized,
      raw: data,
      payload: redactedPayload
    };
  } catch (error) {
    const fallback = [
      "Long video analysis completed, but the final stitching pass failed.",
      "",
      ...chunks
        .filter((chunk) => chunk.status === "done")
        .map((chunk) => `- ${chunk.timeRange}: ${chunk.parsed?.summary || chunk.content || "No summary"}`),
      ...failedChunks.map((chunk) => `- ${chunk.timeRange}: failed (${chunk.error})`)
    ].join("\n");
    return {
      status: "success",
      content: fallback,
      reasoning: "",
      combined_content: fallback,
      schema: "local_chunk_fallback",
      message: error instanceof Error ? `Reducer failed: ${error.message}` : "Reducer failed",
      payload: redactedPayload
    };
  }
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

app.post("/api/reason/long/stream", async (request, response) => {
  const upstreamAbort = new AbortController();
  let clientClosed = false;
  let extracted = null;
  const startedAt = Date.now();

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

  function elapsedSeconds() {
    return (Date.now() - startedAt) / 1000;
  }

  try {
    const stepState = {
      read_video: { label: "Read video", status: "pending", progress: 0, detail: "" },
      decode_frames: { label: "Decode frames", status: "pending", progress: 0, detail: "" },
      plan_chunks: { label: "Plan chunks", status: "pending", progress: 0, detail: "" },
      run_chunks: { label: "Run chunks", status: "pending", progress: 0, detail: "" },
      stitch_timeline: { label: "Stitch timeline", status: "pending", progress: 0, detail: "" }
    };
    const stepPayload = () =>
      Object.entries(stepState).map(([key, value]) => ({
        key,
        label: value.label,
        status: value.status,
        progress: Math.max(0, Math.min(100, Math.round(Number(value.progress) || 0))),
        detail: value.detail || ""
      }));
    const updateStep = (key, patch) => {
      stepState[key] = { ...stepState[key], ...patch };
    };
    const emitLongState = (payload) => {
      sse(response, "long_state", {
        ...payload,
        steps: stepPayload()
      });
    };
    const body = request.body || {};
    const params = body.params || {};
    const selectedModel = body.model || defaultModel;
    const mediaSource = body.video || (body.mediaKind === "video" ? body.mediaDataUrl : "");
    if (!mediaSource) throw new Error("Long Video Analysis requires an MP4/video input.");

    const requestedLongFps = params.frames_per_second ?? body.framesPerSecond ?? body.longVideoFps;
    let config = longVideoPresetConfig(body.preset, 0, body.concurrency, requestedLongFps);
    updateStep("read_video", { status: "running", progress: 10, detail: "Opening media" });
    emitLongState({
      phase: "media_scan",
      message: "Scanning video and extracting timestamped frames",
      percent: 2,
      elapsedSeconds: elapsedSeconds()
    });
    const handleDecodeProgress = (event) => {
      if (event.phase === "loading") {
        updateStep("read_video", { status: "running", progress: 45, detail: "Loading video bytes" });
        emitLongState({
          phase: "media_scan",
          message: "Loading video bytes",
          percent: 2,
          elapsedSeconds: elapsedSeconds()
        });
        return;
      }
      if (event.phase === "decode_start") {
        updateStep("read_video", { status: "done", progress: 100, detail: "Video loaded" });
        updateStep("decode_frames", { status: "running", progress: 2, detail: "Starting decoder" });
      } else if (event.phase === "opened") {
        const target = Number(event.target_frames) || config.maxFrames;
        updateStep("read_video", { status: "done", progress: 100, detail: "Video metadata read" });
        updateStep("decode_frames", {
          status: "running",
          progress: 3,
          detail: `Target ${target} frames at ${config.sampleFps} fps`
        });
      } else if (event.phase === "decoded") {
        const extractedFrames = Number(event.extracted) || 0;
        const target = Number(event.target_frames) || config.maxFrames || extractedFrames || 1;
        const decodePercent = Math.max(0, Math.min(100, Number(event.percent) || (extractedFrames / target) * 100));
        updateStep("decode_frames", {
          status: "running",
          progress: decodePercent,
          detail: `${extractedFrames}/${target} frames${event.timestamp_text ? ` through ${event.timestamp_text}` : ""}`
        });
        emitLongState({
          phase: "media_decode",
          message: `Decoding frames ${extractedFrames}/${target}`,
          percent: Math.min(7, Math.round(2 + decodePercent * 0.05)),
          frameCount: extractedFrames,
          elapsedSeconds: elapsedSeconds()
        });
      } else if (event.phase === "complete") {
        updateStep("decode_frames", {
          status: "done",
          progress: 100,
          detail: `${Number(event.extracted) || 0} frames extracted`
        });
      }
    };
    extracted = await extractLongVideoFrames(mediaSource, config, handleDecodeProgress);
    config = longVideoPresetConfig(body.preset, extracted.durationSeconds, body.concurrency, requestedLongFps);

    if (Math.abs((extracted.sampleFps || 0) - config.sampleFps) > 0.01 || extracted.frames.length > config.maxFrames) {
      await fs.rm(extracted.tempDir, { recursive: true, force: true });
      updateStep("decode_frames", {
        status: "running",
        progress: 0,
        detail: `Re-decoding at ${config.sampleFps} fps after reading duration`
      });
      extracted = await extractLongVideoFrames(mediaSource, config, handleDecodeProgress);
    }

    updateStep("decode_frames", {
      status: "done",
      progress: 100,
      detail: `${extracted.frames.length} frames extracted`
    });
    updateStep("plan_chunks", { status: "running", progress: 35, detail: "Building timestamp windows" });
    const effectiveSampleFps =
      extracted.durationSeconds > 0
        ? Number((extracted.frames.length / extracted.durationSeconds).toFixed(2))
        : extracted.sampleFps;
    const chunks = chunkLongFrames(extracted.frames, config);
    updateStep("plan_chunks", { status: "done", progress: 100, detail: `${chunks.length} chunks planned` });
    updateStep("run_chunks", { status: "pending", progress: 0, detail: `${config.concurrency} concurrent requests` });
    const warnings = longVideoWarnings(extracted.durationSeconds, extracted.frames.length, config.preset, config);
    const { baseUrl } = buildReasoningPayload({ model: selectedModel, prompt: "probe", params: {} });
    const endpoints = reasonerEndpointPool(baseUrl);
    const progress = new Map();
    let completed = 0;
    let failed = 0;
    let latencyTotal = 0;
    let latencyCount = 0;

    sse(response, "long_plan", {
      phase: "extraction_plan",
      preset: config.preset,
      durationSeconds: extracted.durationSeconds,
      durationText: formatTimestamp(extracted.durationSeconds),
      sourceFps: extracted.sourceFps,
      sourceWidth: extracted.sourceWidth,
      sourceHeight: extracted.sourceHeight,
      frameCount: extracted.frames.length,
      requestedFps: config.requestedFps || null,
      sampleFps: effectiveSampleFps,
      extractionFps: extracted.sampleFps,
      frameLimit: config.frameLimit,
      chunkSize: config.chunkSize,
      maxImagesPerChunk: LONG_VIDEO_MAX_IMAGES_PER_CHUNK,
      totalChunks: chunks.length,
      concurrency: config.concurrency,
      maxConcurrency: config.maxConcurrency,
      endpoints,
      warnings,
      steps: stepPayload(),
      chunks: chunks.map((chunk) => ({
        index: chunk.index,
        status: "queued",
        timeRange: chunk.timeRange,
        frameCount: chunk.frames.length,
        thumbnailUrl: chunk.frames[0]?.dataUrl || null
      })),
      percent: 8,
      elapsedSeconds: elapsedSeconds()
    });

    const emitProgress = (phase, message) => {
      const total = Math.max(1, chunks.length);
      const active = [...progress.values()].filter((item) => item.status === "running").length;
      const doneRatio = (completed + failed) / total;
      const averageLatency = latencyCount > 0 ? latencyTotal / latencyCount : null;
      const remaining = Math.max(0, total - completed - failed);
      const etaSeconds = averageLatency ? (remaining / Math.max(1, config.concurrency)) * averageLatency : null;
      updateStep("run_chunks", {
        status: remaining > 0 ? "running" : failed > 0 ? "error" : "done",
        progress: doneRatio * 100,
        detail: `${completed + failed}/${chunks.length} chunks complete; ${active} running`
      });
      emitLongState({
        phase,
        message,
        completedChunks: completed,
        failedChunks: failed,
        runningChunks: active,
        totalChunks: chunks.length,
        percent: Math.min(95, Math.round(8 + doneRatio * 82)),
        elapsedSeconds: elapsedSeconds(),
        etaSeconds,
        concurrency: config.concurrency,
        maxConcurrency: config.maxConcurrency
      });
    };

    emitProgress("chunking", `Running chunks 0/${chunks.length}`);
    const results = await runConcurrent(chunks, config.concurrency, async (chunk) => {
      progress.set(chunk.index, { status: "running" });
      sse(response, "long_chunk", {
        index: chunk.index,
        status: "running",
        timeRange: chunk.timeRange,
        frameCount: chunk.frames.length
      });
      emitProgress("chunking", `Running chunks ${completed + failed}/${chunks.length}`);

      const result = await submitLongChunk({
        chunk,
        config,
        durationSeconds: extracted.durationSeconds,
        endpoints,
        model: selectedModel,
        params,
        prompt: body.prompt || body.userPrompt || "",
        signal: upstreamAbort.signal,
        systemPrompt: body.systemPrompt || body.system_prompt || ""
      });

      progress.set(chunk.index, { status: result.status });
      if (result.status === "done") {
        completed += 1;
        latencyTotal += result.elapsedSeconds || 0;
        latencyCount += 1;
      } else {
        failed += 1;
      }
      sse(response, "long_chunk", {
        index: result.index,
        status: result.status,
        timeRange: result.timeRange,
        elapsedSeconds: result.elapsedSeconds,
        summary: result.summary || result.parsed?.summary || result.content,
        content: result.content || "",
        parsed: result.parsed || null,
        events: Array.isArray(result.parsed?.events) ? result.parsed.events : [],
        eventsCount: Array.isArray(result.parsed?.events) ? result.parsed.events.length : undefined,
        error: result.error
      });
      if (result.status === "done") {
        sse(response, "long_partial", {
          index: result.index,
          timeRange: result.timeRange,
          summary: result.summary || result.parsed?.summary || result.content || ""
        });
      }
      emitProgress("chunking", `Running chunks ${completed + failed}/${chunks.length}`);
      return result;
    });

    if (clientClosed || upstreamAbort.signal.aborted) return;
    const doneChunks = results.filter((item) => item?.status === "done");
    const failedChunks = results.filter((item) => item?.status === "error");
    updateStep("run_chunks", {
      status: failedChunks.length > 0 ? "error" : "done",
      progress: 100,
      detail: `${doneChunks.length}/${chunks.length} chunks succeeded`
    });
    updateStep("stitch_timeline", { status: "running", progress: 20, detail: "Combining chunk results" });
    emitLongState({
      phase: "stitching",
      message: "Stitching timeline",
      completedChunks: completed,
      failedChunks: failed,
      totalChunks: chunks.length,
      percent: 96,
      elapsedSeconds: elapsedSeconds()
    });

    const stitched = await stitchLongVideo({
      chunks: results,
      config,
      durationSeconds: extracted.durationSeconds,
      endpoints,
      failedChunks,
      model: selectedModel,
      params,
      prompt: body.prompt || body.userPrompt || "",
      signal: upstreamAbort.signal,
      warnings
    });

    const result = {
      ...stitched,
      media: {
        mode: "long-video-image-chunks",
        duration_seconds: extracted.durationSeconds,
        frame_count: extracted.frames.length,
        requested_fps: config.requestedFps || null,
        sample_fps: effectiveSampleFps,
        extraction_fps: extracted.sampleFps,
        frame_limit: config.frameLimit,
        chunk_count: chunks.length,
        concurrency: config.concurrency,
        max_concurrency: config.maxConcurrency,
        max_images_per_chunk: LONG_VIDEO_MAX_IMAGES_PER_CHUNK,
        completed_chunks: doneChunks.length,
        failed_chunks: failedChunks.length,
        warnings
      },
      long_video: {
        preset: config.preset,
        chunks: results.map((item) => ({
          index: item.index,
          status: item.status,
          timeRange: item.timeRange,
          summary: item.summary || item.parsed?.summary || item.content || "",
          content: item.content || "",
          parsed: item.parsed || null,
          events: item.parsed?.events || [],
          error: item.error,
          elapsedSeconds: item.elapsedSeconds
        }))
      }
    };

    sse(response, "raw", result);
    updateStep("stitch_timeline", { status: "done", progress: 100, detail: "Final response ready" });
    emitLongState({
      phase: "complete",
      message: "Complete",
      completedChunks: completed,
      failedChunks: failed,
      totalChunks: chunks.length,
      percent: 100,
      elapsedSeconds: elapsedSeconds(),
      etaSeconds: 0
    });
    sse(response, "state", { phase: "complete" });
  } catch (error) {
    if (!clientClosed && !response.writableEnded) {
      sse(response, "error", { message: error instanceof Error ? error.message : "Long video analysis failed" });
      sse(response, "state", { phase: "error" });
    }
  } finally {
    if (extracted?.tempDir) await fs.rm(extracted.tempDir, { recursive: true, force: true });
    if (!response.writableEnded) response.end();
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
    if (isAlpamayoBackend) sse(response, "state", { phase: "preparing_media" });
    prepared = await prepareReasonRequest(request.body || {});
    if (isAlpamayoBackend) {
      sse(response, "state", {
        phase: "alpamayo_generating",
        note:
          "Alpamayo generate_text returns a complete answer rather than token callbacks; Vite uses a non-streaming adapter request for this backend."
      });
      const fallback = await submitPreparedReasoning(prepared);
      emitFallback(response, fallback);
      return;
    }
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
