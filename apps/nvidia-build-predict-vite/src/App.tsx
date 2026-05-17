"use client";

import {
  ChevronDown,
  ChevronRight,
  Copy,
  ExternalLink,
  FileVideo,
  HelpCircle,
  Image as ImageIcon,
  Info,
  Menu,
  Play,
  RotateCcw,
  Search,
  Upload,
  X
} from "lucide-react";
import { ChangeEvent, CSSProperties, DragEvent, useEffect, useMemo, useRef, useState } from "react";

const HERO_IMAGE = "https://assets.ngc.nvidia.com/products/api-catalog/images/cosmos-predict1-5b.jpg";
const QUICK_VIDEO_PARAMS = {
  resolution: "256",
  aspect_ratio: "16,9",
  frames_count: 25,
  frames_per_sec: 24,
  num_steps: 4,
  guidance: 6
};
const ENABLE_ACTION_POLICY =
  (typeof import.meta !== "undefined" &&
    (import.meta as { env?: Record<string, string | undefined> }).env?.VITE_ENABLE_ACTION_POLICY === "1") ||
  false;

const COSMOS3_INFO_URL =
  (typeof import.meta !== "undefined" && (import.meta as { env?: Record<string, string> }).env?.VITE_COSMOS3_INFO_URL) ||
  "/api/active-model";
const DEFAULT_MODEL =
  (typeof import.meta !== "undefined" && (import.meta as { env?: Record<string, string> }).env?.VITE_MODEL_NAME) ||
  "Detecting model...";
const HERO_TAGS = [
  "physical ai",
  "world foundation model",
  "robotics",
  "simulation",
  "synthetic data generation",
  "text-to-video",
  "image-to-video",
  "future state generation"
];
const MODEL_CHOICES = [
  "nvidia/cosmos3-gen",
  "nvidia/cosmos-predict1-5b",
  "nvidia/cosmos-predict1-7b-video2world",
  "nvidia/cosmos-predict2-5-2b",
  "nvidia/cosmos-predict2-5-14b",
  "Cosmos3-Nano"
];
const COLLECTIONS = ["cosmos-predict1", "cosmos-predict25", "cosmos3", "nvidia-cosmos-2", "cosmos"];
const PROGRESS_FRAME_COUNT = 6;

const ETA_PIXELS_BY_RESOLUTION: Record<string, number> = {
  "256": 256 * 256,
  "480": 854 * 480,
  "720": 1280 * 720,
  "1080": 1920 * 1080
};
const ETA_BASELINE_SECONDS = 5;
const ETA_OPS_PER_SECOND = 10_000_000;

type GeneratorMode = "Text-to-Video" | "Image-to-Video" | "Action Policy";
type SchemaMode = "local_nim" | "build_openapi";
type OutputTab = "preview" | "json";
type MobilePanel = "input" | "output";
type MediaState = {
  name: string;
  kind: "video" | "image";
  previewUrl: string;
  dataUrl?: string;
  sourceUrl?: string;
};
type ExampleItem = {
  id: string;
  label: string;
  eyebrow: string;
  mode: GeneratorMode;
  prompt: string;
  mediaUrl?: string;
  mediaName?: string;
  mediaKind?: "image" | "video";
  hidden?: boolean;
  params?: {
    resolution?: string;
    numFrames?: number;
    fps?: number;
    steps?: number;
    guidance?: number;
    seed?: number;
    shift?: number;
    imageSize?: number;
    actionChunkSize?: number;
    rawActionDim?: number;
    domainName?: string;
    actionMode?: string;
  };
};
type ApiResult = {
  videoDataUrl?: string;
  imageDataUrl?: string;
  assetUrl?: string;
  error?: string;
  status?: string;
  message?: string;
  content?: Record<string, unknown> | null;
  action?: unknown;
  files?: Array<{
    path?: string;
    mime?: string;
    hasInlineData?: boolean;
    url?: string;
    error?: string;
  }>;
  diagnostic?: Record<string, unknown>;
  payload?: unknown;
  raw?: unknown;
};
type BackendInfo = {
  checkpoint?: string;
  display_name?: string;
  cosmos3_version?: string;
  backend?: string;
  gpu_name?: string;
  vram_free_gib?: number;
  vram_total_gib?: number;
  base_url?: string;
  infer_url?: string;
  image?: string;
  output_dir?: string;
  warning?: string;
  capabilities?: Record<string, boolean>;
  staged_checkpoint?: {
    image?: string;
    served_model?: string;
    updated_at?: string;
    [key: string]: unknown;
  } | null;
  environment?: {
    cosmos3_version?: string;
    commit_sha?: string;
    [key: string]: unknown;
  };
};

const EXAMPLES: ExampleItem[] = [
  {
    id: "omni-t2v",
    label: "Robotic Fruit Picking",
    eyebrow: "T2V",
    mode: "Text-to-Video",
    prompt:
      "A smooth first-person robot manipulation video in a greenhouse. The robot arm reaches toward a ripe red apple, gently grasps it, twists, and places it into a harvest bin. Natural daylight, stable camera, realistic physics.",
    params: {
      resolution: QUICK_VIDEO_PARAMS.resolution,
      numFrames: QUICK_VIDEO_PARAMS.frames_count,
      fps: QUICK_VIDEO_PARAMS.frames_per_sec,
      steps: QUICK_VIDEO_PARAMS.num_steps,
      guidance: QUICK_VIDEO_PARAMS.guidance,
      seed: 0
    }
  },
  {
    id: "omni-i2v",
    label: "Robot Tabletop Motion",
    eyebrow: "I2V",
    mode: "Image-to-Video",
    prompt:
      "Animate the robot arm so it moves with small, precise adjustments while keeping the tabletop scene consistent. Maintain the original camera angle and realistic lighting.",
    mediaUrl:
      "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/robot_153.jpg",
    mediaName: "robot_153.jpg",
    mediaKind: "image",
    params: {
      resolution: QUICK_VIDEO_PARAMS.resolution,
      numFrames: QUICK_VIDEO_PARAMS.frames_count,
      fps: QUICK_VIDEO_PARAMS.frames_per_sec,
      steps: QUICK_VIDEO_PARAMS.num_steps,
      guidance: QUICK_VIDEO_PARAMS.guidance,
      seed: 0
    }
  },
  {
    id: "omni-action-policy",
    label: "Bridge Robot Policy",
    eyebrow: "Action",
    mode: "Action Policy",
    prompt: "Predict the next robot action chunk from the bridge robot observation video.",
    mediaUrl:
      "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/bridge_orig_lerobot.mp4",
    mediaName: "bridge_orig_lerobot.mp4",
    mediaKind: "video",
    hidden: !ENABLE_ACTION_POLICY,
    params: {
      resolution: "480",
      imageSize: 480,
      numFrames: 24,
      fps: 5,
      steps: 30,
      guidance: 1,
      seed: 0,
      shift: 5,
      actionMode: "policy",
      actionChunkSize: 16,
      rawActionDim: 10,
      domainName: "bridge_orig_lerobot"
    }
  }
];
const DEFAULT_PROMPT = EXAMPLES[0].prompt;

const GENERATOR_MODES: Array<{
  id: GeneratorMode;
  label: string;
  description: string;
  icon: typeof FileVideo;
  enabled: boolean;
  visible: boolean;
}> = [
  {
    id: "Text-to-Video",
    label: "Text-to-Video",
    description: "Prompt-only generation with quick staging defaults.",
    icon: FileVideo,
    enabled: true,
    visible: true
  },
  {
    id: "Image-to-Video",
    label: "Image-to-Video",
    description: "Animate one conditioning image plus a prompt.",
    icon: ImageIcon,
    enabled: true,
    visible: true
  },
  {
    id: "Action Policy",
    label: "Action Policy",
    description: "Robot action scaffold, hidden until backend smoke passes.",
    icon: FileVideo,
    enabled: false,
    visible: ENABLE_ACTION_POLICY
  }
];

function estimateWallSeconds(resolution: string | number, numFrames: number, numSteps: number): number {
  const px = ETA_PIXELS_BY_RESOLUTION[String(resolution)] || ETA_PIXELS_BY_RESOLUTION["480"];
  const f = Math.max(1, numFrames);
  const s = Math.max(1, numSteps);
  return Math.ceil(ETA_BASELINE_SECONDS + (px * f * s) / ETA_OPS_PER_SECOND);
}

function formatDuration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}

function displayValue(value: unknown, fallback = "unknown") {
  if (value === undefined || value === null || value === "") return fallback;
  return String(value);
}

function shortSha(sha?: string | null) {
  return sha ? sha.slice(0, 12) : "unknown";
}

function nimFrameCount(frames: number) {
  const requested = Math.max(25, Math.round(frames));
  const remainder = (requested - 1) % 4;
  return remainder === 0 ? requested : requested + (4 - remainder);
}

function nimRequestParams(resolution: string | number, frames: number, fps: number) {
  return {
    resolution: String(resolution),
    num_output_frames: nimFrameCount(frames),
    fps
  };
}

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function drawVideoCover(video: HTMLVideoElement, canvas: HTMLCanvasElement) {
  const context = canvas.getContext("2d");
  if (!context || !video.videoWidth || !video.videoHeight) return;
  const sourceRatio = video.videoWidth / video.videoHeight;
  const targetRatio = canvas.width / canvas.height;
  let sx = 0;
  let sy = 0;
  let sw = video.videoWidth;
  let sh = video.videoHeight;

  if (sourceRatio > targetRatio) {
    sw = video.videoHeight * targetRatio;
    sx = (video.videoWidth - sw) / 2;
  } else {
    sh = video.videoWidth / targetRatio;
    sy = (video.videoHeight - sh) / 2;
  }

  context.drawImage(video, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
}

function extractVideoFrames(src: string, count = PROGRESS_FRAME_COUNT): Promise<string[]> {
  return new Promise((resolve) => {
    const video = document.createElement("video");
    const canvas = document.createElement("canvas");
    const frames: string[] = [];
    let isDone = false;

    const finish = () => {
      if (isDone) return;
      isDone = true;
      resolve(frames);
    };

    const seekNext = () => {
      if (frames.length >= count) {
        finish();
        return;
      }
      const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 5;
      const nextTime = Math.min(duration - 0.04, (duration * (frames.length + 1)) / (count + 1));
      video.currentTime = Math.max(0, nextTime);
    };

    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";
    video.crossOrigin = "anonymous";
    video.addEventListener("loadedmetadata", () => {
      canvas.width = 360;
      canvas.height = 202;
      seekNext();
    });
    video.addEventListener("seeked", () => {
      try {
        drawVideoCover(video, canvas);
        frames.push(canvas.toDataURL("image/jpeg", 0.82));
        seekNext();
      } catch {
        finish();
      }
    });
    video.addEventListener("error", finish);
    window.setTimeout(finish, 8000);
    video.src = src;
    video.load();
  });
}

async function buildProgressFrames(media: MediaState): Promise<string[]> {
  if (media.kind === "image") {
    return Array.from({ length: PROGRESS_FRAME_COUNT }, () => media.previewUrl);
  }
  return extractVideoFrames(media.previewUrl);
}

function resultAssetUrl(result: ApiResult | null) {
  if (!result) return null;
  return result.assetUrl || result.files?.find((file) => file.url)?.url || null;
}

export default function Page() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [collection, setCollection] = useState("cosmos3");
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [models, setModels] = useState<string[]>(MODEL_CHOICES);
  const [backendInfo, setBackendInfo] = useState<BackendInfo | null>(null);
  const [runtimeOpen, setRuntimeOpen] = useState(false);
  const [generatorMode, setGeneratorMode] = useState<GeneratorMode>("Text-to-Video");
  const [schemaMode, setSchemaMode] = useState<SchemaMode>("local_nim");
  const [media, setMedia] = useState<MediaState | null>(null);
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [resolution, setResolution] = useState(QUICK_VIDEO_PARAMS.resolution);
  const [numFrames, setNumFrames] = useState(QUICK_VIDEO_PARAMS.frames_count);
  const [fps, setFps] = useState(QUICK_VIDEO_PARAMS.frames_per_sec);
  const [guidanceScale, setGuidanceScale] = useState(QUICK_VIDEO_PARAMS.guidance);
  const [steps, setSteps] = useState(QUICK_VIDEO_PARAMS.num_steps);
  const [seed, setSeed] = useState(0);
  const [status, setStatus] = useState("Ready");
  const [isRunning, setIsRunning] = useState(false);
  const [progressPercent, setProgressPercent] = useState(0);
  const [submittedAt, setSubmittedAt] = useState<number | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [progressFrames, setProgressFrames] = useState<string[]>([]);
  const [result, setResult] = useState<ApiResult | null>(null);
  const [copied, setCopied] = useState(false);
  const [outputTab, setOutputTab] = useState<OutputTab>("preview");
  const [mobilePanel, setMobilePanel] = useState<MobilePanel>("input");
  const [examplesOpen, setExamplesOpen] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [selectedExampleId, setSelectedExampleId] = useState(EXAMPLES[0].id);

  const activeExample = useMemo(
    () => EXAMPLES.find((example) => example.id === selectedExampleId) ?? EXAMPLES[0],
    [selectedExampleId]
  );
  const visibleExamples = useMemo(() => EXAMPLES.filter((example) => !example.hidden), []);
  const visibleModes = useMemo(() => GENERATOR_MODES.filter((mode) => mode.visible), []);
  const mediaRequired = generatorMode !== "Text-to-Video";
  const accepts = generatorMode === "Image-to-Video" ? ".jpg,.jpeg,.png,.webp" : ".mp4,.mov,.jpg,.jpeg,.png,.webp";
  const assetUrl = useMemo(() => resultAssetUrl(result), [result]);
  const isNimBackend = useMemo(
    () => String(backendInfo?.backend || "").toLowerCase().includes("nim"),
    [backendInfo?.backend]
  );

  useEffect(() => {
    let cancelled = false;
    fetch(COSMOS3_INFO_URL, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled || !d) return;
        const name =
          (d.checkpoint as string | undefined) ||
          (d.display_name as string | undefined) ||
          (Array.isArray(d.models) ? (d.models[0] as string | undefined) : undefined);
        if (name) {
          setModel(name);
          setModels((prev) => (prev.includes(name) ? prev : [name, ...prev]));
        }
        setBackendInfo(d);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    fetch("/api/models")
      .then((response) => response.json())
      .then((data) => {
        if (Array.isArray(data.models) && data.models.length > 0) {
          const merged = Array.from(new Set([...data.models, ...MODEL_CHOICES]));
          setModels(merged);
        }
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    document.title = `${model} Model by NVIDIA | NVIDIA NIM`;
  }, [model]);

  useEffect(() => {
    let cancelled = false;
    setProgressFrames([]);
    if (!media) return () => undefined;

    buildProgressFrames(media)
      .then((frames) => {
        if (!cancelled) setProgressFrames(frames);
      })
      .catch(() => {
        if (!cancelled) setProgressFrames([]);
      });

    return () => {
      cancelled = true;
    };
  }, [media]);

  const etaSeconds = useMemo(() => estimateWallSeconds(resolution, numFrames, steps), [numFrames, resolution, steps]);

  useEffect(() => {
    if (!isRunning || submittedAt === null) return () => undefined;
    const tick = () => {
      const elapsed = (Date.now() - submittedAt) / 1000;
      setElapsedSeconds(elapsed);
      const pct = Math.min(96, (elapsed / Math.max(1, etaSeconds)) * 96);
      setProgressPercent(pct);
    };
    tick();
    const interval = window.setInterval(tick, 250);
    return () => window.clearInterval(interval);
  }, [isRunning, submittedAt, etaSeconds]);

  const requestPreview = useMemo(() => {
    const visionPath = media?.sourceUrl ?? (media?.dataUrl ? "<written to /tmp/uploads/...>" : null);
    const params: Record<string, unknown> = {
      num_frames: numFrames,
      resolution,
      aspect_ratio: QUICK_VIDEO_PARAMS.aspect_ratio,
      fps,
      num_steps: steps,
      guidance: guidanceScale,
      seed
    };
    if (generatorMode === "Action Policy") {
      params.action_mode = activeExample.params?.actionMode ?? "policy";
      params.domain_name = activeExample.params?.domainName;
      params.image_size = activeExample.params?.imageSize ?? Number(resolution);
      params.action_chunk_size = activeExample.params?.actionChunkSize;
      params.raw_action_dim = activeExample.params?.rawActionDim;
      params.shift = activeExample.params?.shift;
    }

    if (schemaMode === "build_openapi") {
      return {
        endpoint: "POST https://ai.api.nvidia.com/v1/infer",
        collection,
        model,
        mode: generatorMode,
        payload: {
          prompt,
          image_url: generatorMode === "Image-to-Video" ? visionPath : undefined,
          seed,
          ...params
        },
        response: { asset_url: "https://..." }
      };
    }

    if (isNimBackend) {
      const payload: Record<string, unknown> = {
        prompt,
        guidance_scale: guidanceScale,
        steps,
        ...nimRequestParams(resolution, numFrames, fps),
        seed
      };
      if (generatorMode === "Image-to-Video") payload.image = visionPath ? "<base64 image omitted>" : undefined;
      if (generatorMode === "Action Policy") payload.video = visionPath ? "<base64 video omitted>" : undefined;
      return {
        endpoint: `POST ${backendInfo?.infer_url || "/v1/infer"}`,
        collection,
        model,
        mode: generatorMode,
        payload
      };
    }

    return {
      endpoint: "POST /generate",
      collection,
      model,
      mode: generatorMode,
      payload: {
        name: "ui-<auto>",
        model,
        prompt,
        vision_path: visionPath,
        ...params
      }
    };
  }, [
    activeExample.params,
    backendInfo?.infer_url,
    collection,
    fps,
    generatorMode,
    guidanceScale,
    isNimBackend,
    media?.dataUrl,
    media?.sourceUrl,
    model,
    numFrames,
    prompt,
    resolution,
    schemaMode,
    seed,
    steps
  ]);

  async function loadFile(file: File | null) {
    if (!file) return;
    if (generatorMode === "Image-to-Video" && !file.type.startsWith("image/")) {
      setResult({
        error: "Image-to-Video expects a JPG, PNG, or WebP image.",
        diagnostic: { layer: "frontend", issue: "The selected file was not an image." }
      });
      setStatus("Wrong input type");
      setOutputTab("preview");
      setMobilePanel("output");
      return;
    }
    const kind = file.type.startsWith("image/") ? "image" : "video";
    setMedia({
      name: file.name,
      kind,
      previewUrl: URL.createObjectURL(file),
      dataUrl: await readFileAsDataUrl(file)
    });
    setResult(null);
    setProgressPercent(0);
    setStatus("Conditioning media loaded");
  }

  async function handleFile(event: ChangeEvent<HTMLInputElement>) {
    await loadFile(event.target.files?.[0] ?? null);
  }

  function setMode(mode: GeneratorMode) {
    setGeneratorMode(mode);
    setMedia(null);
    setProgressFrames([]);
    setProgressPercent(0);
    setResult(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  function applyExample(example: ExampleItem) {
    setSelectedExampleId(example.id);
    setGeneratorMode(example.mode);
    setPrompt(example.prompt);
    setResolution(example.params?.resolution ?? QUICK_VIDEO_PARAMS.resolution);
    setNumFrames(example.params?.numFrames ?? QUICK_VIDEO_PARAMS.frames_count);
    setFps(example.params?.fps ?? QUICK_VIDEO_PARAMS.frames_per_sec);
    setSteps(example.params?.steps ?? QUICK_VIDEO_PARAMS.num_steps);
    setGuidanceScale(example.params?.guidance ?? QUICK_VIDEO_PARAMS.guidance);
    setSeed(example.params?.seed ?? 0);
    setMedia(
      example.mediaUrl
        ? {
            name: example.mediaName ?? "example asset",
            kind: example.mediaKind ?? "image",
            previewUrl: example.mediaUrl,
            sourceUrl: example.mediaUrl
          }
        : null
    );
    setResult(null);
    setProgressPercent(0);
    setStatus(`${example.eyebrow} example loaded`);
    setOutputTab("preview");
    setMobilePanel("input");
    setExamplesOpen(false);
    if (inputRef.current) inputRef.current.value = "";
  }

  function reset() {
    setCollection("cosmos3");
    setModel(DEFAULT_MODEL);
    setGeneratorMode("Text-to-Video");
    setSchemaMode("local_nim");
    setMedia(null);
    setPrompt(DEFAULT_PROMPT);
    setResolution(QUICK_VIDEO_PARAMS.resolution);
    setNumFrames(QUICK_VIDEO_PARAMS.frames_count);
    setFps(QUICK_VIDEO_PARAMS.frames_per_sec);
    setGuidanceScale(QUICK_VIDEO_PARAMS.guidance);
    setSteps(QUICK_VIDEO_PARAMS.num_steps);
    setSeed(0);
    setSelectedExampleId(EXAMPLES[0].id);
    setStatus("Ready");
    setProgressPercent(0);
    setProgressFrames([]);
    setResult(null);
    setOutputTab("preview");
    if (inputRef.current) inputRef.current.value = "";
  }

  async function copyJson() {
    await navigator.clipboard.writeText(JSON.stringify(outputTab === "json" ? { request: requestPreview, response: result } : requestPreview, null, 2));
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  async function handleDrop(event: DragEvent<HTMLButtonElement>) {
    event.preventDefault();
    setDragActive(false);
    await loadFile(event.dataTransfer.files?.[0] ?? null);
  }

  async function run() {
    if (mediaRequired && !media) {
      setResult({
        error: "Add a conditioning image or choose an example before generating.",
        diagnostic: {
          layer: "frontend",
          issue: `${generatorMode} requires a vision_path.`,
          suggestions: ["Open Examples and choose the I2V sample, or upload a local image."]
        }
      });
      setStatus("Missing input asset");
      setOutputTab("preview");
      setMobilePanel("output");
      return;
    }

    setSubmittedAt(Date.now());
    setElapsedSeconds(0);
    setIsRunning(true);
    setResult(null);
    setProgressPercent(0);
    setStatus("Predicting frames");
    setOutputTab("preview");
    setMobilePanel("output");
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mediaDataUrl: media?.dataUrl,
          visionPath: media?.sourceUrl,
          mediaKind: media?.kind,
          mode: generatorMode,
          prompt,
          model,
          guidanceScale,
          steps,
          resolution,
          numFrames,
          fps,
          seed,
          actionMode: generatorMode === "Action Policy" ? activeExample.params?.actionMode ?? "policy" : undefined,
          domainName: generatorMode === "Action Policy" ? activeExample.params?.domainName : undefined,
          imageSize: generatorMode === "Action Policy" ? activeExample.params?.imageSize : undefined,
          actionChunkSize: generatorMode === "Action Policy" ? activeExample.params?.actionChunkSize : undefined,
          rawActionDim: generatorMode === "Action Policy" ? activeExample.params?.rawActionDim : undefined,
          shift: generatorMode === "Action Policy" ? activeExample.params?.shift : undefined
        })
      });

      const RESULT_SENTINEL = "\n\n---PREDICT-RESULT---\n";
      let buffer = "";
      const reader = response.body?.getReader();
      if (reader) {
        const decoder = new TextDecoder();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
        }
        buffer += decoder.decode();
      } else {
        buffer = await response.text();
      }

      const sentinelIdx = buffer.lastIndexOf(RESULT_SENTINEL);
      let data: ApiResult;
      if (sentinelIdx >= 0) {
        const jsonText = buffer.slice(sentinelIdx + RESULT_SENTINEL.length).trim();
        try {
          data = JSON.parse(jsonText) as ApiResult;
        } catch (parseError) {
          data = {
            error: "Response payload was not valid JSON.",
            diagnostic: {
              parseError: parseError instanceof Error ? parseError.message : String(parseError),
              tail: jsonText.slice(-512)
            }
          };
        }
      } else {
        try {
          data = JSON.parse(buffer.trim()) as ApiResult;
        } catch {
          data = {
            error: "Predict response missing the streaming sentinel.",
            diagnostic: { bufferTail: buffer.slice(-512) }
          };
        }
      }

      setResult(data);
      setProgressPercent(100);
      setStatus(response.ok && !data.error ? "Complete" : "Backend error");
    } catch (error) {
      setResult({ error: error instanceof Error ? error.message : "Request failed" });
      setProgressPercent(100);
      setStatus("Request failed");
    } finally {
      setIsRunning(false);
      setSubmittedAt(null);
    }
  }

  return (
    <main>
      <header className="appbar">
        <div className="brandLockup">
          <button className="mobileMenu" aria-label="Open Menu">
            <Menu size={21} />
          </button>
          <a className="nvidiaMark" href="#" aria-label="NVIDIA">
            <img src="https://build.nvidia.com/nvidia-logo.png" alt="NVIDIA" />
          </a>
          <nav>
            {["Explore", "Models", "Blueprints", "GPUs", "Docs"].map((item) => (
              <a href="#" key={item}>
                {item}
                {item === "Docs" ? <ExternalLink size={13} /> : null}
              </a>
            ))}
          </nav>
        </div>
        <div className="appbarActions">
          <button className="searchButton">
            <Search size={15} />
            <span>Search</span>
            <kbd>Ctrl K</kbd>
          </button>
          <button className="iconButton" aria-label="Help">
            <HelpCircle size={18} />
          </button>
          <button className="loginButton">Login</button>
        </div>
      </header>

      <section className="hero predictHero">
        <div className="heroArt">
          <img src={HERO_IMAGE} alt="" />
        </div>
        <div className="heroCopy">
          <a className="publisher" href="#">
            nvidia
          </a>
          <div className="titleLine">
            <h1>{model}</h1>
            <div className="heroMeta">
              <span>Downloadable</span>
            </div>
          </div>
          <p>
            Cosmos Generator stages fast text-to-video and image-to-video predictions for physical AI simulation and
            synthetic data workflows.
          </p>
          <div className="tagRow">
            {HERO_TAGS.map((tag, index) => (
              <span className={index > 1 ? "desktopOnlyTag" : ""} key={tag}>
                {tag}
              </span>
            ))}
            <button className="mobileMoreTags">+6</button>
          </div>
        </div>
      </section>

      <RuntimeDetailsToggle
        backendInfo={backendInfo}
        detailsUrl={`${window.location.origin}/api/active-model`}
        model={model}
        open={runtimeOpen}
        setOpen={setRuntimeOpen}
      />

      <div className="tabs" role="tablist" aria-label="Model sections">
        <button className="active">Experience</button>
      </div>

      <section className="experience">
        <div className="aiNotice">
          <Info size={16} />
          <span className="noticeDesktop">
            AI-generated videos may be inaccurate, biased, unsafe, or physically inconsistent. Review generated content
            before using it for simulation, training, or dataset approval workflows.
          </span>
          <span className="noticeMobile">AI Response Message</span>
          <button className="noticeView">View</button>
        </div>

        <div className="mobileIOTabs" aria-label="Mobile workspace tabs">
          <button className={mobilePanel === "input" ? "active" : ""} onClick={() => setMobilePanel("input")}>
            Input
          </button>
          <button className={mobilePanel === "output" ? "active" : ""} onClick={() => setMobilePanel("output")}>
            Output
          </button>
        </div>

        <div className="workspace">
          <section className={`panel inputPanel ${mobilePanel === "input" ? "mobileActivePanel" : ""}`}>
            <div className="panelHeader">
              <h2>Input</h2>
              <button className="secondaryAction" onClick={() => setExamplesOpen(true)}>
                View Examples <ChevronDown size={14} />
              </button>
            </div>

            <label className="fieldLabel">Generator Mode</label>
            <div className="modeGrid">
              {visibleModes.map((mode) => {
                const Icon = mode.icon;
                return (
                  <button
                    className={`${generatorMode === mode.id ? "active" : ""} ${!mode.enabled ? "disabledMode" : ""}`}
                    key={mode.id}
                    onClick={() => mode.enabled && setMode(mode.id)}
                    disabled={!mode.enabled}
                  >
                    <Icon size={16} />
                    <span>{mode.label}</span>
                    <small>{mode.description}</small>
                  </button>
                );
              })}
            </div>

            <label className="fieldLabel">Input</label>
            {mediaRequired ? (
              <button
                className={`dropzone predictDropzone ${media ? "hasMedia" : ""} ${dragActive ? "dragActive" : ""}`}
                onClick={() => inputRef.current?.click()}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragActive(true);
                }}
                onDragLeave={(event) => {
                  event.preventDefault();
                  setDragActive(false);
                }}
                onDrop={handleDrop}
                type="button"
              >
                <input ref={inputRef} type="file" accept={accepts} onChange={handleFile} hidden />
                {media ? (
                  <span className="mediaLoaded mediaPreview">
                    {media.kind === "image" ? <img src={media.previewUrl} alt="" /> : <video src={media.previewUrl} muted playsInline />}
                    <span>
                      <strong>{media.name}</strong>
                      <small>{media.sourceUrl ? "Remote example asset" : `Upload staged for ${isNimBackend ? "NIM" : "Ray Serve"}`}</small>
                    </span>
                  </span>
                ) : (
                  <>
                    <Upload size={22} />
                    <span>{generatorMode === "Image-to-Video" ? "Drop source image here" : "Drop conditioning asset here"}</span>
                    <small>{accepts}</small>
                  </>
                )}
              </button>
            ) : (
              <div className="textOnlyNotice">
                <FileVideo size={22} />
                <div>
                  <strong>Prompt-only generation</strong>
                  <span>{isNimBackend ? "NIM receives this request without conditioning media." : "Cosmos3 receives this request without a vision_path."}</span>
                </div>
              </div>
            )}

            <PromptBox
              label="Prompt"
              hint="Describe the future world state to generate."
              max={1000}
              value={prompt}
              onChange={setPrompt}
              rows={4}
            />

            <div className="parameterGrid predictParams">
              <SelectField label="Resolution" value={resolution} options={["256", "480", "720", "1080"]} onChange={setResolution} />
              <NumberField label="Guidance" value={guidanceScale} min={1} max={12} step={0.5} onChange={setGuidanceScale} />
              <NumberField label="Steps" value={steps} min={1} max={80} step={1} onChange={setSteps} />
              <NumberField label="Frames" value={numFrames} min={8} max={121} step={1} onChange={setNumFrames} />
              <NumberField label="FPS" value={fps} min={1} max={30} step={1} onChange={setFps} />
              <NumberField label="Seed" value={seed} min={-1} max={2147483647} step={1} onChange={setSeed} />
            </div>

            <div className="runBar">
              <button className="resetButton" onClick={reset}>
                <RotateCcw size={16} />
                Reset
              </button>
              <span
                className="etaBadge"
                title={`Estimator from RTX PRO 6000 Blackwell smoke runs. ${etaSeconds}s for ${numFrames} frames at ${resolution} resolution x ${steps} steps.`}
              >
                Est. wall ~{formatDuration(etaSeconds)}
              </span>
              <button className="runButton" onClick={run} disabled={isRunning || (mediaRequired && !media)}>
                <Play size={16} fill="currentColor" />
                {isRunning ? "Generating" : "Generate"}
              </button>
            </div>
          </section>

          <section className={`panel outputPanel ${mobilePanel === "output" ? "mobileActivePanel" : ""}`}>
            <div className="panelHeader">
              <div className="outputTabs">
                <h2>Output</h2>
                <button className={outputTab === "preview" ? "previewPill activeOutputTab" : "previewPill"} onClick={() => setOutputTab("preview")}>
                  Preview
                </button>
                <button className={outputTab === "json" ? "jsonTab activeOutputTab" : "jsonTab"} onClick={() => setOutputTab("json")}>
                  JSON
                </button>
              </div>
              <div className="outputActions">
                <span className={`statusPill ${isRunning ? "working" : result?.error ? "error" : status === "Complete" ? "success" : ""}`}>
                  {status}
                </span>
                <button className="copyButton" onClick={copyJson}>
                  <Copy size={14} />
                  {copied ? "Copied" : "Copy"}
                </button>
              </div>
            </div>
            <div className="outputBody">
              {outputTab === "json" ? (
                <pre className="jsonOutput">{JSON.stringify({ request: requestPreview, response: result }, null, 2)}</pre>
              ) : isRunning ? (
                <GenerationProgress
                  media={media}
                  frames={progressFrames}
                  progress={progressPercent}
                  elapsedSeconds={elapsedSeconds}
                  etaSeconds={etaSeconds}
                  totalFrames={numFrames}
                />
              ) : result?.error ? (
                <FailureReport result={result} />
              ) : result?.videoDataUrl ? (
                <video className="resultVideo" src={result.videoDataUrl} controls />
              ) : result?.imageDataUrl ? (
                <img className="resultVideo" src={result.imageDataUrl} alt="Generated world state" />
              ) : assetUrl ? (
                <a className="assetLink" href={assetUrl} target="_blank">
                  Open generated asset
                  <ExternalLink size={15} />
                </a>
              ) : (
                <WorldPreview mode={generatorMode} />
              )}
            </div>
          </section>
        </div>

        <aside className="apiPanel">
          <div className="apiTopline">API request</div>
          <div className="apiButtons">
            <button onClick={copyJson}>
              <Copy size={14} />
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
          <div className="apiControls">
            <label>
              Collection
              <select value={collection} onChange={(event) => setCollection(event.target.value)}>
                {COLLECTIONS.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Model
              <select value={model} onChange={(event) => setModel(event.target.value)}>
                {models.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Contract
              <select value={schemaMode} onChange={(event) => setSchemaMode(event.target.value as SchemaMode)}>
                <option value="local_nim">Local NIM /v1/infer</option>
                <option value="build_openapi">NVIDIA Build OpenAPI preview</option>
              </select>
            </label>
          </div>
          <pre className="codeBlock">{JSON.stringify(requestPreview, null, 2)}</pre>
          {backendInfo?.warning ? <p className="backendWarning">{backendInfo.warning}</p> : null}
        </aside>
      </section>

      {examplesOpen ? (
        <ExampleModal examples={visibleExamples} onClose={() => setExamplesOpen(false)} onSelect={applyExample} />
      ) : null}
    </main>
  );
}

function FailureReport({ result }: { result: ApiResult }) {
  const diagnostic = result.diagnostic ?? {};
  const layer = typeof diagnostic.layer === "string" ? diagnostic.layer : "unknown";
  const issue = typeof diagnostic.issue === "string" ? diagnostic.issue : result.error ?? "Request failed";
  const likelyCause = typeof diagnostic.likelyCause === "string" ? diagnostic.likelyCause : "No structured cause was returned.";
  const endpoint = typeof diagnostic.endpoint === "string" ? diagnostic.endpoint : null;
  const suggestions = Array.isArray(diagnostic.suggestions) ? diagnostic.suggestions.map(String) : [];
  const layerLabel =
    layer === "backend" ? "Backend" : layer === "frontend" ? "Frontend/API" : layer === "parameters" ? "Parameters" : "Unknown";

  return (
    <article className="failureReport">
      <div className="failureTopline">
        <span>Failure source</span>
        <strong>{layerLabel}</strong>
      </div>
      <h3>{result.error}</h3>
      <dl>
        <div>
          <dt>Why it failed</dt>
          <dd>{issue}</dd>
        </div>
        <div>
          <dt>Likely cause</dt>
          <dd>{likelyCause}</dd>
        </div>
        {endpoint ? (
          <div>
            <dt>Backend endpoint</dt>
            <dd>{endpoint}</dd>
          </div>
        ) : null}
      </dl>
      {suggestions.length > 0 ? (
        <div className="failureSuggestions">
          <span>What to check next</span>
          <ul>
            {suggestions.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}
      <details>
        <summary>Diagnostic JSON</summary>
        <pre>{JSON.stringify({ diagnostic, payload: result.payload, raw: result.raw }, null, 2)}</pre>
      </details>
    </article>
  );
}

function GenerationProgress({
  media,
  frames,
  progress,
  elapsedSeconds,
  etaSeconds,
  totalFrames
}: {
  media: MediaState | null;
  frames: string[];
  progress: number;
  elapsedSeconds: number;
  etaSeconds: number;
  totalFrames: number;
}) {
  const generatedFrames = Math.max(1, Math.min(totalFrames, Math.round((progress / 100) * totalFrames)));
  const displayFrames = frames.length > 0 ? frames : Array.from({ length: PROGRESS_FRAME_COUNT }, () => "");
  const fallbackLabel = media?.kind === "image" ? "Image condition" : media?.kind === "video" ? "Video condition" : "Text prompt";
  const remainingSeconds = Math.max(0, etaSeconds - elapsedSeconds);
  const overrun = elapsedSeconds > etaSeconds;

  return (
    <article className="generationProgress" aria-live="polite">
      <div className="progressHeader">
        <p>
          <strong>The diffusion model is working:</strong> denoising {totalFrames} latent frames in parallel, then
          VAE-decoding and encoding the output.
        </p>
        <span>
          {generatedFrames} / {totalFrames} frames - elapsed {formatDuration(elapsedSeconds)}
          {overrun ? ` - over est. by ${formatDuration(elapsedSeconds - etaSeconds)}` : ` - ETA ${formatDuration(remainingSeconds)}`}
        </span>
      </div>
      <div
        className="progressTrack"
        aria-label="Generation progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(progress)}
        role="progressbar"
      >
        <span style={{ width: `${Math.max(4, progress)}%` }} />
      </div>
      <div className="generatedFrameStrip">
        {displayFrames.map((src, index) => {
          const threshold = (index / PROGRESS_FRAME_COUNT) * 86;
          const readiness = Math.max(0, Math.min(1, (progress - threshold) / 34));
          const frameNumber = Math.min(totalFrames, Math.max(1, Math.round(((index + 1) / PROGRESS_FRAME_COUNT) * totalFrames)));
          const style = {
            "--frame-blur": `${Math.max(0, 10 - readiness * 10)}px`,
            "--frame-opacity": String(0.38 + readiness * 0.62)
          } as CSSProperties;

          return (
            <figure className="generatedFrame" style={style} key={`${src || "fallback"}-${index}`}>
              {src ? <img src={src} alt="" /> : <div className="generatedFrameFallback" />}
              <figcaption>Frame {frameNumber}</figcaption>
            </figure>
          );
        })}
      </div>
      <p className="progressFootnote">
        Conditioning source: {media?.name || fallbackLabel}. The final MP4 replaces this preview as soon as inference completes.
      </p>
    </article>
  );
}

function WorldPreview({ mode }: { mode: GeneratorMode }) {
  const labels = mode === "Text-to-Video" ? ["Prompt", "Latent rollout", "Video"] : ["Condition", "Latent rollout", "Future frames"];
  return (
    <article className="worldPreview">
      <div className="worldFrameGrid">
        {labels.map((label, index) => (
          <div className="worldFrame" key={label}>
            <span>{String(index + 1).padStart(2, "0")}</span>
            <strong>{label}</strong>
          </div>
        ))}
      </div>
      <h3>Generated Future World</h3>
      <p>
        Run a quick T2V prompt or add an I2V conditioning image. The local Ray Serve path returns an inline video when
        output files are readable, plus raw JSON for diagnostics.
      </p>
    </article>
  );
}

function PromptBox({
  label,
  hint,
  value,
  max,
  rows,
  onChange
}: {
  label: string;
  hint: string;
  value: string;
  max: number;
  rows: number;
  onChange: (value: string) => void;
}) {
  return (
    <div className="promptBox">
      <div className="promptTopline">
        <label>
          {label}
          <span className="requiredDot">*</span>
          <span className="infoDot">i</span>
        </label>
        <span>
          {value.length}/{max}
        </span>
      </div>
      <textarea
        rows={rows}
        value={value}
        maxLength={max}
        placeholder={hint}
        onChange={(event) => onChange(event.target.value)}
      />
      <small>{hint}</small>
    </div>
  );
}

function SelectField({
  label,
  value,
  options,
  onChange
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="numberField">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  step,
  onChange
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="numberField">
      <span>{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function RuntimeDetailsToggle({
  backendInfo,
  detailsUrl,
  model,
  open,
  setOpen
}: {
  backendInfo: BackendInfo | null;
  detailsUrl: string;
  model: string;
  open: boolean;
  setOpen: (open: boolean) => void;
}) {
  return (
    <section className="runtimeShell" aria-label="Active model details">
      <button
        aria-controls="active-model-runtime-details"
        aria-expanded={open}
        className="runtimeToggle"
        onClick={() => setOpen(!open)}
        type="button"
      >
        Active Model details
        {open ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
      </button>
      {open ? <RuntimeBar backendInfo={backendInfo} detailsUrl={detailsUrl} model={model} /> : null}
    </section>
  );
}

function RuntimeBar({
  backendInfo,
  detailsUrl,
  model
}: {
  backendInfo: BackendInfo | null;
  detailsUrl: string;
  model: string;
}) {
  const environment = backendInfo?.environment;
  const commitSha = environment?.commit_sha;
  const isNim = String(backendInfo?.backend || "").toLowerCase().includes("nim");
  const staged = backendInfo?.staged_checkpoint;
  const nimImage = backendInfo?.image || staged?.image;
  const capabilities = backendInfo?.capabilities
    ? Object.entries(backendInfo.capabilities)
        .filter(([, enabled]) => enabled)
        .map(([key]) => key.replace(/_/g, "-"))
        .join(", ")
    : "";

  return (
    <div className="runtimeBar" id="active-model-runtime-details">
      {backendInfo?.warning ? (
        <div className="runtimeWarning">
          <Info size={15} />
          <span>{backendInfo.warning}</span>
        </div>
      ) : null}
      <div className="runtimeMetric">
        <span className="runtimeLabel">Live Model</span>
        <span className="runtimeValue">{displayValue(backendInfo?.display_name || backendInfo?.checkpoint || model)}</span>
      </div>
      <div className="runtimeMetric">
        <span className="runtimeLabel">Backend</span>
        <span className="runtimeValue">
          {displayValue(backendInfo?.backend || "cosmos3-generate")} at {displayValue(backendInfo?.base_url)}
        </span>
      </div>
      {nimImage ? (
        <div className="runtimeMetric wideRuntimeMetric">
          <span className="runtimeLabel">NIM Image</span>
          <span className="runtimeValue">{displayValue(nimImage)}</span>
        </div>
      ) : null}
      {backendInfo?.infer_url ? (
        <div className="runtimeMetric">
          <span className="runtimeLabel">Infer URL</span>
          <span className="runtimeValue">{displayValue(backendInfo.infer_url)}</span>
        </div>
      ) : null}
      <div className="runtimeMetric">
        <span className="runtimeLabel">Cosmos3</span>
        <span className="runtimeValue">{displayValue(backendInfo?.cosmos3_version || environment?.cosmos3_version)}</span>
      </div>
      <div className="runtimeMetric">
        <span className="runtimeLabel">{isNim ? "Staged Model" : "Ray Commit"}</span>
        <span className="runtimeValue">
          {isNim ? displayValue(staged?.served_model || backendInfo?.checkpoint) : <code>{shortSha(commitSha)}</code>}
        </span>
      </div>
      <div className="runtimeMetric">
        <span className="runtimeLabel">Output Dir</span>
        <span className="runtimeValue">{displayValue(backendInfo?.output_dir)}</span>
      </div>
      <div className="runtimeMetric">
        <span className="runtimeLabel">Capabilities</span>
        <span className="runtimeValue">{displayValue(capabilities)}</span>
      </div>
      <a className="runtimeJsonLink" href={detailsUrl} rel="noreferrer" target="_blank">
        Full JSON details
        <ExternalLink size={13} />
      </a>
    </div>
  );
}

function ExampleModal({
  examples,
  onClose,
  onSelect
}: {
  examples: ExampleItem[];
  onClose: () => void;
  onSelect: (example: ExampleItem) => void;
}) {
  return (
    <div className="modalBackdrop" role="presentation" onMouseDown={onClose}>
      <div
        className="examplesModal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="examples-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="modalHeader">
          <div>
            <p>Examples</p>
            <h2 id="examples-title">Cosmos3 Generator presets</h2>
          </div>
          <button className="iconButton" aria-label="Close examples" onClick={onClose}>
            <X size={18} />
          </button>
        </div>
        <div className="exampleList">
          {examples.map((example) => (
            <button key={example.id} className="exampleItem" onClick={() => onSelect(example)}>
              <div className="exampleThumb">
                {example.mediaUrl ? (
                  example.mediaKind === "video" ? (
                    <video src={example.mediaUrl} muted playsInline />
                  ) : (
                    <img src={example.mediaUrl} alt="" />
                  )
                ) : (
                  <FileVideo size={28} />
                )}
              </div>
              <div className="exampleText">
                <span>{example.eyebrow}</span>
                <strong>{example.label}</strong>
                <p>{example.prompt}</p>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
