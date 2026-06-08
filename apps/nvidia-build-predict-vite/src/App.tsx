"use client";

import { ChevronDown, ChevronLeft, ChevronRight, ExternalLink, FileVideo, HelpCircle, Info, Menu, Play, RotateCcw, Search, Upload, X } from "lucide-react";
import { ChangeEvent, CSSProperties, DragEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { CANONICAL_LONG_PROMPTS } from "./data/canonicalLongPrompts";

const HERO_IMAGE = "https://assets.ngc.nvidia.com/products/api-catalog/images/cosmos-predict1-5b.jpg";
const BUILD_PREDICT_MODEL_CARD_URL = "https://build.nvidia.com/nvidia/cosmos-predict1-5b/modelcard";
const BUILD_PREDICT_SYSTEM_CARD_URL = "https://build.nvidia.com/nvidia/cosmos-predict1-5b/systemcard";
const MODEL_CARD_TITLE = "Image-to-World";
const PAGE_TAGLINE = "Generates future frames based upon an image and text input.";
const MODEL_CARD_LEAD =
  "Generates future frames of a physics-aware world state based on simply an image or short video along with a text prompt for physical AI development.";
const DISPLAY_MODEL_FALLBACK = "Cosmos3-Nano";
const CURATED_PREVIEW_VIDEO = "/examples/race-car.mp4";
const QUICK_VIDEO_PARAMS = {
  resolution: "256",
  aspect_ratio: "16,9",
  frames_count: 121,
  frames_per_sec: 24,
  num_steps: 50,
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
  DISPLAY_MODEL_FALLBACK;
const HERO_TAGS = [
  "physical ai",
  "world foundation model",
  "robotics",
  "simulation",
  "synthetic data generation",
  "image-conditioned video",
  "image-to-world",
  "future state generation"
];
const PROGRESS_FRAME_COUNT = 6;
const COMMUNITY_VIDEOS = [
  "/examples/canonical/007.mp4",
  "/examples/canonical/009.mp4",
  "/examples/canonical/011.mp4"
];

const ETA_PIXELS_BY_RESOLUTION: Record<string, number> = {
  "256": 256 * 256,
  "480": 854 * 480,
  "720": 1280 * 720,
  "1080": 1920 * 1080
};
const ETA_BASELINE_SECONDS = 5;
const ETA_OPS_PER_SECOND = 10_000_000;
const NIM_ETA_BASELINE_SECONDS = 24;
const NIM_ETA_OPS_PER_SECOND = 2_500_000;

type GeneratorMode = "Image-to-Video" | "Action Policy";
type SectionTab = "Experience" | "Model Card" | "System Card";
type MobilePanel = "input" | "output";
type MediaState = {
  name: string;
  kind: "video" | "image";
  previewUrl: string;
  dataUrl?: string;
  sourceUrl?: string;
};
type GenerationParams = {
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
  negativePrompt?: string;
  conditionFrameIndexesVision?: number[];
  conditionVideoKeep?: string;
  modelMode?: string;
  sigmaMax?: number;
  guidanceInterval?: number | null;
  normalizeCfg?: boolean;
  videoSaveQuality?: number;
  imageSaveQuality?: number;
  enableSound?: boolean;
  negativeMetadataMode?: string;
  negativePromptKeepMetadata?: boolean;
  numOutputs?: number;
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
  params?: GenerationParams;
};
type ContentSelectItem = {
  id: string;
  title: string;
  domain: string;
  description: string;
  shortPrompt?: string;
  prompt: string;
  mediaUrl: string;
  mediaName: string;
  previewVideoUrl: string;
  previewVideoName: string;
  params?: GenerationParams;
};
type PromptChoice = "short" | "long";
type ContentSelectGroup = {
  title: string;
  summary: string;
  items: ContentSelectItem[];
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
  budget?: BudgetInfo;
};
type BudgetSuggestion = {
  resolution: string;
  num_frames: number;
  num_steps: number;
  est_seconds?: number;
};
type BudgetInfo = {
  suggestions?: BudgetSuggestion[];
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
  served_model?: string;
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

const CANONICAL_SAMPLE_NEGATIVE_PROMPT =
  "The video captures a series of frames showing macroblocking artifacts, chromatic aberration, high-frequency noise, and rolling shutter distortion. It includes static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, bit-depth compression artifacts, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Avoid moire patterns, edge halos, and temporal aliasing. Furthermore, the content defies common sense, generating illogical scenarios, nonsensical entities, absurd character behaviors, and conceptual paradoxes that violate basic human reasoning and everyday reality. The video looks like a surreal or glitchy hallucination. Overall, the video is of poor quality.";
const ASSEMBLY_LINE_NEGATIVE_PROMPT =
  "The video captures a series of frames showing macroblocking artifacts, chromatic aberration, high-frequency noise, and rolling shutter distortion. It includes static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, bit-depth compression artifacts, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Avoid moire patterns, edge halos, temporal aliasing, floating parts, newly appearing objects, detached car panels, deformed robot arms, extra grippers, warped geometry, melting metal, and hallucinated tools. Overall, the video is of poor quality.";
const CANONICAL_SAMPLE_PARAMS: GenerationParams = {
  resolution: "720",
  numFrames: 189,
  fps: 24,
  steps: 35,
  guidance: 6,
  shift: 10,
  imageSize: 256,
  conditionFrameIndexesVision: [0],
  conditionVideoKeep: "first",
  modelMode: "image2video",
  sigmaMax: 80,
  guidanceInterval: null,
  normalizeCfg: false,
  videoSaveQuality: 10,
  imageSaveQuality: 95,
  enableSound: false,
  negativeMetadataMode: "same",
  negativePromptKeepMetadata: true,
  numOutputs: 1,
  negativePrompt: CANONICAL_SAMPLE_NEGATIVE_PROMPT
};
const SAMPLE_PARAMS_BY_ID: Record<"001" | "005" | "007" | "009" | "021" | "022" | "012" | "085", GenerationParams> = {
  "001": { ...CANONICAL_SAMPLE_PARAMS, seed: 400 },
  "005": { ...CANONICAL_SAMPLE_PARAMS, seed: 403 },
  "007": { ...CANONICAL_SAMPLE_PARAMS, seed: 202 },
  "009": { ...CANONICAL_SAMPLE_PARAMS, seed: 201 },
  "021": { ...CANONICAL_SAMPLE_PARAMS, seed: 301 },
  "022": { ...CANONICAL_SAMPLE_PARAMS, seed: 200 },
  "012": {
    ...CANONICAL_SAMPLE_PARAMS,
    seed: 109,
    negativePrompt: ASSEMBLY_LINE_NEGATIVE_PROMPT
  },
  "085": { ...CANONICAL_SAMPLE_PARAMS, seed: 109 }
};

const CANONICAL_PROMPTS = CANONICAL_LONG_PROMPTS;

const CANONICAL_SHORT_PROMPTS: Partial<Record<keyof typeof CANONICAL_PROMPTS, string>> = {
  "001":
    "Static locked-off overhead camera looking down at a clean white laboratory workbench. A black robotic arm with a two-finger gripper descends to the yellow-green pear at center, closes around it, lifts it cleanly, translates laterally toward the dark bowl in the upper-left of the workspace, lowers, and releases the pear into the bowl. The arm retracts back to a neutral position. The pink object stays untouched. Bright even lab lighting; sharp soft-shadow under each object.\n\n(Camera motion is suppressed via the separate negative prompt.)",
  "005":
    "Static locked-off overhead camera looking down at a wooden workshop tabletop. The right humanoid robotic arm extends to the bok choy, the gripper closes firmly around the pale white stem at the base of the bok choy (gripping the stem, not the green leafy top), lifts it cleanly off the table holding it by its base stem, carries it left across the table directly over the white frying pan, and lowers and releases the bok choy down into the frying pan so it settles inside the pan with its leaves splaying upward. The left arm stays still throughout. Even workshop lighting.\n\n(Camera motion is suppressed via the separate negative prompt.)",
  "007":
    "A black robotic arm descends to the toy salmon sashimi at front-left of the cutting board, the gripper closes around it, lifts it slightly, moves it right toward the orange bowl, and releases it into the bowl. The toy tomato and toy carrot stay untouched throughout. Static medium shot; even, bright workshop lighting.",
  "009":
    "A dashcam view from an ego vehicle approaching a suburban residential intersection. The ego slows and comes to a stop, yielding to an oncoming vehicle while waiting to turn right. A white car finishes a right turn and exits the frame to the left. Calm, bright, late-winter daylight. The camera is a fixed dashcam; the scene stays calm and typical.",
  "021":
    "A dashcam view from an ego vehicle approaching a signalized intersection on a multi-lane suburban arterial. The traffic lights facing the ego are red for most of the video. The ego decelerates from a moderate forward pace, gradually slowing as it closes distance. At about 3/4 of the way through (around 0:06), the signals visibly change from red to green, and the ego accelerates again toward the intersection. A white van directly ahead mirrors the ego's slow-then-accelerate motion. Warm golden late-afternoon light throughout; long shadows. The camera is a fixed dashcam.",
  "022":
    "A dashcam view from an ego vehicle in a dedicated left-turn lane executing a continuous queued left turn through a multi-lane intersection. The traffic signals facing the ego stay green throughout. The white truck at the front of the queue arcs left into the cross-street, the grey sedan immediately ahead follows, and the ego follows the sedan around the turn in one continuous flowing motion. Bright midday daylight; soft short shadows. The camera is a fixed dashcam."
};

const EXAMPLES: ExampleItem[] = [
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
const CONTENT_SELECT_GROUPS: ContentSelectGroup[] = [
  {
    title: "Robotics",
    summary: "Robot manipulation scenes with tabletop objects, grippers, and goal-directed motion.",
    items: [
      {
        id: "robot-cutting-board-sorting",
        title: "Robot Cutting Board Sorting",
        domain: "Robotics",
        description: "A tabletop robot scene with food items, a bowl, and a gripper in a workshop setting.",
        shortPrompt: CANONICAL_SHORT_PROMPTS["007"],
        prompt: CANONICAL_PROMPTS["007"],
        mediaUrl: "/examples/canonical/007.png",
        mediaName: "Robot Cutting Board Sorting.png",
        previewVideoUrl: "/examples/canonical/007.mp4",
        previewVideoName: "Robot Cutting Board Sorting.mp4",
        params: SAMPLE_PARAMS_BY_ID["007"]
      },
      {
        id: "robot-pear-bowl-placement",
        title: "Robot Pear Bowl Placement",
        domain: "Robotics",
        description: "A robot manipulation setup with a pear and a dark bowl.",
        shortPrompt: CANONICAL_SHORT_PROMPTS["001"],
        prompt: CANONICAL_PROMPTS["001"],
        mediaUrl: "/examples/canonical/001.png",
        mediaName: "Robot Pear Bowl Placement.png",
        previewVideoUrl: "/examples/canonical/001.mp4",
        previewVideoName: "Robot Pear Bowl Placement.mp4",
        params: SAMPLE_PARAMS_BY_ID["001"]
      },
      {
        id: "robot-bok-choy-pan-placement",
        title: "Robot Bok Choy Pan Placement",
        domain: "Robotics",
        description: "A gripper positions bok choy near a frying pan on a tabletop.",
        shortPrompt: CANONICAL_SHORT_PROMPTS["005"],
        prompt: CANONICAL_PROMPTS["005"],
        mediaUrl: "/examples/canonical/005.png",
        mediaName: "Robot Bok Choy Pan Placement.png",
        previewVideoUrl: "/examples/canonical/005.mp4",
        previewVideoName: "Robot Bok Choy Pan Placement.mp4",
        params: SAMPLE_PARAMS_BY_ID["005"]
      }
    ]
  },
  {
    title: "Autonomous Vehicles",
    summary: "Forward-facing driving scenes that exercise road layout, ego motion, and traffic context.",
    items: [
      {
        id: "suburban-intersection-yield",
        title: "Suburban Intersection Yield",
        domain: "Autonomous Vehicles",
        description: "An ego-vehicle view approaching a residential intersection under clear daylight.",
        shortPrompt: CANONICAL_SHORT_PROMPTS["009"],
        prompt: CANONICAL_PROMPTS["009"],
        mediaUrl: "/examples/canonical/009.png",
        mediaName: "Suburban Intersection Yield.png",
        previewVideoUrl: "/examples/canonical/009.mp4",
        previewVideoName: "Suburban Intersection Yield.mp4",
        params: SAMPLE_PARAMS_BY_ID["009"]
      },
      {
        id: "urban-signal-approach",
        title: "Urban Signal Approach",
        domain: "Autonomous Vehicles",
        description: "An ego-vehicle view following a minivan toward a signalized city intersection.",
        shortPrompt: CANONICAL_SHORT_PROMPTS["021"],
        prompt: CANONICAL_PROMPTS["021"],
        mediaUrl: "/examples/canonical/021.png",
        mediaName: "Urban Signal Approach.png",
        previewVideoUrl: "/examples/canonical/021.mp4",
        previewVideoName: "Urban Signal Approach.mp4",
        params: SAMPLE_PARAMS_BY_ID["021"]
      },
      {
        id: "signalized-left-turn-completion",
        title: "Signalized Left Turn Completion",
        domain: "Autonomous Vehicles",
        description: "An ego-vehicle view from a turn lane completing a left turn through an intersection.",
        shortPrompt: CANONICAL_SHORT_PROMPTS["022"],
        prompt: CANONICAL_PROMPTS["022"],
        mediaUrl: "/examples/canonical/022.png",
        mediaName: "Signalized Left Turn Completion.png",
        previewVideoUrl: "/examples/canonical/022.mp4",
        previewVideoName: "Signalized Left Turn Completion.mp4",
        params: SAMPLE_PARAMS_BY_ID["022"]
      }
    ]
  }
];
const DEFAULT_CONTENT_ITEM =
  findContentItem(CONTENT_SELECT_GROUPS, "robot-cutting-board-sorting") ?? CONTENT_SELECT_GROUPS[0].items[0];
const DEFAULT_PROMPT = DEFAULT_CONTENT_ITEM.prompt;
const DEFAULT_CONTENT_PARAMS = visibleParams(DEFAULT_CONTENT_ITEM.params);
const DEFAULT_ADVANCED_PARAMS = advancedParams(DEFAULT_CONTENT_ITEM.params);

function visibleParams(params?: GenerationParams) {
  return {
    resolution: params?.resolution ?? QUICK_VIDEO_PARAMS.resolution,
    numFrames: params?.numFrames ?? QUICK_VIDEO_PARAMS.frames_count,
    fps: params?.fps ?? QUICK_VIDEO_PARAMS.frames_per_sec,
    steps: params?.steps ?? QUICK_VIDEO_PARAMS.num_steps,
    guidance: params?.guidance ?? QUICK_VIDEO_PARAMS.guidance,
    seed: params?.seed ?? 0
  };
}

function advancedParams(params?: GenerationParams) {
  return {
    imageSize: params?.imageSize ?? CANONICAL_SAMPLE_PARAMS.imageSize ?? 256,
    shift: params?.shift ?? CANONICAL_SAMPLE_PARAMS.shift ?? 10,
    sigmaMax: params?.sigmaMax ?? CANONICAL_SAMPLE_PARAMS.sigmaMax ?? 80
  };
}

function estimateWallSeconds(resolution: string | number, numFrames: number, numSteps: number, isNimBackend: boolean): number {
  const px = ETA_PIXELS_BY_RESOLUTION[String(resolution)] || ETA_PIXELS_BY_RESOLUTION["480"];
  const f = Math.max(1, numFrames);
  const s = Math.max(1, numSteps);
  const baseline = isNimBackend ? NIM_ETA_BASELINE_SECONDS : ETA_BASELINE_SECONDS;
  const opsPerSecond = isNimBackend ? NIM_ETA_OPS_PER_SECOND : ETA_OPS_PER_SECOND;
  return Math.ceil(baseline + (px * f * s) / opsPerSecond);
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

function displayMediaName(name: string) {
  return name.replace(/\.(png|jpe?g|webp|mp4|mov|webm)$/i, "");
}

function shortSha(sha?: string | null) {
  return sha ? sha.slice(0, 12) : "unknown";
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

function contentItemToMedia(item: ContentSelectItem): MediaState {
  return {
    name: item.mediaName,
    kind: "image",
    previewUrl: item.mediaUrl,
    sourceUrl: item.mediaUrl
  };
}

function absoluteMediaUrl(url?: string | null) {
  if (!url) return undefined;
  if (/^https?:\/\//i.test(url)) return url;
  return new URL(url, window.location.origin).href;
}

function findContentItem(groups: ContentSelectGroup[], id: string) {
  return groups.flatMap((group) => group.items).find((item) => item.id === id) ?? groups[0].items[0];
}

function contentPromptForChoice(item: ContentSelectItem, choice: PromptChoice) {
  return choice === "short" && item.shortPrompt ? item.shortPrompt : item.prompt;
}

function waitForContentLoad(ms: number) {
  return new Promise<void>((resolve) => window.setTimeout(resolve, ms));
}

function preloadContentImage(url: string) {
  return new Promise<void>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve();
    image.onerror = () => reject(new Error(`Failed to load ${url}`));
    image.src = url;
  });
}

function preloadContentVideo(url: string) {
  return new Promise<void>((resolve, reject) => {
    const video = document.createElement("video");
    let settled = false;
    const done = (error?: Error) => {
      if (settled) return;
      settled = true;
      if (error) {
        reject(error);
      } else {
        resolve();
      }
    };
    video.preload = "metadata";
    video.muted = true;
    video.playsInline = true;
    video.addEventListener("loadedmetadata", () => done());
    video.addEventListener("canplay", () => done());
    video.addEventListener("error", () => done(new Error(`Failed to load ${url}`)));
    window.setTimeout(() => done(), 5000);
    video.src = url;
    video.load();
  });
}

export default function Page() {
  const inputRef = useRef<HTMLInputElement>(null);
  const contentLoadTokenRef = useRef(0);
  const [activeTab, setActiveTab] = useState<SectionTab>("Experience");
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [backendInfo, setBackendInfo] = useState<BackendInfo | null>(null);
  const [runtimeOpen, setRuntimeOpen] = useState(false);
  const [generatorMode, setGeneratorMode] = useState<GeneratorMode>("Image-to-Video");
  const [media, setMedia] = useState<MediaState | null>(() => contentItemToMedia(DEFAULT_CONTENT_ITEM));
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [resolution, setResolution] = useState(DEFAULT_CONTENT_PARAMS.resolution);
  const [numFrames, setNumFrames] = useState(DEFAULT_CONTENT_PARAMS.numFrames);
  const [fps, setFps] = useState(DEFAULT_CONTENT_PARAMS.fps);
  const [guidanceScale, setGuidanceScale] = useState(DEFAULT_CONTENT_PARAMS.guidance);
  const [steps, setSteps] = useState(DEFAULT_CONTENT_PARAMS.steps);
  const [seed, setSeed] = useState(DEFAULT_CONTENT_PARAMS.seed);
  const [imageSize, setImageSize] = useState(DEFAULT_ADVANCED_PARAMS.imageSize);
  const [shift, setShift] = useState(DEFAULT_ADVANCED_PARAMS.shift);
  const [sigmaMax, setSigmaMax] = useState(DEFAULT_ADVANCED_PARAMS.sigmaMax);
  const [status, setStatus] = useState("Ready");
  const [isRunning, setIsRunning] = useState(false);
  const [progressPercent, setProgressPercent] = useState(0);
  const [submittedAt, setSubmittedAt] = useState<number | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [progressFrames, setProgressFrames] = useState<string[]>([]);
  const [result, setResult] = useState<ApiResult | null>(null);
  const [mobilePanel, setMobilePanel] = useState<MobilePanel>("input");
  const [examplesOpen, setExamplesOpen] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [selectedExampleId, setSelectedExampleId] = useState(EXAMPLES[0].id);
  const [selectedContentItemId, setSelectedContentItemId] = useState(DEFAULT_CONTENT_ITEM.id);
  const [selectedPromptChoice, setSelectedPromptChoice] = useState<PromptChoice>("long");
  const [selectedPreviewVideo, setSelectedPreviewVideo] = useState(() => ({
    url: DEFAULT_CONTENT_ITEM.previewVideoUrl,
    name: DEFAULT_CONTENT_ITEM.previewVideoName
  }));
  const [loadingContentItemId, setLoadingContentItemId] = useState<string | null>(null);

  const activeExample = useMemo(
    () => EXAMPLES.find((example) => example.id === selectedExampleId) ?? EXAMPLES[0],
    [selectedExampleId]
  );
  const activeContentItem = useMemo(
    () => findContentItem(CONTENT_SELECT_GROUPS, selectedContentItemId),
    [selectedContentItemId]
  );
  const mediaRequired = true;
  const accepts = ".jpg,.jpeg,.png,.webp";
  const assetUrl = useMemo(() => resultAssetUrl(result), [result]);
  const isContentLoading = loadingContentItemId !== null;
  const isNimBackend = useMemo(
    () => Boolean(backendInfo && String(backendInfo.backend || "").toLowerCase().includes("nim")),
    [backendInfo]
  );
  useEffect(() => {
    let cancelled = false;
    fetch(COSMOS3_INFO_URL, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled || !d) return;
        const name =
          (d.display_name as string | undefined) ||
          (d.checkpoint as string | undefined) ||
          (Array.isArray(d.models) ? (d.models[0] as string | undefined) : undefined);
        if (name) {
          setModel(name);
        }
        setBackendInfo(d);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
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

  const etaSeconds = useMemo(
    () => estimateWallSeconds(resolution, numFrames, steps, isNimBackend),
    [isNimBackend, numFrames, resolution, steps]
  );

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

  async function loadFile(file: File | null) {
    if (!file) return;
    contentLoadTokenRef.current += 1;
    setLoadingContentItemId(null);
    if (generatorMode === "Image-to-Video" && !file.type.startsWith("image/")) {
      setResult({
        error: "Image-to-Video expects a JPG, PNG, or WebP image.",
        diagnostic: { layer: "frontend", issue: "The selected file was not an image." }
      });
      setStatus("Wrong input type");
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
    setSelectedPreviewVideo({ url: "", name: "" });
    setStatus("Conditioning media loaded");
  }

  async function handleFile(event: ChangeEvent<HTMLInputElement>) {
    await loadFile(event.target.files?.[0] ?? null);
  }

  function applyExample(example: ExampleItem) {
    const params = visibleParams(example.params);
    const advanced = advancedParams(example.params);
    setSelectedExampleId(example.id);
    setGeneratorMode(example.mode);
    setSelectedPromptChoice("long");
    setPrompt(example.prompt);
    setResolution(params.resolution);
    setNumFrames(params.numFrames);
    setFps(params.fps);
    setSteps(params.steps);
    setGuidanceScale(params.guidance);
    setSeed(params.seed);
    setImageSize(advanced.imageSize);
    setShift(advanced.shift);
    setSigmaMax(advanced.sigmaMax);
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
    setSelectedPreviewVideo({ url: "", name: "" });
    setProgressPercent(0);
    setStatus(`${example.eyebrow} example loaded`);
    setMobilePanel("input");
    setExamplesOpen(false);
    if (inputRef.current) inputRef.current.value = "";
  }

  async function applyContentSelect(item: ContentSelectItem, promptChoice: PromptChoice = "long") {
    const loadToken = contentLoadTokenRef.current + 1;
    contentLoadTokenRef.current = loadToken;
    setLoadingContentItemId(item.id);
    setStatus(`Loading ${item.title}`);
    setResult(null);
    setProgressPercent(0);
    setProgressFrames([]);
    setMobilePanel("input");
    if (inputRef.current) inputRef.current.value = "";

    try {
      await Promise.all([
        preloadContentImage(item.mediaUrl),
        preloadContentVideo(item.previewVideoUrl),
        waitForContentLoad(450)
      ]);
      if (contentLoadTokenRef.current !== loadToken) return;
      const params = visibleParams(item.params);
      const advanced = advancedParams(item.params);
      setSelectedContentItemId(item.id);
      setSelectedPromptChoice(promptChoice);
      setGeneratorMode("Image-to-Video");
      setPrompt(contentPromptForChoice(item, promptChoice));
      setResolution(params.resolution);
      setNumFrames(params.numFrames);
      setFps(params.fps);
      setSteps(params.steps);
      setGuidanceScale(params.guidance);
      setSeed(params.seed);
      setImageSize(advanced.imageSize);
      setShift(advanced.shift);
      setSigmaMax(advanced.sigmaMax);
      setMedia(contentItemToMedia(item));
      setSelectedPreviewVideo({ url: item.previewVideoUrl, name: item.previewVideoName });
      setStatus("Ready");
    } catch (error) {
      if (contentLoadTokenRef.current !== loadToken) return;
      setStatus("Example failed to load");
      setResult({
        error: "Example image could not be loaded.",
        diagnostic: {
          layer: "frontend",
          issue: error instanceof Error ? error.message : String(error),
          suggestions: ["Choose another content tile or refresh the page before generating."]
        }
      });
      setMobilePanel("output");
    } finally {
      if (contentLoadTokenRef.current === loadToken) {
        setLoadingContentItemId(null);
      }
    }
  }

  function reset() {
    const params = visibleParams(DEFAULT_CONTENT_ITEM.params);
    const advanced = advancedParams(DEFAULT_CONTENT_ITEM.params);
    contentLoadTokenRef.current += 1;
    setLoadingContentItemId(null);
    setModel(DEFAULT_MODEL);
    setGeneratorMode("Image-to-Video");
    setMedia(contentItemToMedia(DEFAULT_CONTENT_ITEM));
    setPrompt(DEFAULT_PROMPT);
    setResolution(params.resolution);
    setNumFrames(params.numFrames);
    setFps(params.fps);
    setGuidanceScale(params.guidance);
    setSteps(params.steps);
    setSeed(params.seed);
    setImageSize(advanced.imageSize);
    setShift(advanced.shift);
    setSigmaMax(advanced.sigmaMax);
    setSelectedExampleId(EXAMPLES[0].id);
    setSelectedContentItemId(DEFAULT_CONTENT_ITEM.id);
    setSelectedPromptChoice("long");
    setSelectedPreviewVideo({ url: DEFAULT_CONTENT_ITEM.previewVideoUrl, name: DEFAULT_CONTENT_ITEM.previewVideoName });
    setStatus("Ready");
    setProgressPercent(0);
    setProgressFrames([]);
    setResult(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  async function handleDrop(event: DragEvent<HTMLButtonElement>) {
    event.preventDefault();
    setDragActive(false);
    await loadFile(event.dataTransfer.files?.[0] ?? null);
  }

  async function run(overrides: { resolution?: string; numFrames?: number; steps?: number } = {}) {
    if (isContentLoading) {
      setStatus("Still loading example");
      return;
    }

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
      setMobilePanel("output");
      return;
    }

    const runResolution = overrides.resolution ?? resolution;
    const runNumFrames = overrides.numFrames ?? numFrames;
    const runSteps = overrides.steps ?? steps;

    setSubmittedAt(Date.now());
    setElapsedSeconds(0);
    setIsRunning(true);
    setResult(null);
    setProgressPercent(0);
    setStatus("Predicting frames");
    setMobilePanel("output");
    try {
      const generationParams = generatorMode === "Action Policy" ? activeExample.params : activeContentItem?.params;
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mediaDataUrl: media?.dataUrl,
          visionPath: absoluteMediaUrl(media?.sourceUrl),
          mediaKind: media?.kind,
          mode: generatorMode,
          prompt,
          model,
          guidanceScale,
          steps: runSteps,
          resolution: runResolution,
          numFrames: runNumFrames,
          fps,
          seed,
          actionMode: generatorMode === "Action Policy" ? activeExample.params?.actionMode ?? "policy" : undefined,
          domainName: generatorMode === "Action Policy" ? activeExample.params?.domainName : undefined,
          imageSize: generatorMode === "Action Policy" ? generationParams?.imageSize : imageSize,
          actionChunkSize: generatorMode === "Action Policy" ? activeExample.params?.actionChunkSize : undefined,
          rawActionDim: generatorMode === "Action Policy" ? activeExample.params?.rawActionDim : undefined,
          shift: generatorMode === "Action Policy" ? generationParams?.shift : shift,
          negativePrompt: generationParams?.negativePrompt,
          conditionFrameIndexesVision: generationParams?.conditionFrameIndexesVision,
          conditionVideoKeep: generationParams?.conditionVideoKeep,
          modelMode: generationParams?.modelMode,
          sigmaMax: generatorMode === "Action Policy" ? generationParams?.sigmaMax : sigmaMax,
          guidanceInterval: generationParams?.guidanceInterval,
          normalizeCfg: generationParams?.normalizeCfg,
          videoSaveQuality: generationParams?.videoSaveQuality,
          imageSaveQuality: generationParams?.imageSaveQuality,
          negativeMetadataMode: generationParams?.negativeMetadataMode,
          negativePromptKeepMetadata: generationParams?.negativePromptKeepMetadata,
          numOutputs: generationParams?.numOutputs
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

  function applyBudgetSuggestionAndRun(suggestion: BudgetSuggestion) {
    const nextResolution = String(suggestion.resolution);
    const nextFrames = Number(suggestion.num_frames);
    const nextSteps = Number(suggestion.num_steps);
    if (!nextResolution || !Number.isFinite(nextFrames) || !Number.isFinite(nextSteps)) return;
    setResolution(nextResolution);
    setNumFrames(nextFrames);
    setSteps(nextSteps);
    setStatus("Using suggested parameters");
    void run({ resolution: nextResolution, numFrames: nextFrames, steps: nextSteps });
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
          <p>{PAGE_TAGLINE}</p>
          <div className="heroModelName">{MODEL_CARD_TITLE}</div>
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

      <div className="tabs" role="tablist" aria-label="Model sections">
        {(["Experience", "Model Card", "System Card"] as SectionTab[]).map((tab) => {
          const id = tab.toLowerCase().replace(/\s+/g, "-");
          return (
            <button
              aria-controls={`tabpanel-${id}`}
              aria-selected={activeTab === tab}
              className={activeTab === tab ? "active" : ""}
              id={`tab-${id}`}
              key={tab}
              onClick={() => setActiveTab(tab)}
              role="tab"
              tabIndex={activeTab === tab ? 0 : -1}
            >
              {tab}
            </button>
          );
        })}
      </div>

      <section
        aria-labelledby={`tab-${activeTab.toLowerCase().replace(/\s+/g, "-")}`}
        className="experience"
        id={`tabpanel-${activeTab.toLowerCase().replace(/\s+/g, "-")}`}
        role="tabpanel"
      >
        {activeTab === "Experience" ? (
          <>
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

            <div className="worldModeSummary">
              <span>World Creation Mode</span>
              <div>
                <strong>Image-to-World</strong>
                <p>Generates future frames based upon an image and text input.</p>
              </div>
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
                      <strong>{displayMediaName(media.name)}</strong>
                      {media.sourceUrl ? null : <small>Upload staged for request</small>}
                    </span>
                  </span>
                ) : (
                  <>
                    <Upload size={22} />
                    <span>Drop source image here</span>
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
              label={selectedPromptChoice === "short" ? "Short Prompt" : "Long Prompt"}
              hint={selectedPromptChoice === "short" ? "Selected concise generation prompt." : "Full generation prompt."}
              max={12000}
              value={prompt}
              onChange={setPrompt}
              rows={7}
            />

            <ModelParameterControls
              fps={fps}
              guidanceScale={guidanceScale}
              imageSize={imageSize}
              numFrames={numFrames}
              resolution={String(resolution)}
              seed={seed}
              setFps={setFps}
              setGuidanceScale={setGuidanceScale}
              setImageSize={setImageSize}
              setNumFrames={setNumFrames}
              setResolution={setResolution}
              setSeed={setSeed}
              setShift={setShift}
              setSigmaMax={setSigmaMax}
              setSteps={setSteps}
              shift={shift}
              sigmaMax={sigmaMax}
              steps={steps}
            />

            <div className="runBar">
              <button className="resetButton" onClick={reset}>
                <RotateCcw size={16} />
                Reset
              </button>
              <button className="runButton" onClick={() => run()} disabled={isRunning || isContentLoading || (mediaRequired && !media)}>
                <Play size={16} fill="currentColor" />
                {isContentLoading ? "Loading" : isRunning ? "Generating" : "Generate New Video"}
              </button>
            </div>
            <div className="termsRow">
              <div className="governingTerms">
                <strong>Governing Terms:</strong>
                <span>
                  This trial service is governed by the NVIDIA API Trial Terms of Service. Use of this model is governed
                  by the NVIDIA Open Model License Agreement.
                </span>
              </div>
            </div>
          </section>

          <section className={`panel outputPanel ${mobilePanel === "output" ? "mobileActivePanel" : ""}`}>
            <div className="panelHeader">
              <div className="outputTabs">
                <h2>Output</h2>
              </div>
              <div className="outputActions">
                {isRunning || result?.error || status !== "Ready" ? (
                  <span className={`statusPill ${isRunning ? "working" : result?.error ? "error" : "success"}`}>
                    {isRunning ? "Generating" : result?.error ? "Error" : status}
                  </span>
                ) : null}
              </div>
            </div>
            <p className="outputDescriptor">Generated Future World</p>
            <div className="outputBody">
              {isRunning ? (
                <GenerationProgress
                  media={media}
                  frames={progressFrames}
                  progress={progressPercent}
                  elapsedSeconds={elapsedSeconds}
                  etaSeconds={etaSeconds}
                  totalFrames={numFrames}
                />
              ) : result?.error ? (
                <FailureReport result={result} onApplySuggestedParams={applyBudgetSuggestionAndRun} />
              ) : result?.videoDataUrl ? (
                <video className="resultVideo" src={result.videoDataUrl} controls />
              ) : result?.imageDataUrl ? (
                <img className="resultVideo" src={result.imageDataUrl} alt="Generated world state" />
              ) : assetUrl ? (
                <a className="assetLink" href={assetUrl} target="_blank">
                  Open generated asset
                  <ExternalLink size={15} />
                </a>
              ) : selectedPreviewVideo.url ? (
                <video className="resultVideo" src={selectedPreviewVideo.url} aria-label={selectedPreviewVideo.name} controls />
              ) : (
                <CuratedOutputPreview item={DEFAULT_CONTENT_ITEM} />
              )}
            </div>
          </section>
        </div>
        {backendInfo?.warning ? <p className="backendWarning inlineWarning">{backendInfo.warning}</p> : null}
          </>
        ) : (
          <StaticTab backendInfo={backendInfo} model={model} tab={activeTab} />
        )}
      </section>

      {examplesOpen ? (
        <ExampleModal
          groups={CONTENT_SELECT_GROUPS}
          selectedId={selectedContentItemId}
          selectedPromptChoice={selectedPromptChoice}
          onClose={() => setExamplesOpen(false)}
          onSelect={(item, promptChoice) => {
            applyContentSelect(item, promptChoice);
            setExamplesOpen(false);
          }}
        />
      ) : null}
    </main>
  );
}

function StaticTab({
  backendInfo,
  model,
  tab
}: {
  backendInfo: BackendInfo | null;
  model: string;
  tab: SectionTab;
}) {
  if (tab === "Model Card") {
    return (
      <div className="staticPanel">
        <p className="staticEyebrow">Overview</p>
        <h2>{MODEL_CARD_TITLE}</h2>
        <p className="staticLead">{MODEL_CARD_LEAD}</p>

        <StaticSection title="Title">
          <p>
            <strong>{MODEL_CARD_TITLE}</strong> {PAGE_TAGLINE}
          </p>
        </StaticSection>

        <StaticSection title="Description">
          <p>
            The Vite Generator stages Image-to-World requests for Cosmos3 generation while preserving the NVIDIA Build
            dark control styling, examples modal, and curated preview-first output behavior.
          </p>
        </StaticSection>

        <StaticSection title="Input">
          <dl>
            <dt>Image-to-World</dt>
            <dd>Image plus text prompt.</dd>
            <dt>Formats</dt>
            <dd>.jpg, .jpeg, .png, .webp for image conditioning.</dd>
          </dl>
        </StaticSection>

        <StaticSection title="Output">
          <dl>
            <dt>Type</dt>
            <dd>MP4 video when the backend returns inline media, or an asset URL when hosted output is returned.</dd>
            <dt>Review</dt>
            <dd>Review generated content before promoting it into datasets.</dd>
          </dl>
        </StaticSection>

        <StaticSection title="Content Selects">
          <p>Domain order is Robotics, then Autonomous Vehicles.</p>
          <dl>
            <dt>Robotics</dt>
            <dd>Robot Cutting Board Sorting, Robot Pear Bowl Placement, Robot Bok Choy Pan Placement</dd>
            <dt>Autonomous Vehicles</dt>
            <dd>Suburban Intersection Yield, Urban Signal Approach, Signalized Left Turn Completion</dd>
          </dl>
        </StaticSection>

        <StaticSection title="Software Integration">
          <RuntimeDetails backendInfo={backendInfo} model={model} />
        </StaticSection>

        <StaticSection title="Ethical Considerations">
          <p>
            Generated future frames can be inaccurate, biased, unsafe, or physically inconsistent. Keep human review in
            the loop for simulation, training, safety, and dataset approval workflows.
          </p>
        </StaticSection>

        <a className="staticLink" href={BUILD_PREDICT_MODEL_CARD_URL} rel="noreferrer" target="_blank">
          NVIDIA Build model-card reference <ExternalLink size={16} />
        </a>
      </div>
    );
  }

  return (
    <div className="staticPanel">
      <p className="staticEyebrow">System Card</p>
      <h2>NVIDIA Cosmos Generator</h2>
      <p className="staticLead">{MODEL_CARD_LEAD}</p>

      <StaticSection title="Cosmos Model Family">
        <p>
          Cosmos generation models support physical AI development by creating future video rollouts from text or image
          conditioning. This staging surface is focused on the Generator path and keeps robot policy hidden until a live
          backend advertises support.
        </p>
      </StaticSection>

      <StaticSection title="Governing Terms / Terms of Use">
        <p>
          Use is subject to the license, model terms, and deployment environment attached to the active backend.
          Confirm commercial rights, data handling, and redistribution requirements before production use.
        </p>
      </StaticSection>

      <StaticSection title="Specific Risk Areas and Mitigations">
        <ul>
          <li>Generated frames may violate physics or object permanence; validate outputs against source intent.</li>
          <li>Robotics and AV workflows can carry safety consequences; require domain expert review.</li>
          <li>Record prompts, model details, and backend warnings for repeatable evaluations.</li>
        </ul>
      </StaticSection>

      <StaticSection title="Deployment">
        <RuntimeDetails backendInfo={backendInfo} model={model} />
      </StaticSection>

      <StaticSection title="Content Order">
        <p>Robotics appears first, followed by Autonomous Vehicles.</p>
      </StaticSection>

      <a className="staticLink" href={BUILD_PREDICT_SYSTEM_CARD_URL} rel="noreferrer" target="_blank">
        NVIDIA Build system-card reference <ExternalLink size={16} />
      </a>
    </div>
  );
}

function StaticSection({ children, title }: { children: ReactNode; title: string }) {
  return (
    <section className="staticSection">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function RuntimeDetails({ backendInfo, model }: { backendInfo: BackendInfo | null; model: string }) {
  const capabilities = backendInfo?.capabilities
    ? Object.entries(backendInfo.capabilities)
        .filter(([, enabled]) => enabled)
        .map(([key]) => key.replace(/_/g, "-"))
        .join(", ")
    : "";

  return (
    <dl className="runtimeDetailsList">
      <dt>Live model</dt>
      <dd>{displayValue(backendInfo?.display_name || backendInfo?.checkpoint || model)}</dd>
      <dt>Backend</dt>
      <dd>{displayValue(backendInfo?.backend || "cosmos3-generate")}</dd>
      <dt>Base URL</dt>
      <dd>{displayValue(backendInfo?.base_url)}</dd>
      <dt>Infer URL</dt>
      <dd>{displayValue(backendInfo?.infer_url)}</dd>
      <dt>Backend image</dt>
      <dd>{displayValue(backendInfo?.image || backendInfo?.staged_checkpoint?.image)}</dd>
      <dt>Capabilities</dt>
      <dd>{displayValue(capabilities)}</dd>
    </dl>
  );
}

function FailureReport({
  result,
  onApplySuggestedParams
}: {
  result: ApiResult;
  onApplySuggestedParams?: (suggestion: BudgetSuggestion) => void;
}) {
  const diagnostic = result.diagnostic ?? {};
  const layer = typeof diagnostic.layer === "string" ? diagnostic.layer : "unknown";
  const issue = typeof diagnostic.issue === "string" ? diagnostic.issue : result.error ?? "Request failed";
  const likelyCause = typeof diagnostic.likelyCause === "string" ? diagnostic.likelyCause : "No structured cause was returned.";
  const endpoint = typeof diagnostic.endpoint === "string" ? diagnostic.endpoint : null;
  const suggestions = Array.isArray(diagnostic.suggestions) ? diagnostic.suggestions.map(String) : [];
  const budgetSuggestion = result.budget?.suggestions?.find(
    (suggestion) =>
      suggestion &&
      typeof suggestion.resolution === "string" &&
      Number.isFinite(Number(suggestion.num_frames)) &&
      Number.isFinite(Number(suggestion.num_steps))
  );
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
      {budgetSuggestion && onApplySuggestedParams ? (
        <button className="suggestedParamsButton" type="button" onClick={() => onApplySuggestedParams(budgetSuggestion)}>
          Use suggested parameters and regenerate
          {budgetSuggestion.est_seconds ? <small>~{formatDuration(budgetSuggestion.est_seconds)}</small> : null}
        </button>
      ) : null}
      <details>
        <summary>Diagnostic JSON</summary>
        <pre>{JSON.stringify({ diagnostic, budget: result.budget, payload: result.payload, raw: result.raw }, null, 2)}</pre>
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
  void progress;
  void elapsedSeconds;
  void etaSeconds;
  void totalFrames;
  const sourceFrame = media?.previewUrl || frames[0] || "";
  const displayFrames =
    frames.length > 0
      ? frames.slice(0, PROGRESS_FRAME_COUNT)
      : Array.from({ length: PROGRESS_FRAME_COUNT }, () => sourceFrame);
  while (displayFrames.length < PROGRESS_FRAME_COUNT) {
    displayFrames.push(sourceFrame);
  }

  return (
    <article className="generationProgress" aria-live="polite">
      <div className="autoregressiveHeader">
        <p>
          <strong>The autoregressive model is working:</strong> Predicting future frames for you...
        </p>
      </div>
      <div className="autoregressiveFrameStrip">
        {displayFrames.map((src, index) => {
          const style = {
            "--future-blur": `${index * 1.1}px`,
            "--future-scale": String(1 + index * 0.014),
            "--future-pixel": index < 1 ? "auto" : "pixelated",
            "--future-opacity": String(Math.max(0.42, 1 - index * 0.1))
          } as CSSProperties;

          return (
            <figure className="autoregressiveFrame" style={style} key={`${src || "fallback"}-${index}`}>
              {src ? <img src={src} alt="" /> : <span aria-hidden="true" />}
            </figure>
          );
        })}
      </div>
      <section className="communityWaitShelf" aria-label="Community-generated videos">
        <h3>While you wait! Check out these community-generated videos</h3>
        <div className="communityVideoRow">
          <button className="carouselArrow leftArrow" type="button" aria-label="Previous community video">
            <ChevronLeft size={30} />
          </button>
          {COMMUNITY_VIDEOS.map((src, index) => (
            <video
              className={index === 1 ? "communityVideo primaryCommunityVideo" : "communityVideo"}
              controls
              key={src}
              muted
              playsInline
              preload="metadata"
              src={src}
            />
          ))}
          <button className="carouselArrow rightArrow" type="button" aria-label="Next community video">
            <ChevronRight size={30} />
          </button>
        </div>
      </section>
    </article>
  );
}

function CuratedOutputPreview({ item = DEFAULT_CONTENT_ITEM }: { item?: ContentSelectItem }) {
  const previewVideoUrl = item.previewVideoUrl || CURATED_PREVIEW_VIDEO;
  return (
    <article className="curatedOutputPreview">
      <video src={previewVideoUrl} autoPlay muted loop playsInline controls />
      <div>
        <p className="curatedEyebrow">Generated preview</p>
        <h3>{item.title}</h3>
        <p>Canonical output is preloaded. Run the active backend again to create a fresh video.</p>
      </div>
    </article>
  );
}

function ModelParameterControls({
  fps,
  guidanceScale,
  imageSize,
  numFrames,
  resolution,
  seed,
  setFps,
  setGuidanceScale,
  setImageSize,
  setNumFrames,
  setResolution,
  setSeed,
  setShift,
  setSigmaMax,
  setSteps,
  shift,
  sigmaMax,
  steps
}: {
  fps: number;
  guidanceScale: number;
  imageSize: number;
  numFrames: number;
  resolution: string;
  seed: number;
  setFps: (value: number) => void;
  setGuidanceScale: (value: number) => void;
  setImageSize: (value: number) => void;
  setNumFrames: (value: number) => void;
  setResolution: (value: string) => void;
  setSeed: (value: number) => void;
  setShift: (value: number) => void;
  setSigmaMax: (value: number) => void;
  setSteps: (value: number) => void;
  shift: number;
  sigmaMax: number;
  steps: number;
}) {
  return (
    <section className="parameterShell" aria-label="Model parameters">
      <div className="parameterTopline">
        <h3>Model Parameters</h3>
        <span>Applied on Generate</span>
      </div>
      <div className="parameterGrid">
        <label className="numberField">
          <span>Resolution</span>
          <select value={resolution} onChange={(event) => setResolution(event.target.value)}>
            <option value="480">480</option>
            <option value="720">720</option>
            <option value="256">256</option>
          </select>
        </label>
        <NumberControl label="Frames" min={1} max={189} step={1} value={numFrames} onChange={setNumFrames} />
        <NumberControl label="Steps" min={1} max={80} step={1} value={steps} onChange={setSteps} />
        <NumberControl label="FPS" min={1} max={60} step={1} value={fps} onChange={setFps} />
        <NumberControl label="Guidance" min={0} max={20} step={0.5} value={guidanceScale} onChange={setGuidanceScale} />
        <NumberControl label="Seed" min={0} max={999999} step={1} value={seed} onChange={setSeed} />
      </div>
      <details className="advancedParameterDetails">
        <summary>Advanced parameters</summary>
        <div className="parameterGrid advancedParameterGrid">
          <NumberControl label="Shift" min={0} max={20} step={0.5} value={shift} onChange={setShift} />
          <NumberControl label="Sigma max" min={1} max={120} step={1} value={sigmaMax} onChange={setSigmaMax} />
          <NumberControl label="Image size" min={128} max={1024} step={1} value={imageSize} onChange={setImageSize} />
        </div>
      </details>
    </section>
  );
}

function NumberControl({
  label,
  max,
  min,
  onChange,
  step,
  value
}: {
  label: string;
  max: number;
  min: number;
  onChange: (value: number) => void;
  step: number;
  value: number;
}) {
  return (
    <label className="numberField">
      <span>{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => {
          const next = Number(event.target.value);
          if (!Number.isFinite(next)) return;
          onChange(Math.min(max, Math.max(min, next)));
        }}
      />
    </label>
  );
}

function PromptBox({
  label,
  hint,
  value,
  max,
  rows,
  readOnly = false,
  onChange
}: {
  label: string;
  hint: string;
  value: string;
  max: number;
  rows: number;
  readOnly?: boolean;
  onChange: (value: string) => void;
}) {
  return (
    <div className="promptBox">
      <div className="promptTopline">
        <label>
          {label}
          {readOnly ? null : <span className="requiredDot">*</span>}
          <span className="infoDot promptInfoDot" tabIndex={0}>
            i
            <span className="promptTooltip" role="tooltip">
              Text prompts are used for world generation. For best results aligned to Cosmos capabilities, use prompts
              for autonomous driving, robotics, and industrial domains.
            </span>
          </span>
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
        readOnly={readOnly}
        aria-readonly={readOnly}
        onChange={(event) => {
          if (!readOnly) onChange(event.target.value);
        }}
      />
      <small>{hint}</small>
    </div>
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
          <span className="runtimeLabel">Backend Image</span>
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
  groups,
  onClose,
  onSelect,
  selectedId,
  selectedPromptChoice
}: {
  groups: ContentSelectGroup[];
  onClose: () => void;
  onSelect: (example: ContentSelectItem, promptChoice: PromptChoice) => void;
  selectedId: string;
  selectedPromptChoice: PromptChoice;
}) {
  const [pendingId, setPendingId] = useState(selectedId);
  const [pendingPromptChoice, setPendingPromptChoice] = useState<PromptChoice>(selectedPromptChoice);
  const pending = findContentItem(groups, pendingId);
  const promptChoices = [
    pending.shortPrompt ? { choice: "short" as const, label: "Short Prompt", text: pending.shortPrompt } : null,
    { choice: "long" as const, label: "Long Prompt", text: pending.prompt }
  ].filter((choice): choice is { choice: PromptChoice; label: string; text: string } => Boolean(choice));

  useEffect(() => {
    if (pendingPromptChoice === "short" && !pending.shortPrompt) {
      setPendingPromptChoice("long");
    }
  }, [pending.shortPrompt, pendingPromptChoice]);

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
            <h2 id="examples-title">Select an Example</h2>
            <p>Select an image and prompt from the examples below.</p>
          </div>
          <button className="iconButton" aria-label="Close examples" onClick={onClose}>
            <X size={18} />
          </button>
        </div>
        <div className="exampleDomainList">
          {groups.map((group) => (
            <section className="exampleDomain" key={group.title}>
              <div className="exampleDomainHeader">
                <h3>{group.title}</h3>
                <p>{group.summary}</p>
              </div>
              <div className="exampleGrid">
                {group.items.map((item) => (
                  <button
                    aria-pressed={pendingId === item.id}
                    className={pendingId === item.id ? "exampleImageCard selectedExample" : "exampleImageCard"}
                    key={item.id}
                    onClick={() => setPendingId(item.id)}
                    type="button"
                  >
                    <img src={item.mediaUrl} alt="" />
                    <span>{item.title}</span>
                  </button>
                ))}
              </div>
            </section>
          ))}
        </div>
        <div className="modalFooter">
          <div className="selectedExampleSummary">
            <div className="selectedExampleIntro">
              <img className="selectedExampleHero" src={pending.mediaUrl} alt="" />
              <div className="selectedExampleCopy">
                <span>{pending.domain}</span>
                <strong>{pending.title}</strong>
                <p>{pending.description}</p>
              </div>
            </div>
            <div className="selectedPromptPair" role="group" aria-label="Prompt choice">
              {promptChoices.map((choice) => (
                <button
                  aria-pressed={pendingPromptChoice === choice.choice}
                  className={
                    pendingPromptChoice === choice.choice
                      ? "selectedPromptChoice activePromptChoice"
                      : "selectedPromptChoice"
                  }
                  key={choice.choice}
                  onClick={() => setPendingPromptChoice(choice.choice)}
                  type="button"
                >
                  <span>{choice.label}</span>
                  <p>{choice.text}</p>
                </button>
              ))}
            </div>
          </div>
          <div className="modalFooterActions">
            <button className="cancelButton" onClick={onClose} type="button">
              Cancel
            </button>
            <button className="selectButton" onClick={() => onSelect(pending, pendingPromptChoice)} type="button">
              Select
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
