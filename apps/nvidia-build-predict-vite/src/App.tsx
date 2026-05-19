"use client";

import { ChevronDown, ChevronRight, ExternalLink, FileVideo, HelpCircle, Info, Menu, Play, RotateCcw, Search, Upload, X } from "lucide-react";
import { ChangeEvent, CSSProperties, DragEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";

const HERO_IMAGE = "https://assets.ngc.nvidia.com/products/api-catalog/images/cosmos-predict1-5b.jpg";
const BUILD_PREDICT_MODEL_CARD_URL = "https://build.nvidia.com/nvidia/cosmos-predict1-5b/modelcard";
const BUILD_PREDICT_SYSTEM_CARD_URL = "https://build.nvidia.com/nvidia/cosmos-predict1-5b/systemcard";
const MODEL_CARD_TITLE = "Image-to-World";
const PAGE_TAGLINE = "Generates future frames based upon an image and text input.";
const MODEL_CARD_LEAD =
  "Generates future frames of a physics-aware world state based on simply an image or short video along with a text prompt for physical AI development.";
const DISPLAY_MODEL_FALLBACK = "Cosmos3-Nano";
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
type ContentSelectItem = {
  id: string;
  title: string;
  domain: string;
  description: string;
  prompt: string;
  mediaUrl: string;
  mediaName: string;
  previewVideoUrl: string;
  previewVideoName: string;
};
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

const CANONICAL_PROMPTS = {
  "001": `The robotic arm picks up the pear and place it in the dark-color bowl`,
  "005": `The robotic hand picks up the bok choy and places it to the left of the frying pan`,
  "007": `A robotic arm interacts with various objects on a wooden cutting board placed within an open cardboard box. The cutting board contains a small orange bowl, a red tomato with green leaves, a piece of salmon with a pinkish-orange hue, and a small orange carrot. The robotic arm, black and metallic, is positioned above these items, seemingly preparing to pick up one of them. In subsequent frames, the robotic arm descends and uses its claw-like mechanism to grasp the salmon. It lifts the carrot slightly off the board, moves it towards the orange bowl, and releases it, causing the salmon to fall into the bowl. The background includes a tiled wall and a glimpse of a workshop setting. A medium shot captures the robotic arm's interaction with the objects.`,
  "009": `The video begins with a view from inside a vehicle, approaching an intersection in a suburban neighborhood under a clear blue sky. The road is marked with double yellow lines and features a stop lane marker painted on the asphalt. To the right, there is a house with a well-maintained hedge, and a stop sign in front of it, with a parked car on the street. A white car is seen turning right at the intersection, heading down the street. On the left side of the road, there is a red brick wall and another parked car. The background shows overhead utility poles with wires crisscrossing the sky, and some bare trees line the streets, indicating it might be late fall or early spring. The scene is calm and typical of a residential area. As the video progresses, the white car exits the frame, revealing more of the intersection and the surrounding residential area. The ego vehicle comes to a stop, yielding to an oncoming vehicle while waiting to turn right. The background scenery of houses, trees, and utility poles remains consistent, with the lighting suggesting that the sun is still high, maintaining the bright and clear conditions observed in the initial frame. The overall atmosphere remains calm and typical of a suburban neighborhood.`,
  "010": `A close-up view captures a melting popsicle on a white plate. The popsicle features a vibrant red top, likely strawberry-flavored, transitioning into a creamy white base, possibly vanilla or yogurt-based. The popsicle stick is visible on the right side, and the surface of the popsicle shows signs of melting, with the red portion becoming more fluid and spreading outward. The creamy white base remains relatively intact initially but eventually starts to soften and blend with the red, creating a smooth transition between the two flavors. The camera remains steady, focusing on the popsicle as it transforms from a solid form into a liquid, with the background remaining plain and white to ensure all attention is on the melting process. The lighting is consistent throughout, highlighting the colors and textures of the ice cream.`,
  "011": `A close-up of a precision metalworking process in a controlled industrial setting. The first frame captures a cylindrical metal workpiece securely mounted on a lathe, rotating smoothly as a cutting machine, held by a black, angular fixture, approaches from above. The cutting machine, marked with numerical identifiers (5513 020-10), engages with the workpiece, shaving off thin metal shavings that are visibly ejected into the air, creating a fine mist around the machining area. The background is blurred, focusing attention on the interaction between the cutting machine and the workpiece, which reflects light, indicating its polished surface. As the video progresses, the cutting machine continues its linear motion along the length of the workpiece, maintaining a steady pace. The tool's engagement with the material results in consistent metal shaving, producing a continuous stream of shavings that are dispersed into the surrounding space. The workpiece remains stationary relative to the camera's perspective, ensuring a clear view of the cutting metal process. The environment suggests a well-lit workshop, emphasizing the precision and efficiency of the operation. By the final frame, the cutting machine has almost completed its pass along the workpiece, leaving behind a smooth, polished surface. The metal shavings continue to be ejected, and the overall scene maintains a focused and industrious atmosphere, underscoring the meticulous nature of the metalworking process.`
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
    title: "Industrial Smart Spaces",
    summary: "Facility and physical-process scenes for smart-space simulation and monitoring.",
    items: [
      {
        id: "melting-popsicle-physics",
        title: "Melting Popsicle Physics",
        domain: "Industrial Smart Spaces",
        description: "A close-up physical-process scene focused on melting and material change.",
        prompt: CANONICAL_PROMPTS["010"],
        mediaUrl: "/examples/canonical/010.png",
        mediaName: "Melting Popsicle Physics.png",
        previewVideoUrl: "/examples/canonical/010.mp4",
        previewVideoName: "Melting Popsicle Physics.mp4"
      },
      {
        id: "industrial-metal-lathe",
        title: "Industrial Metal Lathe",
        domain: "Industrial Smart Spaces",
        description: "A precision machining scene with a rotating workpiece and cutting tool.",
        prompt: CANONICAL_PROMPTS["011"],
        mediaUrl: "/examples/canonical/011.png",
        mediaName: "Industrial Metal Lathe.png",
        previewVideoUrl: "/examples/canonical/011.mp4",
        previewVideoName: "Industrial Metal Lathe.mp4"
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
        prompt: CANONICAL_PROMPTS["009"],
        mediaUrl: "/examples/canonical/009.png",
        mediaName: "Suburban Intersection Yield.png",
        previewVideoUrl: "/examples/canonical/009.mp4",
        previewVideoName: "Suburban Intersection Yield.mp4"
      }
    ]
  },
  {
    title: "Robotics",
    summary: "Robot manipulation scenes with tabletop objects, grippers, and goal-directed motion.",
    items: [
      {
        id: "robot-cutting-board-sorting",
        title: "Robot Cutting Board Sorting",
        domain: "Robotics",
        description: "A tabletop robot scene with food items, a bowl, and a gripper in a workshop setting.",
        prompt: CANONICAL_PROMPTS["007"],
        mediaUrl: "/examples/canonical/007.png",
        mediaName: "Robot Cutting Board Sorting.png",
        previewVideoUrl: "/examples/canonical/007.mp4",
        previewVideoName: "Robot Cutting Board Sorting.mp4"
      },
      {
        id: "robot-pear-bowl-placement",
        title: "Robot Pear Bowl Placement",
        domain: "Robotics",
        description: "A robot manipulation setup with a pear and a dark bowl.",
        prompt: CANONICAL_PROMPTS["001"],
        mediaUrl: "/examples/canonical/001.png",
        mediaName: "Robot Pear Bowl Placement.png",
        previewVideoUrl: "/examples/canonical/001.mp4",
        previewVideoName: "Robot Pear Bowl Placement.mp4"
      },
      {
        id: "robot-bok-choy-pan-placement",
        title: "Robot Bok Choy Pan Placement",
        domain: "Robotics",
        description: "A gripper positions bok choy near a frying pan on a tabletop.",
        prompt: CANONICAL_PROMPTS["005"],
        mediaUrl: "/examples/canonical/005.png",
        mediaName: "Robot Bok Choy Pan Placement.png",
        previewVideoUrl: "/examples/canonical/005.mp4",
        previewVideoName: "Robot Bok Choy Pan Placement.mp4"
      }
    ]
  }
];
const DEFAULT_CONTENT_ITEM =
  findContentItem(CONTENT_SELECT_GROUPS, "robot-cutting-board-sorting") ?? CONTENT_SELECT_GROUPS[0].items[0];
const DEFAULT_PROMPT = DEFAULT_CONTENT_ITEM.prompt;

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
  const [mobilePanel, setMobilePanel] = useState<MobilePanel>("input");
  const [examplesOpen, setExamplesOpen] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [selectedExampleId, setSelectedExampleId] = useState(EXAMPLES[0].id);
  const [selectedContentItemId, setSelectedContentItemId] = useState(DEFAULT_CONTENT_ITEM.id);
  const [selectedPreviewVideo, setSelectedPreviewVideo] = useState(() => ({
    url: DEFAULT_CONTENT_ITEM.previewVideoUrl,
    name: DEFAULT_CONTENT_ITEM.previewVideoName
  }));
  const [loadingContentItemId, setLoadingContentItemId] = useState<string | null>(null);

  const activeExample = useMemo(
    () => EXAMPLES.find((example) => example.id === selectedExampleId) ?? EXAMPLES[0],
    [selectedExampleId]
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
    setSelectedPreviewVideo({ url: "", name: "" });
    setProgressPercent(0);
    setStatus(`${example.eyebrow} example loaded`);
    setMobilePanel("input");
    setExamplesOpen(false);
    if (inputRef.current) inputRef.current.value = "";
  }

  async function applyContentSelect(item: ContentSelectItem) {
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
      setSelectedContentItemId(item.id);
      setGeneratorMode("Image-to-Video");
      setPrompt(item.prompt);
      setResolution(QUICK_VIDEO_PARAMS.resolution);
      setNumFrames(QUICK_VIDEO_PARAMS.frames_count);
      setFps(QUICK_VIDEO_PARAMS.frames_per_sec);
      setSteps(QUICK_VIDEO_PARAMS.num_steps);
      setGuidanceScale(QUICK_VIDEO_PARAMS.guidance);
      setSeed(0);
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
    contentLoadTokenRef.current += 1;
    setLoadingContentItemId(null);
    setGeneratorMode("Image-to-Video");
    setMedia(contentItemToMedia(DEFAULT_CONTENT_ITEM));
    setPrompt(DEFAULT_PROMPT);
    setResolution(QUICK_VIDEO_PARAMS.resolution);
    setNumFrames(QUICK_VIDEO_PARAMS.frames_count);
    setFps(QUICK_VIDEO_PARAMS.frames_per_sec);
    setGuidanceScale(QUICK_VIDEO_PARAMS.guidance);
    setSteps(QUICK_VIDEO_PARAMS.num_steps);
    setSeed(0);
    setSelectedExampleId(EXAMPLES[0].id);
    setSelectedContentItemId(DEFAULT_CONTENT_ITEM.id);
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

  async function run() {
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

    setSubmittedAt(Date.now());
    setElapsedSeconds(0);
    setIsRunning(true);
    setResult(null);
    setProgressPercent(0);
    setStatus("Predicting frames");
    setMobilePanel("output");
    try {
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
                <strong>Video-to-World</strong>
                <p>Generate future frames based on a video input.</p>
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
              label="Prompt (Read Only)"
              hint="Describe the future world state to generate."
              max={1000}
              value={prompt}
              onChange={setPrompt}
              readOnly
              rows={4}
            />

            <div className="runBar">
              <button className="resetButton" onClick={reset}>
                <RotateCcw size={16} />
                Reset
              </button>
              <button className="runButton" onClick={run} disabled={isRunning || isContentLoading || (mediaRequired && !media)}>
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
                {isRunning || result?.error || status === "Complete" ? (
                  <span className={`statusPill ${isRunning ? "working" : result?.error ? "error" : "success"}`}>
                    {isRunning ? "Generating" : result?.error ? "Error" : "Complete"}
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
                  isNimBackend={isNimBackend}
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
          onClose={() => setExamplesOpen(false)}
          onSelect={(item) => {
            applyContentSelect(item);
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
          <p>Domain order is Industrial Smart Spaces, Autonomous Vehicles, then Robotics.</p>
          <dl>
            <dt>Industrial Smart Spaces</dt>
            <dd>Melting Popsicle Physics, Industrial Metal Lathe</dd>
            <dt>Autonomous Vehicles</dt>
            <dd>Suburban Intersection Yield</dd>
            <dt>Robotics</dt>
            <dd>Robot Cutting Board Sorting, Robot Pear Bowl Placement, Robot Bok Choy Pan Placement</dd>
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
        <p>Industrial Smart Spaces appears first, followed by Autonomous Vehicles, followed by Robotics.</p>
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
  isNimBackend,
  totalFrames
}: {
  media: MediaState | null;
  frames: string[];
  progress: number;
  elapsedSeconds: number;
  etaSeconds: number;
  isNimBackend: boolean;
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
          <strong>Generation is running:</strong> denoising{" "}
          {totalFrames} latent frames in parallel, then VAE-decoding and encoding the output.
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
          const stage = readiness < 0.25 ? "Latent" : readiness < 0.78 ? "Denoising" : "Decoding";
          const style = {
            "--frame-blur": `${Math.max(0, 10 - readiness * 10)}px`,
            "--frame-opacity": String(0.38 + readiness * 0.62),
            "--diffusion-opacity": String(Math.max(0.18, 0.74 - readiness * 0.5)),
            "--diffusion-reveal": `${Math.max(8, readiness * 100)}%`
          } as CSSProperties;

          return (
            <figure className={src ? "generatedFrame diffusionFrame" : "generatedFrame latentFrame"} style={style} key={`${src || "fallback"}-${index}`}>
              {src ? (
                <>
                  <img src={src} alt="" />
                  <span className="diffusionField" aria-hidden="true" />
                  <span className="diffusionState">{stage}</span>
                </>
              ) : (
                <div className="generatedFrameFallback">
                  <span>Latent</span>
                </div>
              )}
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

function CuratedOutputPreview({ item }: { item: ContentSelectItem }) {
  return (
    <article className="curatedOutputPreview">
      <video src={item.previewVideoUrl} autoPlay muted loop playsInline controls />
      <div>
        <p className="curatedEyebrow">Generated preview</p>
        <h3>{item.title}</h3>
        <p>Canonical output is preloaded. Run the active backend again to create a fresh video.</p>
      </div>
    </article>
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
  selectedId
}: {
  groups: ContentSelectGroup[];
  onClose: () => void;
  onSelect: (example: ContentSelectItem) => void;
  selectedId: string;
}) {
  const [pendingId, setPendingId] = useState(selectedId);
  const pending = findContentItem(groups, pendingId);

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
            <span>{pending.domain}</span>
            <strong>{pending.title}</strong>
            <p>{pending.description}</p>
          </div>
          <div className="modalFooterActions">
            <button className="cancelButton" onClick={onClose} type="button">
              Cancel
            </button>
            <button className="selectButton" onClick={() => onSelect(pending)} type="button">
              Select
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
