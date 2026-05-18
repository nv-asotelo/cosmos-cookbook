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
import { ChangeEvent, CSSProperties, DragEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";

const HERO_IMAGE = "https://assets.ngc.nvidia.com/products/api-catalog/images/cosmos-predict1-5b.jpg";
const BUILD_PREDICT_MODEL_CARD_URL = "https://build.nvidia.com/nvidia/cosmos-predict1-5b/modelcard";
const BUILD_PREDICT_SYSTEM_CARD_URL = "https://build.nvidia.com/nvidia/cosmos-predict1-5b/systemcard";
const MODEL_CARD_TITLE = "Image-to-World";
const PAGE_TAGLINE = "Generates future frames based upon an image and text input.";
const MODEL_CARD_LEAD =
  "Generates future frames of a physics-aware world state based on simply an image or short video along with a text prompt for physical AI development.";
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
  "Detecting model...";
const HERO_TAGS = [
  "physical ai",
  "world foundation model",
  "robotics",
  "simulation",
  "synthetic data generation",
  "text-to-video",
  "image-to-world",
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
const NIM_ETA_BASELINE_SECONDS = 24;
const NIM_ETA_OPS_PER_SECOND = 2_500_000;

type GeneratorMode = "Text-to-Video" | "Image-to-Video" | "Action Policy";
type SectionTab = "Experience" | "Model Card" | "System Card";
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
type ContentSelectItem = {
  id: string;
  title: string;
  domain: string;
  description: string;
  prompt: string;
  mediaUrl: string;
  mediaName: string;
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
    label: "Robotic Arm Ingredient Sorting",
    eyebrow: "T2V",
    mode: "Text-to-Video",
    prompt:
      "A robotic arm interacts with various objects on a wooden cutting board placed within an open cardboard box. The cutting board contains a small orange bowl, a red tomato with green leaves, a piece of salmon with a pinkish-orange hue, and a small orange carrot. The robotic arm, black and metallic, is positioned above these items, seemingly preparing to pick up one of them. In subsequent frames, the robotic arm descends and uses its claw-like mechanism to grasp the salmon. It lifts the carrot slightly off the board, moves it towards the orange bowl, and releases it, causing the salmon to fall into the bowl. The background includes a tiled wall and a glimpse of a workshop setting. A medium shot captures the robotic arm's interaction with the objects.",
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
const CONTENT_SELECT_GROUPS: ContentSelectGroup[] = [
  {
    title: "Robotics",
    summary: "Robot manipulation scenes with tabletop objects, grippers, and goal-directed motion.",
    items: [
      {
        id: "robot-ingredient-sorting",
        title: "Robot Arm Ingredient Sorting",
        domain: "Robotics",
        description: "A tabletop robot scene with food items, a bowl, and a gripper poised for manipulation.",
        prompt:
          "Animate the robotic arm as it carefully reaches toward the tabletop objects, grasps one item, and places it into the bowl. Preserve the tabletop layout, camera angle, workshop lighting, and realistic contact physics.",
        mediaUrl: "/examples/robot_apple.png",
        mediaName: "Robot Arm Ingredient Sorting.png"
      },
      {
        id: "robot-tabletop-motion",
        title: "Robot Tabletop Motion",
        domain: "Robotics",
        description: "A robot-hand view over a workbench with a raised platform and object targets.",
        prompt:
          "Animate the robot hand with small, precise motions toward the tabletop target. Keep the original scene geometry, camera position, and lab lighting stable while the end effector moves naturally.",
        mediaUrl: "/examples/robot_tabletop.jpg",
        mediaName: "Robot Tabletop Motion.jpg"
      },
      {
        id: "robot-tape-placement",
        title: "Robot Tape Placement",
        domain: "Robotics",
        description: "Two robot hands observe tape and a basket from a first-person manipulation setup.",
        prompt:
          "Animate the robot hands reaching toward the blue tape, lifting it from the table, and placing it into the basket. Preserve object identity, hand geometry, and the tabletop perspective.",
        mediaUrl: "/examples/robot_tape.png",
        mediaName: "Robot Tape Placement.png"
      }
    ]
  },
  {
    title: "Autonomous Vehicles",
    summary: "Forward-facing driving scenes that exercise road layout, ego motion, and traffic context.",
    items: [
      {
        id: "residential-intersection-turn",
        title: "Residential Intersection Turn",
        domain: "Autonomous Vehicles",
        description: "An ego-vehicle view at an urban intersection under clear daylight.",
        prompt:
          "Animate the ego vehicle rolling forward slowly through the intersection while preserving lane geometry, crosswalk markings, parked cars, building facades, and bright daylight. Keep the camera fixed to the vehicle and maintain realistic traffic motion.",
        mediaUrl: "/examples/av_intersection.jpg",
        mediaName: "Residential Intersection Turn.jpg"
      },
      {
        id: "rainy-night-traffic",
        title: "Rainy Night Traffic",
        domain: "Autonomous Vehicles",
        description: "A low-light roadway scene with wet pavement, vehicles, and headlight reflections.",
        prompt:
          "Animate the vehicle queue inching forward on the wet road at night. Preserve headlight reflections, road boundaries, vehicle spacing, and low-light atmosphere with smooth ego-camera motion.",
        mediaUrl: "/examples/av_rainy_night.jpg",
        mediaName: "Rainy Night Traffic.jpg"
      },
      {
        id: "race-car-track",
        title: "Race Car Track",
        domain: "Autonomous Vehicles",
        description: "A high-speed driving condition with track context and strong forward motion.",
        prompt:
          "Animate a forward driving rollout on the track with smooth ego motion, stable road boundaries, and realistic vehicle dynamics. Preserve the original camera framing and lighting.",
        mediaUrl: "/examples/av_race_car.jpg",
        mediaName: "Race Car Track.jpg"
      }
    ]
  },
  {
    title: "Industrial Smart Spaces",
    summary: "Warehouse and facility scenes for physical AI simulation and smart-space monitoring.",
    items: [
      {
        id: "warehouse-camera-grid",
        title: "Warehouse Camera Grid",
        domain: "Industrial Smart Spaces",
        description: "A multi-camera warehouse overview with shelves, floor lanes, and varied lighting.",
        prompt:
          "Animate the smart-space warehouse scene with subtle worker and equipment movement across camera views. Preserve the multi-camera structure, shelves, floor markings, and lighting consistency.",
        mediaUrl: "/examples/warehouse_grid.jpg",
        mediaName: "Warehouse Camera Grid.jpg"
      },
      {
        id: "warehouse-summary",
        title: "Warehouse Summary View",
        domain: "Industrial Smart Spaces",
        description: "A wide warehouse monitoring view with spatial context for goods and aisles.",
        prompt:
          "Animate the warehouse monitoring view with realistic small movements in the aisles while preserving camera perspective, object permanence, shelving, and traffic lanes.",
        mediaUrl: "/examples/warehouse_summary.png",
        mediaName: "Warehouse Summary View.png"
      }
    ]
  }
];
const DEFAULT_CONTENT_ITEM = CONTENT_SELECT_GROUPS[0].items[0];
const DEFAULT_PROMPT = DEFAULT_CONTENT_ITEM.prompt;

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
    description: "Prompt-only generation with quality staging defaults.",
    icon: FileVideo,
    enabled: true,
    visible: true
  },
  {
    id: "Image-to-Video",
    label: "Image-to-Video",
    description: "Image-to-World generation from one image plus text.",
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

export default function Page() {
  const inputRef = useRef<HTMLInputElement>(null);
  const contentLoadTokenRef = useRef(0);
  const [activeTab, setActiveTab] = useState<SectionTab>("Experience");
  const [collection, setCollection] = useState("cosmos3");
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [models, setModels] = useState<string[]>(MODEL_CHOICES);
  const [backendInfo, setBackendInfo] = useState<BackendInfo | null>(null);
  const [runtimeOpen, setRuntimeOpen] = useState(false);
  const [generatorMode, setGeneratorMode] = useState<GeneratorMode>("Image-to-Video");
  const [schemaMode, setSchemaMode] = useState<SchemaMode>("local_nim");
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
  const [copied, setCopied] = useState(false);
  const [outputTab, setOutputTab] = useState<OutputTab>("preview");
  const [mobilePanel, setMobilePanel] = useState<MobilePanel>("input");
  const [examplesOpen, setExamplesOpen] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [selectedExampleId, setSelectedExampleId] = useState(EXAMPLES[0].id);
  const [selectedContentItemId, setSelectedContentItemId] = useState(DEFAULT_CONTENT_ITEM.id);
  const [loadingContentItemId, setLoadingContentItemId] = useState<string | null>(null);

  const activeExample = useMemo(
    () => EXAMPLES.find((example) => example.id === selectedExampleId) ?? EXAMPLES[0],
    [selectedExampleId]
  );
  const visibleModes = useMemo(() => GENERATOR_MODES.filter((mode) => mode.visible), []);
  const mediaRequired = generatorMode !== "Text-to-Video";
  const accepts = generatorMode === "Image-to-Video" ? ".jpg,.jpeg,.png,.webp" : ".mp4,.mov,.jpg,.jpeg,.png,.webp";
  const assetUrl = useMemo(() => resultAssetUrl(result), [result]);
  const isContentLoading = loadingContentItemId !== null;
  const isNimBackend = useMemo(
    () => (backendInfo ? String(backendInfo.backend || "").toLowerCase().includes("nim") : schemaMode === "local_nim"),
    [backendInfo, schemaMode]
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

  const etaSeconds = useMemo(
    () => estimateWallSeconds(resolution, numFrames, steps, isNimBackend),
    [isNimBackend, numFrames, resolution, steps]
  );
  const generatedFrameCount = useMemo(
    () => Math.max(1, Math.min(numFrames, Math.round((progressPercent / 100) * numFrames))),
    [numFrames, progressPercent]
  );
  const remainingSeconds = Math.max(0, etaSeconds - elapsedSeconds);

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
    const visionPath = absoluteMediaUrl(media?.sourceUrl) ?? (media?.dataUrl ? "<written to /tmp/uploads/...>" : null);
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
    contentLoadTokenRef.current += 1;
    setLoadingContentItemId(null);
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
    contentLoadTokenRef.current += 1;
    setLoadingContentItemId(null);
    setGeneratorMode(mode);
    setMedia(mode === "Image-to-Video" ? contentItemToMedia(findContentItem(CONTENT_SELECT_GROUPS, selectedContentItemId)) : null);
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

  async function applyContentSelect(item: ContentSelectItem) {
    const loadToken = contentLoadTokenRef.current + 1;
    contentLoadTokenRef.current = loadToken;
    setLoadingContentItemId(item.id);
    setStatus(`Loading ${item.title}`);
    setResult(null);
    setProgressPercent(0);
    setProgressFrames([]);
    setOutputTab("preview");
    setMobilePanel("input");
    if (inputRef.current) inputRef.current.value = "";

    try {
      await Promise.all([preloadContentImage(item.mediaUrl), waitForContentLoad(450)]);
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
      setStatus(`${item.title} loaded`);
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
      setOutputTab("preview");
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
    setCollection("cosmos3");
    setModel(DEFAULT_MODEL);
    setGeneratorMode("Image-to-Video");
    setSchemaMode("local_nim");
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

      <RuntimeDetailsToggle
        backendInfo={backendInfo}
        detailsUrl={`${window.location.origin}/api/active-model`}
        model={model}
        open={runtimeOpen}
        setOpen={setRuntimeOpen}
      />

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
              <NumberField label="Frames" value={numFrames} min={25} max={241} step={4} onChange={setNumFrames} />
              <NumberField label="FPS" value={fps} min={1} max={30} step={1} onChange={setFps} />
              <NumberField label="Seed" value={seed} min={-1} max={2147483647} step={1} onChange={setSeed} />
            </div>

            <div className="runBar">
              <button className="resetButton" onClick={reset}>
                <RotateCcw size={16} />
                Reset
              </button>
              <span
                className={isRunning ? "etaBadge runningEta" : "etaBadge"}
                title={`${isNimBackend ? "NIM staging" : "Ray Serve"} estimator. ${etaSeconds}s for ${numFrames} frames at ${resolution} resolution x ${steps} steps.`}
              >
                {isRunning
                  ? `${isNimBackend ? "NIM" : "Est."} ${generatedFrameCount}/${numFrames} frames · ${formatDuration(remainingSeconds)} left`
                  : `${isNimBackend ? "NIM est." : "Est. wall"} ~${formatDuration(etaSeconds)}`}
              </span>
              <button className="runButton" onClick={run} disabled={isRunning || isContentLoading || (mediaRequired && !media)}>
                <Play size={16} fill="currentColor" />
                {isContentLoading ? "Loading" : isRunning ? "Generating" : "Generate"}
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
              ) : (
                <WorldPreview isNimBackend={isNimBackend} mode={generatorMode} />
              )}
            </div>
          </section>
        </div>

        <ContentSelects
          disabled={isRunning}
          groups={CONTENT_SELECT_GROUPS}
          loadingId={loadingContentItemId}
          onApply={applyContentSelect}
          selectedId={selectedContentItemId}
        />

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

function ContentSelects({
  disabled,
  groups,
  loadingId,
  onApply,
  selectedId
}: {
  disabled: boolean;
  groups: ContentSelectGroup[];
  loadingId: string | null;
  onApply: (item: ContentSelectItem) => void;
  selectedId: string;
}) {
  return (
    <section className="contentSelects" aria-label="Content selects">
      <div className="contentSelectsHeader">
        <div>
          <p>Content Selects</p>
          <h2>Generator examples by domain</h2>
        </div>
        <span>Robotics, Autonomous Vehicles, Industrial Smart Spaces</span>
      </div>
      <div className="contentGroupGrid">
        {groups.map((group) => (
          <article className="contentGroup" key={group.title}>
            <div className="contentGroupTopline">
              <h3>{group.title}</h3>
              <p>{group.summary}</p>
            </div>
            <div className="contentCardGrid">
              {group.items.map((item) => {
                const isLoading = loadingId === item.id;
                const isSelected = selectedId === item.id && !isLoading;
                const isDisabled = disabled || loadingId !== null;

                return (
                  <button
                    aria-busy={isLoading}
                    aria-pressed={isSelected}
                    className={`contentCard ${isSelected ? "selectedContentCard" : ""} ${isLoading ? "loadingContentCard" : ""}`}
                    disabled={isDisabled}
                    key={`${group.title}-${item.id}`}
                    onClick={() => onApply(item)}
                    type="button"
                  >
                    <div
                      aria-hidden="true"
                      className="contentThumb"
                      style={{ backgroundImage: `url("${item.mediaUrl}")` }}
                    >
                      <span className="contentDomainPill">{item.domain}</span>
                      {isLoading ? (
                        <div className="contentLoadingOverlay">
                          <span className="contentSpinner" />
                          <strong>Loading image and prompt</strong>
                          <small>Generate will enable when ready.</small>
                        </div>
                      ) : null}
                    </div>
                    <div className="contentCardBody">
                      <span className="contentMediaLabel">{item.domain}</span>
                      <strong>{item.title}</strong>
                      <p>{item.description}</p>
                    </div>
                    <div className="contentActions">
                      <span>{isLoading ? "Loading..." : isSelected ? "Selected" : "Select image"}</span>
                    </div>
                  </button>
                );
              })}
            </div>
          </article>
        ))}
      </div>
    </section>
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
            The Vite Generator stages Text-to-Video and Image-to-Video requests for Cosmos3 generation while preserving
            the NVIDIA Build dark control styling, active model details, examples modal, preview/JSON output tabs, and
            quick staging controls.
          </p>
        </StaticSection>

        <StaticSection title="Input">
          <dl>
            <dt>Text-to-Video</dt>
            <dd>Text prompt only.</dd>
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
            <dd>Use the Preview and JSON tabs together before promoting generated content into datasets.</dd>
          </dl>
        </StaticSection>

        <StaticSection title="Content Selects">
          <p>Domain order is Robotics, Autonomous Vehicles, then Industrial Smart Spaces.</p>
          <dl>
            <dt>Robotics</dt>
            <dd>Robot Arm Ingredient Sorting, Robot Tabletop Motion, Robot Tape Placement</dd>
            <dt>Autonomous Vehicles</dt>
            <dd>Residential Intersection Turn, Rainy Night Traffic, Race Car Track</dd>
            <dt>Industrial Smart Spaces</dt>
            <dd>Warehouse Camera Grid, Warehouse Summary View</dd>
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
          Use is subject to the license, NIM image terms, and deployment environment attached to the active backend.
          Confirm commercial rights, data handling, and redistribution requirements before production use.
        </p>
      </StaticSection>

      <StaticSection title="Specific Risk Areas and Mitigations">
        <ul>
          <li>Generated frames may violate physics or object permanence; validate outputs against source intent.</li>
          <li>Robotics and AV workflows can carry safety consequences; require domain expert review.</li>
          <li>Record prompts, parameters, model details, and backend warnings for repeatable evaluations.</li>
        </ul>
      </StaticSection>

      <StaticSection title="Deployment">
        <RuntimeDetails backendInfo={backendInfo} model={model} />
      </StaticSection>

      <StaticSection title="Content Order">
        <p>Robotics appears first, followed by Autonomous Vehicles, followed by Industrial Smart Spaces.</p>
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
      <dt>NIM image</dt>
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
          <strong>{isNimBackend ? "NIM latent rollout is running:" : "The diffusion model is working:"}</strong> denoising{" "}
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

function WorldPreview({ isNimBackend, mode }: { isNimBackend: boolean; mode: GeneratorMode }) {
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
        Run a T2V prompt or add an I2V conditioning image. The active {isNimBackend ? "NIM" : "Ray Serve"} backend returns
        a generated MP4 plus raw JSON for diagnostics.
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
