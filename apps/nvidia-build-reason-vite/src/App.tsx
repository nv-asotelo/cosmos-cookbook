import {
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Copy,
  ExternalLink,
  FileImage,
  FileVideo,
  HelpCircle,
  Info,
  Menu,
  Play,
  RotateCcw,
  Search,
  Upload,
  X
} from "lucide-react";
import { ChangeEvent, DragEvent, ReactNode, RefObject, useEffect, useMemo, useRef, useState } from "react";

const HERO_IMAGE = "https://assets.ngc.nvidia.com/products/api-catalog/images/cosmos-reason2-8b.jpg";
const BUILD_REASON2_MODEL_CARD_URL = "https://build.nvidia.com/nvidia/cosmos-reason2-8b/modelcard";
const BUILD_REASON2_SYSTEM_CARD_URL = "https://build.nvidia.com/nvidia/cosmos-reason2-8b/systemcard";
const BUILD_REASON2_DEPLOY_URL = "https://build.nvidia.com/nvidia/cosmos-reason2-8b/deploy";
const COSMOS3_INFO_URL =
  (typeof import.meta !== "undefined" && (import.meta as { env?: Record<string, string> }).env?.VITE_COSMOS3_INFO_URL) ||
  "/api/active-model";
const DEFAULT_MODEL =
  (typeof import.meta !== "undefined" && (import.meta as { env?: Record<string, string> }).env?.VITE_MODEL_NAME) ||
  "Detecting model...";
const DEFAULT_BACKEND =
  (typeof import.meta !== "undefined" && (import.meta as { env?: Record<string, string> }).env?.VITE_INFERENCE_BACKEND) ||
  "";
const DEFAULT_USER_PROMPT = "";
const DEFAULT_SYSTEM_PROMPT = "";
const DEPLOY_DOCKER_COMMAND = (model: string, backendInfo: BackendInfo | null) => `docker login nvcr.io
Username: $oauthtoken
Password: <PASTE_API_KEY_HERE>

export NGC_API_KEY=<PASTE_API_KEY_HERE>
export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE"
chmod -R a+w "$LOCAL_NIM_CACHE"

docker run -it --rm \\
  --gpus all \\
  --ipc host \\
  --shm-size=32GB \\
  -e NGC_API_KEY \\
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \\
  -u $(id -u) \\
  -p 8000:8000 \\
  ${nimImageForModel(model, backendInfo)}`;
const DEPLOY_CURL_COMMAND = (model: string) => `curl -X POST "http://0.0.0.0:8000/v1/chat/completions" \\
  -H "Accept: application/json" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${model}",
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "What is in this video?" },
          {
            "type": "video_url",
            "video_url": {
              "url": "https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason1-7b/av_construction_stop_timestamped.mp4"
            }
          }
        ]
      }
    ],
    "max_tokens": 256
  }'`;
const SAMPLING_DEFAULTS = {
  standard: {
    topP: 0.8,
    topK: 20,
    repetitionPenalty: 1.0,
    presencePenalty: 1.5,
    temperature: 0.7
  },
  reasoning: {
    topP: 0.95,
    topK: 20,
    repetitionPenalty: 1.0,
    presencePenalty: 0.0,
    temperature: 0.6
  }
};
const DEFAULT_TEMPERATURE = SAMPLING_DEFAULTS.reasoning.temperature;
const DEFAULT_TOP_P = SAMPLING_DEFAULTS.reasoning.topP;
const DEFAULT_TOP_K = 20;
const DEFAULT_MAX_TOKENS = 512;
const DEFAULT_FRAMES_PER_SECOND = 2;
const DEFAULT_REPETITION_PENALTY = SAMPLING_DEFAULTS.reasoning.repetitionPenalty;
const DEFAULT_PRESENCE_PENALTY = SAMPLING_DEFAULTS.reasoning.presencePenalty;
const DEFAULT_SEED = 42;
const AGIBOT_VIDEO = "/examples/agibot.mp4";
const ROBOT_TAPE_IMAGE = "/examples/robot_tape.png";
const TENNIS_TEMPORAL_EXAMPLE_ID = "tennis-temporal-events";
const TENNIS_TEMPORAL_VIDEO = "/examples/tennis_nim_safe.mp4";
const WAREHOUSE_ROW_D_VIDEO = "/examples/warehouse_7min.mp4";
const ACCEPTED_MEDIA_EXTENSIONS = ["mp4", "mov", "m4v", "webm", "avi", "jpg", "jpeg", "png", "webp"];
const ACCEPTED_MEDIA_ACCEPT = [
  "video/*",
  "image/*",
  ...ACCEPTED_MEDIA_EXTENSIONS.map((extension) => `.${extension}`)
].join(",");
const ACCEPTED_MEDIA_HELP = ".mp4, .mov, .m4v, .webm, .avi, .jpg, .jpeg, .png, .webp";
const PARAMETER_HELP = {
  temperature:
    "Sampling parameter from the guide. Default 0.7, or 0.6 with reasoning. Lower values are more deterministic; higher values are more varied.",
  topP:
    "Sampling parameter top_p from the guide. Default 0.8, or 0.95 with reasoning. Smaller values use a tighter nucleus; larger values allow a broader token pool.",
  topK:
    "Sampling parameter top_k from the guide. Default 20 with or without reasoning. Caps the candidate token set considered during sampling.",
  repetitionPenalty:
    "Sampling parameter repetition_penalty from the guide. Default 1.0 with or without reasoning, which is neutral repetition handling.",
  presencePenalty:
    "Sampling parameter presence_penalty from the guide. Default 1.5, or 0.0 with reasoning. Higher values push novelty; 0.0 applies no novelty push.",
  framesPerSecond:
    "Vite media preprocessing control. It is not part of the guide sampling table; it controls video frame sampling before the request when frame fallback is needed.",
  maxTokens:
    "App completion limit control. It is not part of the guide sampling table; the server forwards it only when configured to send max_tokens.",
  seed:
    "Reproducibility control. It is not part of the guide sampling table; the server forwards it only when configured to send seed.",
  reasoning:
    "Matches the guide's reasoning format. When enabled, Vite appends the <think> instruction and uses reasoning sampling defaults: temperature 0.6, top_p 0.95, top_k 20, repetition_penalty 1.0, presence_penalty 0.0."
};
const REASONING_FORMAT_INSTRUCTION = `Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag.`;
const HERO_TAGS = [
  "physical ai",
  "autonomous vehicles",
  "industrial",
  "reasoning",
  "robotics",
  "smart cities",
  "synthetic data generation",
  "video understanding",
  "vision language model"
];
const VLA_HERO_TAGS = [
  "vision language action",
  "Alpamayo",
  "ego camera",
  "video captioning",
  "VQA mode",
  "BYO video",
  "physical ai",
  "adapter"
];

type SectionTab = "Experience" | "Model Card" | "System Card" | "Deploy";
type OutputTab = "preview" | "json";
type MobilePanel = "input" | "output";
type LongPreset = "fast" | "balanced" | "detailed";
type RunMode = "standard" | "long" | null;
type ExampleGroupId = "build" | "vss" | "av-dense-captioning" | "anomaly-id" | "embodied-reasoning";

const LONG_VIDEO_PRESET_FPS: Record<LongPreset, number> = {
  fast: 2,
  balanced: 4,
  detailed: 6
};
const LONG_VIDEO_DEFAULT_CONCURRENCY = 16;

type MediaState = {
  name: string;
  kind: "video" | "image";
  previewUrl: string;
  dataUrl: string;
  sourceUrl?: string;
};

type ImageSize = {
  width: number;
  height: number;
};

type SpatialSource = "thinking" | "final";

type SpatialMark = {
  id: string;
  kind: "point" | "bbox";
  sourceKey: string;
  coordinateMode: "cosmos-1000" | "unit" | "pixel";
  source: SpatialSource;
  x: number;
  y: number;
  width: number;
  height: number;
  centerX: number;
  centerY: number;
  label: string;
  detail?: string;
  sequence: number;
};

type ExampleItem = {
  id: string;
  title: string;
  mediaUrl: string;
  mediaName: string;
  mediaKind: "video" | "image";
  userPrompt: string;
  systemPrompt: string;
  reasoning: boolean;
  group?: ExampleGroupId;
  judgeNote?: string;
  longVideoEnabled?: boolean;
  parameters?: Partial<{
    framesPerSecond: number;
    maxTokens: number;
    presencePenalty: number;
    repetitionPenalty: number;
    temperature: number;
    topK: number;
    topP: number;
  }>;
};

type ApiFile = {
  path: string;
  b64: string | null;
  mime: string;
  error?: string;
};

type ApiResult = {
  status?: "success" | "error" | "skip";
  message?: string;
  stack_trace?: string | null;
  files?: ApiFile[];
  content?: string;
  reasoning?: string;
  combined_content?: string;
  schema?: string;
  error?: string;
  openai?: unknown;
  payload?: unknown;
  raw?: unknown;
  media?: unknown;
  long_video?: {
    preset?: string;
    chunks?: LongChunkProgress[];
  };
};

type StreamPhase =
  | "idle"
  | "preparing_media"
  | "alpamayo_generating"
  | "waiting_first_token"
  | "reasoning"
  | "answer"
  | "complete"
  | "error"
  | "stopped";

type StreamState = {
  phase: StreamPhase;
  reasoning: string;
  answer: string;
  schema: string;
  raw?: unknown;
  usage?: unknown;
  message?: string;
  created: number;
  model: string;
};

type LongChunkProgress = {
  index: number;
  status: "queued" | "running" | "done" | "error";
  timeRange?: string;
  frameCount?: number;
  thumbnailUrl?: string | null;
  summary?: string;
  content?: string;
  parsed?: unknown;
  events?: unknown[];
  error?: string;
  elapsedSeconds?: number;
  eventsCount?: number;
};

type TimelineEvent = {
  start?: string;
  end?: string;
  event_type?: string;
  type?: string;
  caption?: string;
  summary?: string;
  confidence?: number | string;
  player?: string;
  source_chunk?: number;
  [key: string]: unknown;
};

type TimelineItem = {
  id: string;
  range: string;
  title: string;
  caption: string;
  meta?: string;
};

type LongStepProgress = {
  key: string;
  label: string;
  status: "pending" | "running" | "done" | "error";
  detail?: string;
  progress?: number;
};

type ExampleLoadState = {
  title: string;
  phase: string;
  detail?: string;
  percent: number;
  elapsedSeconds: number;
  etaSeconds?: number | null;
  bytesLoaded?: number;
  bytesTotal?: number;
  steps: LongStepProgress[];
};

type LongProgressState = {
  phase?: string;
  message?: string;
  preset: LongPreset;
  durationSeconds?: number;
  durationText?: string;
  sourceFps?: number;
  frameCount?: number;
  sampleFps?: number;
  requestedFps?: number | null;
  extractionFps?: number;
  frameLimit?: number;
  totalChunks?: number;
  completedChunks?: number;
  failedChunks?: number;
  runningChunks?: number;
  percent?: number;
  elapsedSeconds?: number;
  etaSeconds?: number | null;
  concurrency?: number;
  maxConcurrency?: number;
  steps?: LongStepProgress[];
  warnings: string[];
  chunks: LongChunkProgress[];
  partialTimeline: Array<{ index: number; timeRange?: string; summary: string }>;
};

type ParsedSseEvent = {
  event: string;
  data: unknown;
};

type VllmProcessFlags = Record<string, string | boolean | number | null | undefined>;

type BackendInfo = {
  checkpoint?: string;
  display_name?: string;
  cosmos3_version?: string;
  backend?: string;
  gpu_name?: string;
  vram_free_gib?: number;
  vram_total_gib?: number;
  base_url?: string;
  source?: {
    sha?: string | null;
    timestamp?: string | null;
    branch?: string | null;
    dirty?: boolean | null;
    repo_id?: string | null;
    revision?: string | null;
    cache_path?: string | null;
    commit_url?: string | null;
    source?: string | null;
  };
  app_source?: {
    sha?: string | null;
    timestamp?: string | null;
    branch?: string | null;
    dirty?: boolean | null;
  };
  quantization?: {
    applied?: boolean;
    method?: string | null;
    dtype?: string | null;
    source?: string | null;
  };
  vla?: {
    type?: string;
    family?: string;
    adapter?: string;
    current_mode?: string;
    model_id?: string;
    backbone?: string;
    input_shape?: string;
    frames_per_video?: number;
    max_decoded_video_frames?: number;
    prompt_role?: string;
    full_trajectory_mode?: boolean;
    clip_guidance?: string;
  } | null;
  nim?: {
    name?: string;
    image?: string | null;
    image_id?: string | null;
    status?: string | null;
    started_at?: string | null;
    env?: Record<string, string>;
  } | null;
  vllm?: {
    base_url?: string;
    model?: {
      id?: string;
      created?: number;
      owned_by?: string;
      root?: string;
      max_model_len?: number;
      [key: string]: unknown;
    } | null;
    process?: {
      pid?: number;
      command?: string;
      flags?: VllmProcessFlags;
    } | null;
  };
};

const EXAMPLE_GROUPS: Array<{ id: ExampleGroupId; label: string }> = [
  { id: "build", label: "build.nvidia.com" },
  { id: "vss", label: "VSS" },
  { id: "av-dense-captioning", label: "AV Dense Captioning" },
  { id: "anomaly-id", label: "Anomaly ID" },
  { id: "embodied-reasoning", label: "Embodied Reasoning" }
];

function initialBackendInfo(modelName = DEFAULT_MODEL, backend = DEFAULT_BACKEND): BackendInfo | null {
  const lower = `${modelName} ${backend}`.toLowerCase();
  if (String(backend).toLowerCase() !== "alpamayo" && !lower.includes("alpamayo")) return null;
  const modelId = modelName && modelName !== "Detecting model..." ? modelName : "nvidia/Alpamayo-1.5-10B";
  return {
    checkpoint: modelId,
    display_name: modelId,
    backend: "alpamayo",
    vla: {
      type: "VLA",
      family: "Alpamayo",
      adapter: "alpamayo_openai",
      current_mode: "vqa_text_generation",
      model_id: modelId,
      backbone: "Cosmos Reason2-8B VLM",
      input_shape: "OpenAI chat/completions with video_url or image_url plus text prompt",
      frames_per_video: 4,
      max_decoded_video_frames: 96,
      prompt_role: "VQA/caption question over sampled frames",
      full_trajectory_mode: false,
      clip_guidance: "Use short, front-loaded ego-camera clips for the Alpamayo VQA/captioning adapter."
    }
  };
}

const EXAMPLES: ExampleItem[] = [
  {
    id: "robotics-next-action",
    title: "Robotics Next Action Prediction",
    mediaUrl: AGIBOT_VIDEO,
    mediaName: "agibot.mp4",
    mediaKind: "video",
    userPrompt: "What can be the next immediate action?",
    systemPrompt: "You are a helpful assistant.",
    reasoning: true,
    parameters: {
      framesPerSecond: 4,
      maxTokens: 4096
    }
  },
  {
    id: "robot-arm",
    title: "robot arm pick up stuff",
    mediaUrl: ROBOT_TAPE_IMAGE,
    mediaName: "robot_tape.png",
    mediaKind: "image",
    userPrompt:
      'You are given the task "Move the tape into the basket". Specify the 2D trajectory your end effector should follow in pixel space. Return the trajectory coordinates in JSON format like this: {"point_2d": [x, y], "label": "gripper trajectory"}.\n\nPrompt format:\nAnswer the question using the following format:\n<think>\nYour reasoning.\n</think>\nWrite your final answer immediately after the </think> tag.',
    systemPrompt: "You are a helpful assistant.",
    reasoning: true,
    parameters: {
      framesPerSecond: 2,
      maxTokens: 4096,
      repetitionPenalty: 1.2,
      temperature: 0.3,
      topP: 0.3
    }
  },
  {
    id: "sdg-critic",
    title: "SDG critic",
    mediaUrl: "https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason2/cr2_rejection_sampling.mp4",
    mediaName: "sdg-critic.mp4",
    mediaKind: "video",
    userPrompt:
      "Approve or reject this generated video for inclusion in a dataset for physical world model ai training. It must perfectly adhere to physics, object permanence, and have no anomalies. Any issue or concern causes rejection. Answer with Approve or Reject only.",
    systemPrompt: "You are a helpful assistant.",
    reasoning: true,
    parameters: {
      framesPerSecond: 4,
      maxTokens: 4096,
      repetitionPenalty: 1.2,
      temperature: 0.3,
      topP: 0.3
    }
  },
  {
    id: "warehouse",
    title: "warehouse",
    mediaUrl: "https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason2/cr2_warehouse.mp4",
    mediaName: "warehouse.mp4",
    mediaKind: "video",
    userPrompt: "Which worker picked up the dropped box?",
    systemPrompt: "You are a helpful warehouse monitoring system.",
    reasoning: true,
    parameters: {
      framesPerSecond: 2,
      maxTokens: 4096,
      repetitionPenalty: 1.2,
      temperature: 0.3,
      topP: 0.3
    }
  },
  {
    id: "forklift",
    title: "forklift load weight evaluation",
    mediaUrl: "https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason2/cr2_forklift.jpg",
    mediaName: "forklift-load.jpg",
    mediaKind: "image",
    userPrompt:
      "Locate the bounding box of the load and determine if its size and weight of load within the forklift's limits. Estimate weights. Return all as json. Include json location, estimated weight of the load, and if it's in the limit.",
    systemPrompt: "You are a helpful assistant.",
    reasoning: false,
    parameters: {
      framesPerSecond: 2,
      maxTokens: 4096,
      repetitionPenalty: 1.2,
      temperature: 0.3,
      topP: 0.3
    }
  },
  {
    id: "mail-package",
    title: "mail package",
    mediaUrl: "https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason2/cr2_mail_package.mp4",
    mediaName: "mail-package.mp4",
    mediaKind: "video",
    userPrompt: "Is the person allowed to pick up the packages?",
    systemPrompt: "You are a helpful assistant.",
    reasoning: true,
    parameters: {
      framesPerSecond: 2,
      maxTokens: 4096,
      repetitionPenalty: 1.2,
      temperature: 0.3,
      topP: 0.8
    }
  },
  {
    id: "warehouse-row-d-box",
    title: "Warehouse row D shelf placement",
    group: "vss",
    mediaUrl: WAREHOUSE_ROW_D_VIDEO,
    mediaName: "warehouse_7min.mp4",
    mediaKind: "video",
    userPrompt:
      'In the first two minutes of the video, who put the box on the shelves on row D?\n\nUse only visible evidence from the video. Identify the person by stable visual details such as clothing color, position, or direction of travel, and cite the timestamp range where the box is placed on row D. If the person cannot be determined from visible evidence, say so and explain what is ambiguous.',
    systemPrompt:
      "You are a helpful video security analyst. Use only visible evidence, preserve timestamps, and do not infer actions that are not visible.",
    reasoning: true,
    longVideoEnabled: true,
    parameters: {
      framesPerSecond: 6,
      maxTokens: 4096,
      presencePenalty: 0,
      repetitionPenalty: 1.0,
      temperature: 0.6,
      topK: 20,
      topP: 0.95
    }
  },
  {
    id: TENNIS_TEMPORAL_EXAMPLE_ID,
    title: "Tennis temporal events",
    group: "vss",
    mediaUrl: TENNIS_TEMPORAL_VIDEO,
    mediaName: "tennis_nim_safe.mp4",
    mediaKind: "video",
    userPrompt:
      'Analyze this tennis clip for sports analytics. Identify every visible serve, racquet-ball hit/contact, and point-scoring or end-of-point moment. Use the video timeline, not frame numbers. Return only events that are visible in the clip; do not infer hidden, blurred, or between-sample events.\n\nUse timestamps in "mm:ss.ff" format. For each event, include "start", "end", "event_type" ("serve", "hit", "score", or "uncertain"), "player" ("near", "far", "left", "right", or "unknown"), "confidence" from 0 to 1, and "caption". A "score" event requires visible evidence that the point ended, such as a ball landing out, a winner, a net error, a double bounce, a clear player reaction, or a scoreboard change. If that evidence is not visible, put it in "uncertain_events" rather than "events".\n\nReturn the final answer as valid JSON with this shape: {"events": [], "uncertain_events": [], "sampling_limits": {"motion_blur": "", "occlusion": "", "camera_or_sampling_limits": "", "would_higher_fps_help": true}}.',
    systemPrompt: "You are a sports video analyst. Use only visible evidence and preserve timestamps.",
    reasoning: true,
    longVideoEnabled: true,
    parameters: {
      framesPerSecond: 6,
      maxTokens: 4096,
      presencePenalty: 0,
      repetitionPenalty: 1.2,
      temperature: 0.6,
      topK: 20,
      topP: 0.3
    }
  }
];

const ALPAMAYO_LINGOQA_SYSTEM_PROMPT =
  "You are Alpamayo 1.5 analyzing a LingoQA ego-camera driving clip. Answer the user's driving-scene question from visible evidence in the sampled frames.";

const ALPAMAYO_LINGOQA_EXAMPLES: ExampleItem[] = [
  {
    id: "lingoqa-red-light-slowdown",
    title: "AV LingoQA: slow for red light",
    mediaUrl: "/examples/lingoqa-red-light-slowdown.mp4",
    mediaName: "lingoqa-red-light-slowdown.mp4",
    mediaKind: "video",
    userPrompt: 'What is the current action and its justification? Answer in the form "action, justification".',
    systemPrompt: ALPAMAYO_LINGOQA_SYSTEM_PROMPT,
    reasoning: false,
    judgeNote: "Challenging ego-camera action example. Lingo-Judge: Cosmos3 0.854, Cosmos Reason 2 0.198, Cosmos Reason 1 0.199.",
    parameters: {
      framesPerSecond: 4,
      maxTokens: 256,
      temperature: 0.3,
      topP: 0.8
    }
  },
  {
    id: "lingoqa-return-left-after-truck",
    title: "AV LingoQA: return left after truck",
    mediaUrl: "/examples/lingoqa-return-left-after-truck.mp4",
    mediaName: "lingoqa-return-left-after-truck.mp4",
    mediaKind: "video",
    userPrompt: 'What is the current action and its justification? Answer in the form "action, justification".',
    systemPrompt: ALPAMAYO_LINGOQA_SYSTEM_PROMPT,
    reasoning: false,
    judgeNote: "Challenging ego-camera lane-position example. Lingo-Judge: Cosmos3 0.870, Cosmos Reason 2 0.231, Cosmos Reason 1 0.316.",
    parameters: {
      framesPerSecond: 4,
      maxTokens: 256,
      temperature: 0.3,
      topP: 0.8
    }
  },
  {
    id: "lingoqa-green-light-accelerate",
    title: "AV LingoQA: accelerate on green",
    mediaUrl: "/examples/lingoqa-green-light-accelerate.mp4",
    mediaName: "lingoqa-green-light-accelerate.mp4",
    mediaKind: "video",
    userPrompt: 'What is the current action and its justification? Answer in the form "action, justification".',
    systemPrompt: ALPAMAYO_LINGOQA_SYSTEM_PROMPT,
    reasoning: false,
    judgeNote: "Challenging ego-camera traffic-control example. Lingo-Judge: Cosmos3 0.781, Cosmos Reason 2 0.277, Cosmos Reason 1 0.263.",
    parameters: {
      framesPerSecond: 4,
      maxTokens: 256,
      temperature: 0.3,
      topP: 0.8
    }
  },
  {
    id: "lingoqa-no-cycle-lane",
    title: "AV LingoQA: no dedicated cycle lane",
    mediaUrl: "/examples/lingoqa-no-cycle-lane.mp4",
    mediaName: "lingoqa-no-cycle-lane.mp4",
    mediaKind: "video",
    userPrompt: "Is there a designated cycle lane on this road? If yes, where is it?",
    systemPrompt: ALPAMAYO_LINGOQA_SYSTEM_PROMPT,
    reasoning: false,
    judgeNote: "Challenging ego-camera road-layout example. Lingo-Judge: Cosmos3 0.799, Cosmos Reason 2 0.094, Cosmos Reason 1 0.086.",
    parameters: {
      framesPerSecond: 4,
      maxTokens: 256,
      temperature: 0.3,
      topP: 0.8
    }
  },
  {
    id: "lingoqa-no-traffic-lights",
    title: "AV LingoQA: no traffic lights",
    mediaUrl: "/examples/lingoqa-no-traffic-lights.mp4",
    mediaName: "lingoqa-no-traffic-lights.mp4",
    mediaKind: "video",
    userPrompt: "Are there any traffic lights? What color are they showing?",
    systemPrompt: ALPAMAYO_LINGOQA_SYSTEM_PROMPT,
    reasoning: false,
    judgeNote: "Challenging ego-camera scene-understanding example. Lingo-Judge: Cosmos3 0.576, Cosmos Reason 2 0.037, Cosmos Reason 1 0.078.",
    parameters: {
      framesPerSecond: 4,
      maxTokens: 256,
      temperature: 0.3,
      topP: 0.8
    }
  }
];

const AV_DENSE_CAPTIONING_EXAMPLES: ExampleItem[] = ALPAMAYO_LINGOQA_EXAMPLES.map((example) => ({
  ...example,
  group: "av-dense-captioning",
  reasoning: true,
  parameters: {
    ...example.parameters,
    maxTokens: 4096,
    presencePenalty: 0,
    repetitionPenalty: 1.0,
    temperature: 0.6,
    topK: 20,
    topP: 0.95
  }
}));

const DEFAULT_REASON_EXAMPLES: ExampleItem[] = [...EXAMPLES, ...AV_DENSE_CAPTIONING_EXAMPLES];

function readBlobAsDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

function readFileAsDataUrl(file: File): Promise<string> {
  return readBlobAsDataUrl(file);
}

async function fetchBlobWithProgress(
  url: string,
  onProgress?: (payload: { loaded: number; total?: number; percent?: number; etaSeconds?: number | null }) => void
) {
  const startedAt = performance.now();
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not load example media (${response.status})`);
  }
  const total = Number(response.headers.get("content-length")) || undefined;
  if (!response.body) {
    const blob = await response.blob();
    onProgress?.({ loaded: blob.size, total: total || blob.size, percent: 100, etaSeconds: 0 });
    return { blob, mime: response.headers.get("content-type") || blob.type };
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      loaded += value.byteLength;
      const elapsed = Math.max(0.001, (performance.now() - startedAt) / 1000);
      const rate = loaded / elapsed;
      const etaSeconds = total && rate > 0 ? Math.max(0, (total - loaded) / rate) : null;
      onProgress?.({
        loaded,
        total,
        percent: total ? Math.min(100, (loaded / total) * 100) : undefined,
        etaSeconds
      });
    }
  }
  const blob = new Blob(chunks, { type: response.headers.get("content-type") || undefined });
  onProgress?.({ loaded, total: total || loaded, percent: 100, etaSeconds: 0 });
  return { blob, mime: response.headers.get("content-type") || blob.type };
}

function withReasoningInstruction(prompt: string): string {
  const trimmed = prompt.trim();
  if (!trimmed) return "";
  if (trimmed.includes("<think>") || trimmed.includes(REASONING_FORMAT_INSTRUCTION)) return prompt;
  return `${trimmed}\n\n${REASONING_FORMAT_INSTRUCTION}`;
}

function withoutReasoningInstruction(prompt: string): string {
  return prompt
    .replace(REASONING_FORMAT_INSTRUCTION, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function promptForReasoning(prompt: string, enabled: boolean): string {
  return enabled ? withReasoningInstruction(prompt) : withoutReasoningInstruction(prompt);
}

function inferKind(file: File): "video" | "image" {
  if (file.type.startsWith("image/")) return "image";
  return "video";
}

function isAcceptedFile(file: File): boolean {
  if (file.type.startsWith("image/") || file.type.startsWith("video/")) return true;
  const extension = file.name.split(".").pop()?.toLowerCase() || "";
  return ACCEPTED_MEDIA_EXTENSIONS.includes(extension);
}

function stripAnswerTags(text: string): string {
  return text.replace(/^<answer>\s*/i, "").replace(/\s*<\/answer>$/i, "").trim();
}

function parseReasoning(content?: string, explicitReasoning?: string) {
  const text = content || "";
  const match = text.match(/<think>([\s\S]*?)<\/think>/i);
  const reasoning = (explicitReasoning || match?.[1] || "").trim();
  const answer = stripAnswerTags(match ? text.slice((match.index || 0) + match[0].length).trim() : text.trim());
  const steps = reasoning
    .split(/\n{2,}/)
    .map((step) => step.trim())
    .filter(Boolean);
  return { reasoning, answer, steps };
}

const POINT_KEYS = ["point_2d", "point", "position", "coordinate", "coordinates"];
const BBOX_KEYS = ["bbox_2d", "box_2d", "bounding_box", "bbox", "box"];
const TRAJECTORY_KEYS = ["annotations", "detections", "objects", "points", "steps", "trajectory"];
const COSMOS_COORD_MAX = 1000;

function isObjectRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function numberArray(value: unknown): number[] | null {
  if (Array.isArray(value)) {
    const numbers = value.map((item) => Number(item));
    return numbers.every((item) => Number.isFinite(item)) ? numbers : null;
  }
  if (isObjectRecord(value)) {
    const x = Number(value.x ?? value.left ?? value.cx);
    const y = Number(value.y ?? value.top ?? value.cy);
    const width = Number(value.width ?? value.w);
    const height = Number(value.height ?? value.h);
    if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(width) && Number.isFinite(height)) {
      return [x, y, width, height];
    }
    if (Number.isFinite(x) && Number.isFinite(y)) return [x, y];
  }
  return null;
}

function parseJsonCandidate(candidate: string): unknown | null {
  const trimmed = candidate
    .trim()
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/```$/i, "")
    .trim();
  if (!trimmed) return null;
  try {
    return JSON.parse(trimmed);
  } catch {
    return null;
  }
}

function collectBalancedJsonCandidates(text: string): string[] {
  const candidates: string[] = [];
  const openers: Record<string, string> = { "{": "}", "[": "]" };
  const closers = new Set(Object.values(openers));

  for (let start = 0; start < text.length; start += 1) {
    const opener = text[start];
    const firstCloser = openers[opener];
    if (!firstCloser) continue;

    const stack = [firstCloser];
    let inString = false;
    let escaping = false;

    for (let index = start + 1; index < text.length; index += 1) {
      const char = text[index];

      if (inString) {
        if (escaping) {
          escaping = false;
        } else if (char === "\\") {
          escaping = true;
        } else if (char === "\"") {
          inString = false;
        }
        continue;
      }

      if (char === "\"") {
        inString = true;
        continue;
      }

      const nextCloser = openers[char];
      if (nextCloser) {
        stack.push(nextCloser);
        continue;
      }

      if (closers.has(char)) {
        if (stack.pop() !== char) break;
        if (stack.length === 0) {
          candidates.push(text.slice(start, index + 1));
          start = index;
          break;
        }
      }
    }
  }

  return candidates;
}

function parseJsonPayloads(text: string): unknown[] {
  const payloads: unknown[] = [];
  const seen = new Set<string>();

  function addCandidate(candidate: string) {
    const parsed = parseJsonCandidate(candidate);
    if (parsed === null) return;
    const key = JSON.stringify(parsed);
    if (seen.has(key)) return;
    seen.add(key);
    payloads.push(parsed);
  }

  const trimmed = text.trim();
  if (trimmed) addCandidate(trimmed);

  for (const match of text.matchAll(/```(?:json)?\s*([\s\S]*?)```/gi)) {
    addCandidate(match[1]);
  }

  collectBalancedJsonCandidates(text).forEach(addCandidate);
  return payloads;
}

function parseResponseJsonPayload(text: string): unknown | null {
  const firstPayload = parseJsonPayloads(text)[0];
  if (firstPayload !== undefined) return firstPayload;

  const trimmed = text.trim();
  if (!trimmed) return null;

  const fenced = [...text.matchAll(/```(?:json)?\s*([\s\S]*?)```/gi)];
  if (fenced.length > 0) {
    const parsedFence = parseJsonCandidate(fenced[fenced.length - 1][1]);
    if (parsedFence !== null) return parsedFence;
  }

  const direct = parseJsonCandidate(trimmed);
  if (direct !== null) return direct;

  const firstObject = text.indexOf("{");
  const lastObject = text.lastIndexOf("}");
  if (firstObject >= 0 && lastObject > firstObject) {
    const parsedObject = parseJsonCandidate(text.slice(firstObject, lastObject + 1));
    if (parsedObject !== null) return parsedObject;
  }

  const firstArray = text.indexOf("[");
  const lastArray = text.lastIndexOf("]");
  if (firstArray >= 0 && lastArray > firstArray) return parseJsonCandidate(text.slice(firstArray, lastArray + 1));

  return null;
}

function collectSpatialRecords(value: unknown, records: Record<string, unknown>[] = [], depth = 0) {
  if (depth > 8) return records;
  if (Array.isArray(value)) {
    value.forEach((item) => collectSpatialRecords(item, records, depth + 1));
    return records;
  }
  if (!isObjectRecord(value)) return records;

  const hasSpatialField = [...POINT_KEYS, ...BBOX_KEYS].some((key) => value[key] !== undefined);
  if (hasSpatialField) records.push(value);

  for (const key of TRAJECTORY_KEYS) {
    const nested = value[key];
    if (nested !== undefined) collectSpatialRecords(nested, records, depth + 1);
  }
  return records;
}

function scaleCoordinate(value: number, axisSize: number, mode: "cosmos-1000" | "unit" | "pixel") {
  if (mode === "unit") return value * axisSize;
  if (mode === "cosmos-1000") return (value / COSMOS_COORD_MAX) * axisSize;
  return value;
}

function inferPointCoordinateMode(numbers: number[], key: string): "cosmos-1000" | "unit" | "pixel" {
  if (numbers.every((item) => Math.abs(item) <= 1)) return "unit";
  const lowerKey = key.toLowerCase();
  const looksLikeCosmosKey = lowerKey.includes("2d") || lowerKey.includes("coordinate") || lowerKey.includes("position");
  const allInCosmosPlane = numbers.every((item) => item >= 0 && item <= COSMOS_COORD_MAX);
  return looksLikeCosmosKey && allInCosmosPlane ? "cosmos-1000" : "pixel";
}

function scalePair(key: string, x: number, y: number, imageSize: ImageSize): [number, number, "cosmos-1000" | "unit" | "pixel"] {
  const mode = inferPointCoordinateMode([x, y], key);
  return [scaleCoordinate(x, imageSize.width, mode), scaleCoordinate(y, imageSize.height, mode), mode];
}

function normalizePoint(key: string, value: unknown, imageSize: ImageSize): [number, number, "cosmos-1000" | "unit" | "pixel"] | null {
  const numbers = numberArray(value);
  if (!numbers || numbers.length < 2) return null;
  const [x, y, mode] = scalePair(key, numbers[0], numbers[1], imageSize);
  const clampedX = Math.max(0, Math.min(imageSize.width, x));
  const clampedY = Math.max(0, Math.min(imageSize.height, y));
  return Number.isFinite(clampedX) && Number.isFinite(clampedY) ? [clampedX, clampedY, mode] : null;
}

function inferBboxCoordinateMode(key: string, numbers: number[]): "cosmos-1000" | "unit" | "pixel" {
  if (numbers.every((item) => Math.abs(item) <= 1)) return "unit";
  const lowerKey = key.toLowerCase();
  const allInCosmosPlane = numbers.every((item) => item >= 0 && item <= COSMOS_COORD_MAX);
  const explicitlyCosmos = lowerKey.includes("2d") || lowerKey === "box" || lowerKey === "bounding_box";
  return explicitlyCosmos && allInCosmosPlane ? "cosmos-1000" : "pixel";
}

function isCosmosXyxyBbox(key: string, mode: "cosmos-1000" | "unit" | "pixel") {
  const lowerKey = key.toLowerCase();
  return mode === "cosmos-1000" || lowerKey.includes("2d") || lowerKey === "bounding_box" || lowerKey === "box";
}

function clampRect(x: number, y: number, width: number, height: number, imageSize: ImageSize): [number, number, number, number] | null {
  const x1 = Math.max(0, Math.min(imageSize.width, x));
  const y1 = Math.max(0, Math.min(imageSize.height, y));
  const x2 = Math.max(0, Math.min(imageSize.width, x + width));
  const y2 = Math.max(0, Math.min(imageSize.height, y + height));
  const nextWidth = x2 - x1;
  const nextHeight = y2 - y1;
  if (![x1, y1, nextWidth, nextHeight].every(Number.isFinite) || nextWidth <= 0 || nextHeight <= 0) return null;
  return [x1, y1, nextWidth, nextHeight];
}

function normalizeBbox(
  key: string,
  value: unknown,
  imageSize: ImageSize
): [number, number, number, number, "cosmos-1000" | "unit" | "pixel"] | null {
  const numbers = numberArray(value);
  if (!numbers || numbers.length < 4) return null;
  const mode = inferBboxCoordinateMode(key, numbers);
  let [x, y, width, height] = numbers;
  if (isCosmosXyxyBbox(key, mode)) {
    const x1 = scaleCoordinate(numbers[0], imageSize.width, mode);
    const y1 = scaleCoordinate(numbers[1], imageSize.height, mode);
    const x2 = scaleCoordinate(numbers[2], imageSize.width, mode);
    const y2 = scaleCoordinate(numbers[3], imageSize.height, mode);
    const rect = clampRect(x1, y1, x2 - x1, y2 - y1, imageSize);
    return rect ? [...rect, mode] : null;
  }

  x = scaleCoordinate(x, imageSize.width, mode);
  y = scaleCoordinate(y, imageSize.height, mode);
  width = scaleCoordinate(width, imageSize.width, mode);
  height = scaleCoordinate(height, imageSize.height, mode);
  const rect = clampRect(x, y, width, height, imageSize);
  return rect ? [...rect, mode] : null;
}

function stringField(record: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value.trim()) return value.trim();
    if (typeof value === "number" && Number.isFinite(value)) return String(value);
  }
  return "";
}

function spatialLabel(record: Record<string, unknown>, fallback: string) {
  return (
    stringField(record, [
      "label",
      "category_name",
      "name",
      "class",
      "object",
      "target",
      "action",
      "description",
      "stage",
      "step_label"
    ]) || fallback
  );
}

function spatialDetail(record: Record<string, unknown>, label: string) {
  const detail = stringField(record, ["reason", "rationale", "thought", "observation", "note", "description", "action"]);
  return detail && detail !== label ? detail : undefined;
}

function spatialSequence(record: Record<string, unknown>, fallback: number) {
  const value = Number(record.sequence ?? record.step ?? record.index ?? record.id);
  return Number.isFinite(value) ? value : fallback;
}

function hasSpatialPayload(text: string) {
  return parseJsonPayloads(text).some((payload) => collectSpatialRecords(payload).length > 0);
}

function parseSpatialMarks(text: string, imageSize: ImageSize, source: SpatialSource) {
  const records = parseJsonPayloads(text).flatMap((payload) => collectSpatialRecords(payload));
  const marks: SpatialMark[] = [];
  let fallbackSequence = 1;
  records.forEach((record, index) => {
    const sequence = spatialSequence(record, fallbackSequence);
    fallbackSequence += 1;
    const label = spatialLabel(record, `point ${sequence}`);
    const detail = spatialDetail(record, label);

    for (const key of POINT_KEYS) {
      const point = normalizePoint(key, record[key], imageSize);
      if (!point) continue;
      const side = Math.max(24, Math.min(imageSize.width, imageSize.height) * 0.055);
      const [centerX, centerY, coordinateMode] = point;
      marks.push({
        id: `${source}-${key}-${index}-point`,
        kind: "point",
        sourceKey: key,
        coordinateMode,
        source,
        x: centerX - side / 2,
        y: centerY - side / 2,
        width: side,
        height: side,
        centerX,
        centerY,
        label,
        detail,
        sequence
      });
      break;
    }

    for (const key of BBOX_KEYS) {
      const bbox = normalizeBbox(key, record[key], imageSize);
      if (!bbox) continue;
      const [x, y, width, height, coordinateMode] = bbox;
      marks.push({
        id: `${source}-${key}-${index}-bbox`,
        kind: "bbox",
        sourceKey: key,
        coordinateMode,
        source,
        x,
        y,
        width,
        height,
        centerX: x + width / 2,
        centerY: y + height / 2,
        label,
        detail,
        sequence
      });
      break;
    }
  });
  return marks.slice(0, 160);
}

function makeStreamState(model: string): StreamState {
  return {
    phase: "waiting_first_token",
    reasoning: "",
    answer: "",
    schema: "plain_content",
    created: Math.floor(Date.now() / 1000),
    model
  };
}

function idleStreamState(model = DEFAULT_MODEL): StreamState {
  return {
    phase: "idle",
    reasoning: "",
    answer: "",
    schema: "plain_content",
    created: Math.floor(Date.now() / 1000),
    model
  };
}

function combinedContent(reasoning: string, answer: string) {
  return reasoning ? `<think>\n${reasoning.trim()}\n</think>\n\n${answer || ""}`.trim() : answer || "";
}

function streamStateToResult(streamState: StreamState): ApiResult | null {
  if (streamState.phase === "idle" || streamState.phase === "stopped") return null;
  if (streamState.phase === "error") {
    return { status: "error", message: streamState.message || "Backend stream failed" };
  }
  if (!streamState.reasoning && !streamState.answer && !streamState.raw) return null;
  if (streamState.raw && typeof streamState.raw === "object") return streamState.raw as ApiResult;
  const content = combinedContent(streamState.reasoning, streamState.answer);
  return {
    status: streamState.phase === "complete" ? "success" : undefined,
    content: streamState.answer,
    reasoning: streamState.reasoning,
    combined_content: content,
    schema: streamState.schema,
    openai: {
      id: "chatcmpl-byo-stream-preview",
      object: "chat.completion",
      created: streamState.created,
      model: streamState.model,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content
          },
          finish_reason: streamState.phase === "complete" ? "stop" : null
        }
      ],
      usage: streamState.usage || null
    }
  };
}

function parseSseBlock(block: string): ParsedSseEvent | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  if (dataLines.length === 0) return null;
  const dataText = dataLines.join("\n");
  return { event, data: JSON.parse(dataText) };
}

function drainSseEvents(buffer: string) {
  const blocks = buffer.split(/\r?\n\r?\n/);
  const rest = blocks.pop() || "";
  const events = blocks.map(parseSseBlock).filter((event): event is ParsedSseEvent => Boolean(event));
  return { events, rest };
}

function statusForPhase(phase: StreamPhase) {
  if (phase === "preparing_media") return "Preparing media";
  if (phase === "alpamayo_generating") return "Generating Alpamayo response";
  if (phase === "waiting_first_token") return "Waiting for first token";
  if (phase === "reasoning") return "Streaming reasoning";
  if (phase === "answer") return "Streaming response";
  if (phase === "complete") return "Complete";
  if (phase === "stopped") return "Stopped";
  if (phase === "error") return "Backend error";
  return "Ready";
}

function formatDuration(seconds?: number | null) {
  if (!Number.isFinite(Number(seconds))) return "unknown";
  const total = Math.max(0, Math.round(Number(seconds)));
  const minutes = Math.floor(total / 60);
  const remainder = total % 60;
  if (minutes <= 0) return `${remainder}s`;
  return `${minutes}m ${String(remainder).padStart(2, "0")}s`;
}

function formatBytes(bytes?: number | null) {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value <= 0) return "";
  const units = ["B", "KB", "MB", "GB"];
  let amount = value;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return `${amount >= 10 || unit === 0 ? amount.toFixed(0) : amount.toFixed(1)} ${units[unit]}`;
}

function mergeLongChunk(chunks: LongChunkProgress[], update: Partial<LongChunkProgress> & { index?: number }) {
  if (typeof update.index !== "number") return chunks;
  const next = chunks.slice();
  const current = next[update.index] || { index: update.index, status: "queued" as const };
  next[update.index] = { ...current, ...update, index: update.index } as LongChunkProgress;
  return next;
}

function parseJsonFromText(text?: string): unknown | null {
  const value = String(text || "").trim();
  if (!value) return null;
  const fenced = value.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = fenced ? fenced[1].trim() : value;
  const first = candidate.search(/[\[{]/);
  if (first < 0) return null;
  try {
    return JSON.parse(candidate.slice(first));
  } catch {
    return null;
  }
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

function asTimelineEvents(value: unknown): TimelineEvent[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is TimelineEvent => Boolean(asRecord(item)));
}

function eventsFromChunk(chunk: LongChunkProgress): TimelineEvent[] {
  if (Array.isArray(chunk.events) && chunk.events.length > 0) return asTimelineEvents(chunk.events);
  const parsed = asRecord(chunk.parsed) || asRecord(parseJsonFromText(chunk.content));
  return asTimelineEvents(parsed?.events);
}

function captionFromEvent(event: TimelineEvent, fallback = "") {
  return String(event.caption || event.summary || fallback || "").trim();
}

function typeFromEvent(event: TimelineEvent) {
  return String(event.event_type || event.type || "event").replace(/[_-]+/g, " ");
}

function rangeFromEvent(event: TimelineEvent, fallback = "") {
  const start = String(event.start || "").trim();
  const end = String(event.end || "").trim();
  if (start && end && end !== start) return `${start} - ${end}`;
  if (start) return start;
  return fallback || "timestamp unknown";
}

function timelineItemsFromEvents(events: TimelineEvent[], fallbackRange = "", fallbackCaption = ""): TimelineItem[] {
  return events.map((event, index) => {
    const confidence = event.confidence === undefined || event.confidence === "" ? "" : `confidence ${event.confidence}`;
    const player = event.player ? `player ${event.player}` : "";
    return {
      id: `${fallbackRange}-${index}-${event.start || ""}-${event.end || ""}`,
      range: rangeFromEvent(event, fallbackRange),
      title: typeFromEvent(event),
      caption: captionFromEvent(event, fallbackCaption),
      meta: [player, confidence].filter(Boolean).join(" · ")
    };
  });
}

function timelineItemsFromChunks(chunks: LongChunkProgress[], mode: "events" | "summaries") {
  if (mode === "events") {
    return chunks.flatMap((chunk) => {
      const events = eventsFromChunk(chunk);
      if (events.length === 0 && chunk.summary) {
        return [
          {
            id: `summary-${chunk.index}`,
            range: chunk.timeRange || `Chunk ${chunk.index + 1}`,
            title: "summary",
            caption: chunk.summary,
            meta: `chunk ${chunk.index + 1}`
          }
        ];
      }
      return timelineItemsFromEvents(events, chunk.timeRange, chunk.summary).map((item, index) => ({
        ...item,
        id: `chunk-${chunk.index}-${index}-${item.id}`,
        meta: [item.meta, `chunk ${chunk.index + 1}`].filter(Boolean).join(" · ")
      }));
    });
  }
  return chunks
    .filter((chunk) => chunk.summary)
    .map((chunk) => ({
      id: `summary-${chunk.index}`,
      range: chunk.timeRange || `Chunk ${chunk.index + 1}`,
      title: "summary",
      caption: chunk.summary || "",
      meta: `chunk ${chunk.index + 1}`
    }));
}

function stitchedTimelineItems(result: ApiResult | null): TimelineItem[] {
  if (!result) return [];
  const parsed = asRecord(parseJsonFromText(result.content));
  const finalEvents = timelineItemsFromEvents(asTimelineEvents(parsed?.events));
  if (finalEvents.length > 0) return finalEvents;
  const chunks = result.long_video?.chunks || [];
  return timelineItemsFromChunks(chunks, "events");
}

function looksLikeJsonResponse(text: string) {
  const trimmed = text.trim();
  return trimmed.startsWith("{") || trimmed.startsWith("[") || /^```json/i.test(trimmed);
}

function chunkProgressPercent(status: LongChunkProgress["status"]) {
  if (status === "done") return 100;
  if (status === "running") return 52;
  if (status === "error") return 100;
  return 0;
}

function safeJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function displayValue(value: unknown, fallback = "unknown") {
  if (value === undefined || value === null || value === "") return fallback;
  return String(value);
}

function shortSha(sha?: string | null) {
  return sha ? sha.slice(0, 12) : "unknown";
}

function formatTimestamp(value?: string | number | null) {
  if (!value) return "unknown";
  const date = typeof value === "number" ? new Date(value * 1000) : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short"
  });
}

function flagValue(backendInfo: BackendInfo | null, key: string) {
  return backendInfo?.vllm?.process?.flags?.[key];
}

function quantizationLabel(backendInfo: BackendInfo | null) {
  const quantization = backendInfo?.quantization;
  if (!quantization) return "unknown";
  if (quantization.applied) return quantization.method || "applied";
  return `None detected${quantization.dtype ? `, dtype ${quantization.dtype}` : ""}`;
}

function isVlaMode(modelName: string, backendInfo?: BackendInfo | null) {
  const lower = `${modelName} ${backendInfo?.display_name || ""} ${backendInfo?.checkpoint || ""}`.toLowerCase();
  return Boolean(backendInfo?.vla) || String(backendInfo?.backend || "").toLowerCase() === "alpamayo" || lower.includes("alpamayo");
}

function vlaFrameSummary(backendInfo?: BackendInfo | null) {
  const frames = backendInfo?.vla?.frames_per_video || 4;
  const decoded = backendInfo?.vla?.max_decoded_video_frames || 96;
  return `${frames} frame${frames === 1 ? "" : "s"} from the first ${decoded} decoded frames`;
}

function vlaModeLabel(backendInfo?: BackendInfo | null) {
  const mode = backendInfo?.vla?.current_mode || "vqa_text_generation";
  return mode.replace(/_/g, " ");
}

function heroDescription(modelName: string, backendInfo?: BackendInfo | null) {
  if (isVlaMode(modelName, backendInfo)) {
    return "Vision-language-action model served through the Alpamayo BYO adapter. Current user path is VQA/captioning over sampled video frames.";
  }
  return "Vision language model that excels in understanding the physical world using structured reasoning on videos or images.";
}

function modelDefaults(modelName: string, backendInfo?: BackendInfo | null) {
  if (isVlaMode(modelName, backendInfo)) {
    return { fps: 4, maxTokens: 256 };
  }
  const lower = modelName.toLowerCase();
  if (lower.includes("cosmos3") || lower.includes("c3-") || lower.includes("nano-reasoner")) {
    return { fps: 2, maxTokens: 512 };
  }
  if (lower.includes("32b") || lower.includes("super")) {
    return { fps: 1, maxTokens: 1024 };
  }
  return { fps: DEFAULT_FRAMES_PER_SECOND, maxTokens: DEFAULT_MAX_TOKENS };
}

function modelShortName(modelName: string) {
  const raw = String(modelName || "").trim();
  if (!raw || raw === "Detecting model...") return "Cosmos Reasoner";
  return raw.split("/").filter(Boolean).pop() || raw;
}

function modelSlug(modelName: string) {
  return modelShortName(modelName)
    .replace(/([a-z0-9])([A-Z])/g, "$1-$2")
    .replace(/[^A-Za-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .toLowerCase();
}

function modelFamilyLabel(modelName: string) {
  const lower = String(modelName || "").toLowerCase();
  if (lower.includes("cosmos3-super")) return "Cosmos3 Super Reasoner";
  if (lower.includes("cosmos3-nano")) return "Cosmos3 Nano Reasoner";
  if (lower.includes("cosmos-reason2")) return "Cosmos Reason2";
  if (lower.includes("cosmos-reason1")) return "Cosmos Reason1";
  if (lower.includes("nemotron")) return "Nemotron VL Reasoner";
  if (lower.includes("qwen")) return "Qwen VL Reasoner";
  if (lower.includes("gemma")) return "Gemma VL Reasoner";
  return modelShortName(modelName);
}

function nimImageForModel(modelName: string, backendInfo: BackendInfo | null) {
  const liveImage = backendInfo?.nim?.image;
  if (liveImage) return liveImage;
  const slug = modelSlug(modelName);
  return slug ? `nvcr.io/nim/nvidia/${slug}:latest` : "<NIM_IMAGE>";
}

function usesFrameFallback(modelName: string, backend?: string) {
  const lower = modelName.toLowerCase();
  if (backend === "alpamayo" || lower.includes("alpamayo")) return false;
  if (backend === "nim_local" && !lower.includes("cosmos-reason1")) return false;
  return !(
    lower.includes("nemotron") ||
    lower.includes("qwen3-vl") ||
    lower.includes("qwen3vl") ||
    lower.includes("qwen") ||
    lower.includes("cosmos3") ||
    lower.includes("cosmos-3") ||
    lower.includes("cosmos-reason2") ||
    lower.includes("cosmos-reason-2") ||
    lower.includes("alpamayo")
  );
}

export default function App() {
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const defaultExampleLoadedRef = useRef(false);
  const [activeTab, setActiveTab] = useState<SectionTab>("Experience");
  const [outputTab, setOutputTab] = useState<OutputTab>("preview");
  const [examplesOpen, setExamplesOpen] = useState(false);
  const [selectedExampleId, setSelectedExampleId] = useState(EXAMPLES[0].id);
  const [parametersOpen, setParametersOpen] = useState(false);
  const [runtimeOpen, setRuntimeOpen] = useState(false);
  const [reasoningExpanded, setReasoningExpanded] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [media, setMedia] = useState<MediaState | null>(null);
  const [userPrompt, setUserPrompt] = useState(DEFAULT_USER_PROMPT);
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT);
  const [reasoningEnabled, setReasoningEnabled] = useState(true);
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [models, setModels] = useState<string[]>([DEFAULT_MODEL]);
  const [backendInfo, setBackendInfo] = useState<BackendInfo | null>(() => initialBackendInfo());
  const [temperature, setTemperature] = useState(DEFAULT_TEMPERATURE);
  const [topP, setTopP] = useState(DEFAULT_TOP_P);
  const [topK, setTopK] = useState(DEFAULT_TOP_K);
  const [maxTokens, setMaxTokens] = useState(DEFAULT_MAX_TOKENS);
  const [framesPerSecond, setFramesPerSecond] = useState(DEFAULT_FRAMES_PER_SECOND);
  const [repetitionPenalty, setRepetitionPenalty] = useState(DEFAULT_REPETITION_PENALTY);
  const [presencePenalty, setPresencePenalty] = useState(DEFAULT_PRESENCE_PENALTY);
  const [seed, setSeed] = useState(DEFAULT_SEED);
  const [status, setStatus] = useState("Ready");
  const [isRunning, setIsRunning] = useState(false);
  const [runMode, setRunMode] = useState<RunMode>(null);
  const [longPreset, setLongPreset] = useState<LongPreset>("balanced");
  const [longConcurrency, setLongConcurrency] = useState(LONG_VIDEO_DEFAULT_CONCURRENCY);
  const [exampleLoad, setExampleLoad] = useState<ExampleLoadState | null>(null);
  const [longProgress, setLongProgress] = useState<LongProgressState | null>(null);
  const [result, setResult] = useState<ApiResult | null>(null);
  const [streamState, setStreamState] = useState<StreamState>(() => idleStreamState());
  const [copied, setCopied] = useState(false);
  const vlaMode = isVlaMode(model, backendInfo);
  const activeExamples = useMemo(() => (vlaMode ? ALPAMAYO_LINGOQA_EXAMPLES : DEFAULT_REASON_EXAMPLES), [vlaMode]);

  useEffect(() => {
    let cancelled = false;

    async function refreshActiveModel() {
      try {
        const [activeResponse, modelsResponse] = await Promise.all([
          fetch(COSMOS3_INFO_URL, { cache: "no-store" }),
          fetch("/api/models", { cache: "no-store" })
        ]);
        const active = activeResponse.ok ? await activeResponse.json() : null;
        const listed = modelsResponse.ok ? await modelsResponse.json() : null;
        if (cancelled) return;

        const activeName = (active?.checkpoint as string | undefined) || (active?.display_name as string | undefined);
        const listedModels = Array.isArray(listed?.models) ? (listed.models as string[]).filter(Boolean) : [];
        const nextModels = activeName ? [activeName, ...listedModels.filter((item) => item !== activeName)] : listedModels;

        if (nextModels.length > 0) {
          setModels(nextModels);
          setModel(nextModels[0]);
        } else if (activeName) {
          setModel(activeName);
          setModels([activeName]);
        }
        if (active) setBackendInfo(active);
      } catch {
        // Keep the last known model visible if the backend is mid-restart.
      }
    }

    void refreshActiveModel();
    const timer = window.setInterval(refreshActiveModel, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (!model || model === DEFAULT_MODEL) return;
    document.title = `${model} | ${isVlaMode(model, backendInfo) ? "VLA BYO" : "NVIDIA NIM"}`;
  }, [backendInfo, model]);

  useEffect(() => {
    const example = activeExamples.find((item) => item.id === selectedExampleId);
    if (example) {
      applyExampleParameters(example);
      return;
    }
    if (activeExamples[0]) {
      void applyExample(activeExamples[0].id, { closeModal: false });
      return;
    }
    const defaults = modelDefaults(model, backendInfo);
    setFramesPerSecond(defaults.fps);
    setMaxTokens(defaults.maxTokens);
  }, [activeExamples, backendInfo, model, selectedExampleId]);

  useEffect(() => {
    if (defaultExampleLoadedRef.current) return;
    defaultExampleLoadedRef.current = true;
    void applyExample(activeExamples[0].id, { closeModal: false });
  }, [activeExamples]);

  const effectivePrompt = useMemo(
    () => promptForReasoning(userPrompt || "Describe the provided media.", reasoningEnabled),
    [reasoningEnabled, userPrompt]
  );

  const requestPreview = useMemo(
    () => ({
      name: "req-<timestamp>",
      model,
      messages: [
        ...(systemPrompt ? [{ role: "system", content: systemPrompt }] : []),
        {
          role: "user",
	      content: [
	            ...(media
	              ? [
	                  media.kind === "image"
	                    ? { type: "image_url", image_url: { url: "data:image/<type>;base64,<payload>" } }
	                    : usesFrameFallback(model, backendInfo?.backend)
	                      ? { type: "text", text: `[Video — sampled frames at ${framesPerSecond}fps]\n${effectivePrompt}` }
	                      : { type: "video_url", video_url: { url: "data:video/mp4;base64,<payload>" } }
	                ]
	              : []),
	            ...(media?.kind === "video" && usesFrameFallback(model, backendInfo?.backend)
	              ? [{ type: "image_url", image_url: { url: "data:image/jpeg;base64,<sampled-frame>" } }]
	              : [{ type: "text", text: effectivePrompt }])
	          ]
	        }
	      ],
	      temperature,
	      top_p: topP,
	      top_k: topK,
	      presence_penalty: presencePenalty,
	      max_tokens: maxTokens,
	      repetition_penalty: repetitionPenalty,
	      seed,
	      stream: true,
	      media_sampling:
	        media?.kind === "video"
	          ? {
	              mode: isVlaMode(model, backendInfo)
	                ? "alpamayo_adapter_video_url"
	                : usesFrameFallback(model, backendInfo?.backend)
	                  ? "image-frame-fallback"
	                  : "video_url",
	              fps: isVlaMode(model, backendInfo) ? undefined : framesPerSecond,
	              adapter_frames: isVlaMode(model, backendInfo) ? backendInfo?.vla?.frames_per_video || 4 : undefined,
	              adapter_max_decoded_frames: isVlaMode(model, backendInfo)
	                ? backendInfo?.vla?.max_decoded_video_frames || 96
	                : undefined,
	              note: isVlaMode(model, backendInfo)
	                ? "Alpamayo samples frames inside the adapter; the Vite FPS slider is only used by frame-fallback paths."
	                : undefined
	            }
	          : undefined
	    }),
    [
	      effectivePrompt,
	      framesPerSecond,
	      backendInfo?.backend,
	      backendInfo?.vla,
	      media,
	      model,
	      maxTokens,
	      presencePenalty,
	      repetitionPenalty,
	      seed,
	      systemPrompt,
	      temperature,
	      topK,
	      topP
	    ]
	  );

  const streamResult = useMemo(() => streamStateToResult(streamState), [streamState]);
  const activeResult = result || streamResult;
  const parsedOutput = useMemo(
    () => parseReasoning(activeResult?.content, activeResult?.reasoning),
    [activeResult?.content, activeResult?.reasoning]
  );
  const jsonOutput = useMemo(
    () => activeResult?.openai || activeResult?.raw || activeResult || requestPreview,
    [activeResult, requestPreview]
  );

  async function setFileMedia(file: File) {
    if (!isAcceptedFile(file)) {
      setStatus("Unsupported file type");
      return;
    }
    const kind = inferKind(file);
    setExampleLoad(null);
    setMedia({
      name: file.name,
      kind,
      previewUrl: URL.createObjectURL(file),
      dataUrl: await readFileAsDataUrl(file)
    });
    setResult(null);
    setStreamState(idleStreamState(model));
    setOutputTab("preview");
    setStatus("Media loaded");
  }

  async function handleFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) await setFileMedia(file);
  }

  async function handleDrop(event: DragEvent<HTMLElement>) {
    event.preventDefault();
    setDragActive(false);
    const file = event.dataTransfer.files?.[0];
    if (file) await setFileMedia(file);
  }

  function handleDrag(event: DragEvent<HTMLElement>, active: boolean) {
    event.preventDefault();
    setDragActive(active);
  }

  async function mediaFromExample(
    example: ExampleItem,
    onProgress?: (state: Omit<ExampleLoadState, "title" | "elapsedSeconds">) => void
  ): Promise<MediaState> {
    if (example.mediaUrl.startsWith("/")) {
      const sourceUrl = new URL(example.mediaUrl, window.location.origin).toString();
      if (example.mediaKind === "video") {
        let bytesTotal: number | undefined;
        try {
          const head = await fetch(example.mediaUrl, { method: "HEAD" });
          bytesTotal = Number(head.headers.get("content-length")) || undefined;
        } catch {
          bytesTotal = undefined;
        }
        onProgress?.({
          phase: "Ready",
          detail: `Using URL stream${bytesTotal ? ` (${formatBytes(bytesTotal)})` : ""}; skipped browser base64 encoding`,
          percent: 100,
          etaSeconds: 0,
          bytesLoaded: bytesTotal,
          bytesTotal,
          steps: [
            { key: "locate", label: "Locate media", status: "done", progress: 100, detail: "Found server-hosted file" },
            { key: "stream", label: "Prepare stream URL", status: "done", progress: 100, detail: "Preview and backend will stream by URL" },
            { key: "ready", label: "Ready", status: "done", progress: 100, detail: "No browser copy required" }
          ]
        });
        return {
          name: example.mediaName,
          kind: "video",
          previewUrl: example.mediaUrl,
          dataUrl: "",
          sourceUrl
        };
      }

      const { blob, mime: responseMime } = await fetchBlobWithProgress(example.mediaUrl, (progress) => {
        onProgress?.({
          phase: "Downloading",
          detail: progress.total
            ? `${formatBytes(progress.loaded)} of ${formatBytes(progress.total)}`
            : `${formatBytes(progress.loaded)} downloaded`,
          percent: progress.percent ?? 45,
          etaSeconds: progress.etaSeconds,
          bytesLoaded: progress.loaded,
          bytesTotal: progress.total,
          steps: [
            { key: "locate", label: "Locate media", status: "done", progress: 100, detail: "Found server-hosted file" },
            { key: "download", label: "Download for request", status: "running", progress: progress.percent ?? 45, detail: "Reading browser payload" },
            { key: "encode", label: "Encode request payload", status: "pending", progress: 0 },
            { key: "ready", label: "Ready", status: "pending", progress: 0 }
          ]
        });
      });
      onProgress?.({
        phase: "Encoding",
        detail: "Preparing request payload",
        percent: 92,
        etaSeconds: null,
        bytesLoaded: blob.size,
        bytesTotal: blob.size,
        steps: [
          { key: "locate", label: "Locate media", status: "done", progress: 100 },
          { key: "download", label: "Download for request", status: "done", progress: 100, detail: formatBytes(blob.size) },
          { key: "encode", label: "Encode request payload", status: "running", progress: 70 },
          { key: "ready", label: "Ready", status: "pending", progress: 0 }
        ]
      });
      const mime = blob.type || responseMime || (example.mediaKind === "image" ? "image/png" : "video/mp4");
      return {
        name: example.mediaName,
        kind: mime.startsWith("image/") ? "image" : "video",
        previewUrl: example.mediaUrl,
        dataUrl: await readBlobAsDataUrl(blob),
        sourceUrl
      };
    }

    const params = new URLSearchParams({ url: example.mediaUrl, name: example.mediaName });
    onProgress?.({
      phase: "Fetching",
      detail: "Proxying remote example media",
      percent: 25,
      etaSeconds: null,
      steps: [
        { key: "proxy", label: "Fetch remote media", status: "running", progress: 25 },
        { key: "encode", label: "Encode request payload", status: "pending", progress: 0 },
        { key: "ready", label: "Ready", status: "pending", progress: 0 }
      ]
    });
    const response = await fetch(`/api/example-media?${params.toString()}`);
    if (!response.ok) {
      const error = await response.json().catch(() => null);
      throw new Error(error?.message || "Could not load example media");
    }
    onProgress?.({
      phase: "Encoding",
      detail: "Preparing proxied media",
      percent: 90,
      etaSeconds: null,
      steps: [
        { key: "proxy", label: "Fetch remote media", status: "done", progress: 100 },
        { key: "encode", label: "Encode request payload", status: "running", progress: 80 },
        { key: "ready", label: "Ready", status: "pending", progress: 0 }
      ]
    });
    const data = (await response.json()) as { dataUrl: string; mime: string; name: string };
    return {
      name: data.name,
      kind: data.mime.startsWith("image/") ? "image" : "video",
      previewUrl: example.mediaUrl,
      dataUrl: data.dataUrl,
      sourceUrl: example.mediaUrl
    };
  }

  async function applyExample(exampleId = selectedExampleId, options: { closeModal?: boolean } = {}) {
    const closeModal = options.closeModal ?? true;
    const example = activeExamples.find((item) => item.id === exampleId) || activeExamples[0] || EXAMPLES[0];
    const startedAt = performance.now();
    const setLoadProgress = (state: Omit<ExampleLoadState, "title" | "elapsedSeconds">) => {
      setExampleLoad({
        ...state,
        title: example.title,
        percent: Math.max(0, Math.min(100, Math.round(Number(state.percent) || 0))),
        elapsedSeconds: (performance.now() - startedAt) / 1000
      });
    };
    setSelectedExampleId(example.id);
    if (closeModal) setExamplesOpen(false);
    setStatus("Loading example");
    setLoadProgress({
      phase: "Starting",
      detail: example.mediaKind === "video" && example.mediaUrl.startsWith("/")
        ? "Checking server-hosted video"
        : "Preparing example media",
      percent: 5,
      etaSeconds: null,
      steps: [
        { key: "locate", label: "Locate media", status: "running", progress: 20 },
        { key: "download", label: "Download or stream", status: "pending", progress: 0 },
        { key: "ready", label: "Ready", status: "pending", progress: 0 }
      ]
    });
    try {
      setReasoningEnabled(example.reasoning);
      setUserPrompt(promptForReasoning(example.userPrompt, example.reasoning));
      setSystemPrompt(example.systemPrompt);
      applyExampleParameters(example);
      const nextMedia = await mediaFromExample(example, setLoadProgress);
      setMedia(nextMedia);
      setResult(null);
      setStreamState(idleStreamState(model));
      setOutputTab("preview");
      setStatus(
        nextMedia.kind === "video" && nextMedia.sourceUrl
          ? "Example loaded: video will stream by URL"
          : "Example loaded"
      );
      window.setTimeout(() => setExampleLoad(null), 1200);
    } catch (error) {
      setExampleLoad(null);
      setStatus(error instanceof Error ? error.message : "Example failed to load");
    }
  }

  function setReasoning(next: boolean) {
    setReasoningEnabled(next);
    setUserPrompt((current) => promptForReasoning(current, next));
    applySamplingDefaults(next);
  }

  function applySamplingDefaults(useReasoning: boolean) {
    const defaults = useReasoning ? SAMPLING_DEFAULTS.reasoning : SAMPLING_DEFAULTS.standard;
    setTemperature(defaults.temperature);
    setTopP(defaults.topP);
    setTopK(defaults.topK);
    setRepetitionPenalty(defaults.repetitionPenalty);
    setPresencePenalty(defaults.presencePenalty);
  }

  function applyExampleParameters(example: ExampleItem) {
    const samplingDefaults = example.reasoning ? SAMPLING_DEFAULTS.reasoning : SAMPLING_DEFAULTS.standard;
    const mediaDefaults = modelDefaults(model, backendInfo);
    const params = example.parameters || {};
    setTemperature(params.temperature ?? samplingDefaults.temperature);
    setTopP(params.topP ?? samplingDefaults.topP);
    setTopK(params.topK ?? samplingDefaults.topK);
    setRepetitionPenalty(params.repetitionPenalty ?? samplingDefaults.repetitionPenalty);
    setPresencePenalty(params.presencePenalty ?? samplingDefaults.presencePenalty);
    setFramesPerSecond(example.longVideoEnabled ? LONG_VIDEO_PRESET_FPS[longPreset] : (params.framesPerSecond ?? mediaDefaults.fps));
    if (example.longVideoEnabled) setLongConcurrency(LONG_VIDEO_DEFAULT_CONCURRENCY);
    setMaxTokens(params.maxTokens ?? mediaDefaults.maxTokens);
  }

  function reset() {
    abortRef.current?.abort();
    abortRef.current = null;
    setMedia(null);
    setUserPrompt(DEFAULT_USER_PROMPT);
    setSystemPrompt(DEFAULT_SYSTEM_PROMPT);
    setReasoningEnabled(true);
    setTemperature(DEFAULT_TEMPERATURE);
    setTopP(DEFAULT_TOP_P);
    setTopK(DEFAULT_TOP_K);
    setMaxTokens(DEFAULT_MAX_TOKENS);
    setFramesPerSecond(DEFAULT_FRAMES_PER_SECOND);
    setLongPreset("balanced");
    setLongConcurrency(LONG_VIDEO_DEFAULT_CONCURRENCY);
    setRepetitionPenalty(DEFAULT_REPETITION_PENALTY);
    setPresencePenalty(DEFAULT_PRESENCE_PENALTY);
    setSeed(DEFAULT_SEED);
    setParametersOpen(false);
    setReasoningExpanded(false);
    setOutputTab("preview");
    setStatus("Ready");
    setResult(null);
    setStreamState(idleStreamState(model));
    setIsRunning(false);
    setRunMode(null);
    setExampleLoad(null);
    setLongProgress(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  async function run() {
    if (isRunning) {
      abortRef.current?.abort();
      setStatus("Stopping task");
      return;
    }

    const controller = new AbortController();
    abortRef.current = controller;
    setIsRunning(true);
    setRunMode("standard");
    setLongProgress(null);
    setResult(null);
    const initialStreamState = makeStreamState(model);
    setStreamState(initialStreamState);
    setOutputTab("preview");
    setReasoningExpanded(false);
    setStatus(statusForPhase("waiting_first_token"));

    let reasoning = "";
    let answer = "";
    let schema = "plain_content";
    let usage: unknown = null;
    let rawResult: ApiResult | null = null;

    try {
      const response = await fetch("/api/reason/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          prompt: effectivePrompt,
          systemPrompt,
          model,
          video: media?.kind === "video" ? media.sourceUrl || media.dataUrl : undefined,
          image: media?.kind === "image" ? media.dataUrl : undefined,
            params: {
              temperature,
              top_p: topP,
              top_k: topK,
              presence_penalty: presencePenalty,
              max_tokens: maxTokens,
              frames_per_second: framesPerSecond,
              repetition_penalty: repetitionPenalty,
              seed
            }
        })
      });
      if (!response.ok) {
        const error = await response.text();
        throw new Error(error || `Reasoner stream returned HTTP ${response.status}`);
      }
      if (!response.body) throw new Error("Reasoner stream did not include a response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const drained = drainSseEvents(buffer);
        buffer = drained.rest;

        for (const event of drained.events) {
          if (event.event === "state") {
            const data = event.data as { phase?: StreamPhase };
            if (data.phase) {
              setStreamState((current) => ({ ...current, phase: data.phase || current.phase }));
              setStatus(statusForPhase(data.phase));
            }
          } else if (event.event === "delta") {
            const data = event.data as { channel?: "reasoning" | "answer"; text?: string; schema?: string };
            const text = data.text || "";
            schema = data.schema || schema;
            if (data.channel === "reasoning") {
              reasoning += text;
            } else {
              answer += text;
            }
            const phase: StreamPhase = data.channel === "reasoning" ? "reasoning" : "answer";
            setStreamState((current) => ({
              ...current,
              phase,
              reasoning,
              answer,
              schema
            }));
            setStatus(statusForPhase(phase));
          } else if (event.event === "usage") {
            usage = event.data;
            setStreamState((current) => ({ ...current, usage }));
          } else if (event.event === "raw") {
            rawResult = event.data as ApiResult;
            setStreamState((current) => ({ ...current, raw: rawResult || undefined }));
          } else if (event.event === "error") {
            const data = event.data as { message?: string };
            throw new Error(data.message || "Backend stream failed");
          }
        }

        if (done) break;
      }

      const finalState: StreamState = {
        ...initialStreamState,
        phase: "complete",
        reasoning,
        answer,
        schema,
        raw: rawResult || undefined,
        usage
      };
      setStreamState(finalState);
      setResult(rawResult || streamStateToResult(finalState));
      setStatus("Complete");
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        setResult({ status: "skip", message: "Task stopped by user" });
        setStreamState((current) => ({ ...current, phase: "stopped" }));
        setStatus("Stopped");
      } else {
        const message = error instanceof Error ? error.message : "Request failed";
        setResult({ status: "error", message });
        setStreamState((current) => ({ ...current, phase: "error", message }));
        setStatus("Request failed");
      }
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        setIsRunning(false);
        setRunMode(null);
      }
    }
  }

  async function runLongVideo() {
    if (isRunning) {
      abortRef.current?.abort();
      setStatus("Stopping task");
      return;
    }
    if (media?.kind !== "video") {
      setStatus("Long Video requires a video input");
      return;
    }

    const controller = new AbortController();
    abortRef.current = controller;
    setIsRunning(true);
    setRunMode("long");
    setResult(null);
    setOutputTab("preview");
    setReasoningExpanded(false);
    setStreamState(makeStreamState(model));
    setLongProgress({
      preset: longPreset,
      phase: "media_scan",
      message: "Preparing long video analysis",
      percent: 1,
      warnings: [],
      chunks: [],
      partialTimeline: []
    });
    setStatus("Long Video: preparing");

    try {
      const response = await fetch("/api/reason/long/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          prompt: effectivePrompt,
          systemPrompt,
          model,
          preset: longPreset,
          concurrency: longConcurrency,
          video: media.sourceUrl || media.dataUrl,
          params: {
            temperature,
            top_p: topP,
            top_k: topK,
            presence_penalty: presencePenalty,
            max_tokens: maxTokens,
            frames_per_second: framesPerSecond,
            repetition_penalty: repetitionPenalty,
            seed
          }
        })
      });
      if (!response.ok) {
        const error = await response.text();
        throw new Error(error || `Long video stream returned HTTP ${response.status}`);
      }
      if (!response.body) throw new Error("Long video stream did not include a response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let rawResult: ApiResult | null = null;

      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const drained = drainSseEvents(buffer);
        buffer = drained.rest;

        for (const event of drained.events) {
          if (event.event === "long_plan") {
            const data = event.data as Partial<LongProgressState> & { chunks?: LongChunkProgress[] };
            setLongProgress((current) => {
              const base = current || { preset: longPreset, warnings: [], chunks: [], partialTimeline: [] };
              return {
                ...base,
                ...data,
                preset: longPreset,
                chunks: data.chunks || base.chunks,
                warnings: data.warnings || base.warnings,
                partialTimeline: base.partialTimeline
              };
            });
            setStatus(`Long Video: planned ${data.totalChunks || 0} chunks`);
          } else if (event.event === "long_state") {
            const data = event.data as Partial<LongProgressState>;
            setLongProgress((current) => {
              const base = current || { preset: longPreset, warnings: [], chunks: [], partialTimeline: [] };
              return {
                ...base,
                ...data,
                preset: longPreset,
                warnings: base.warnings.length > 0 ? base.warnings : data.warnings || [],
                chunks: base.chunks,
                partialTimeline: base.partialTimeline
              };
            });
            setStatus(data.message ? `Long Video: ${data.message}` : "Long Video running");
          } else if (event.event === "long_chunk") {
            const data = event.data as Partial<LongChunkProgress> & { index?: number };
            setLongProgress((current) => {
              const base = current || { preset: longPreset, warnings: [], chunks: [], partialTimeline: [] };
              return {
                ...base,
                preset: longPreset,
                chunks: mergeLongChunk(base.chunks, data)
              };
            });
          } else if (event.event === "long_partial") {
            const data = event.data as { index?: number; timeRange?: string; summary?: string };
            if (data.summary) {
              setLongProgress((current) => {
                const base = current || { preset: longPreset, warnings: [], chunks: [], partialTimeline: [] };
                return {
                  ...base,
                  preset: longPreset,
                  partialTimeline: [
                    ...base.partialTimeline.filter((item) => item.index !== data.index),
                    { index: data.index ?? Date.now(), timeRange: data.timeRange, summary: data.summary }
                  ].sort((left, right) => left.index - right.index)
                };
              });
            }
          } else if (event.event === "raw") {
            rawResult = event.data as ApiResult;
            setStreamState((current) => ({
              ...current,
              phase: "complete",
              answer: rawResult?.content || "",
              reasoning: rawResult?.reasoning || "",
              schema: rawResult?.schema || "plain_content",
              raw: rawResult
            }));
            setResult(rawResult);
          } else if (event.event === "error") {
            const data = event.data as { message?: string };
            throw new Error(data.message || "Long video analysis failed");
          }
        }

        if (done) break;
      }

      setLongProgress((current) =>
        current
          ? {
              ...current,
              phase: "complete",
              message: "Complete",
              percent: 100,
              etaSeconds: 0
            }
          : current
      );
      if (!rawResult) {
        setResult({ status: "error", message: "Long video analysis finished without a final result" });
      }
      setStatus("Complete");
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        setResult({ status: "skip", message: "Long video task stopped by user" });
        setLongProgress((current) => (current ? { ...current, phase: "stopped", message: "Stopped" } : current));
        setStatus("Stopped");
      } else {
        const message = error instanceof Error ? error.message : "Long video request failed";
        setResult({ status: "error", message });
        setStreamState((current) => ({ ...current, phase: "error", message }));
        setLongProgress((current) => (current ? { ...current, phase: "error", message } : current));
        setStatus("Request failed");
      }
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        setIsRunning(false);
        setRunMode(null);
      }
    }
  }

  async function copyRequest() {
    await navigator.clipboard.writeText(safeJson(outputTab === "json" ? jsonOutput : requestPreview));
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  const heroTags = vlaMode ? VLA_HERO_TAGS : HERO_TAGS;

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
            <kbd>⌘K</kbd>
          </button>
          <button className="iconButton" aria-label="Help">
            <HelpCircle size={18} />
          </button>
          <button className="loginButton">Login</button>
        </div>
      </header>

      <section className="hero">
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
              <span>{vlaMode ? "VLA adapter active" : "Downloadable"}</span>
            </div>
          </div>
          <p>{heroDescription(model, backendInfo)}</p>
          <div className="tagRow">
            {heroTags.map((tag, index) => (
              <span className={index > 1 ? "desktopOnlyTag" : ""} key={tag}>
                {tag}
              </span>
            ))}
            <button className="mobileMoreTags">+7</button>
          </div>
          <button className="downloadButton">
            Download Now
            <ExternalLink size={15} />
          </button>
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
        {(["Experience", "Model Card", "System Card", "Deploy"] as SectionTab[]).map((tab) => {
          const id = tab.toLowerCase().replace(/\s+/g, "-");
          return (
            <button
              className={activeTab === tab ? "active" : ""}
              key={tab}
              id={`tab-${id}`}
              role="tab"
              aria-selected={activeTab === tab}
              aria-controls={`tabpanel-${id}`}
              tabIndex={activeTab === tab ? 0 : -1}
              onClick={() => setActiveTab(tab)}
            >
              {tab}
              <span aria-hidden="true" data-preserve-spacing="true">
                {tab}
              </span>
            </button>
          );
        })}
      </div>

      <section
        className="experience"
        id={`tabpanel-${activeTab.toLowerCase().replace(/\s+/g, "-")}`}
        role="tabpanel"
        aria-labelledby={`tab-${activeTab.toLowerCase().replace(/\s+/g, "-")}`}
      >
        {activeTab === "Experience" ? (
          <ExperiencePanel
            backendInfo={backendInfo}
            copied={copied}
            copyRequest={copyRequest}
            dragActive={dragActive}
            exampleLoad={exampleLoad}
            examples={activeExamples}
            examplesOpen={examplesOpen}
            handleDrag={handleDrag}
            handleDrop={handleDrop}
            handleFile={handleFile}
            inputRef={inputRef}
            isRunning={isRunning}
            jsonOutput={jsonOutput}
            longConcurrency={longConcurrency}
            longPreset={longPreset}
            longProgress={longProgress}
            maxTokens={maxTokens}
            media={media}
            model={model}
            models={models}
            outputTab={outputTab}
            parametersOpen={parametersOpen}
            parsedOutput={parsedOutput}
            presencePenalty={presencePenalty}
            reasoningEnabled={reasoningEnabled}
            reasoningExpanded={reasoningExpanded}
            repetitionPenalty={repetitionPenalty}
            requestPreview={requestPreview}
            reset={reset}
            result={activeResult}
            run={run}
            runLongVideo={runLongVideo}
            runMode={runMode}
            seed={seed}
            selectedExampleId={selectedExampleId}
            setExamplesOpen={setExamplesOpen}
            setFramesPerSecond={setFramesPerSecond}
            setLongConcurrency={setLongConcurrency}
            setLongPreset={setLongPreset}
            setMaxTokens={setMaxTokens}
            setModel={setModel}
            setOutputTab={setOutputTab}
            setParametersOpen={setParametersOpen}
            setPresencePenalty={setPresencePenalty}
            setReasoning={setReasoning}
            setReasoningExpanded={setReasoningExpanded}
            setRepetitionPenalty={setRepetitionPenalty}
            setSeed={setSeed}
            setSelectedExampleId={setSelectedExampleId}
            setSystemPrompt={setSystemPrompt}
            setTemperature={setTemperature}
            setTopK={setTopK}
            setTopP={setTopP}
            setUserPrompt={setUserPrompt}
            status={status}
            streamPhase={streamState.phase}
            systemPrompt={systemPrompt}
            temperature={temperature}
            topK={topK}
            topP={topP}
            userPrompt={userPrompt}
            framesPerSecond={framesPerSecond}
            applyExample={applyExample}
          />
        ) : (
          <StaticTab tab={activeTab} model={model} backendInfo={backendInfo} />
        )}
      </section>
    </main>
  );
}

function VlaModeBanner({ backendInfo }: { backendInfo: BackendInfo | null }) {
  const currentMode = vlaModeLabel(backendInfo);
  const trajectoryMode = backendInfo?.vla?.full_trajectory_mode ? "Trajectory/action mode active" : "Trajectory/action mode not active";
  return (
    <section className="vlaModeBanner" aria-label="Active VLA mode">
      <div>
        <p className="vlaEyebrow">Alpamayo VLA loaded</p>
        <h2>Use BYO video as captioning or VQA over sampled frames</h2>
        <p>
          Current path is {currentMode}. Prompts are sent as questions to the Alpamayo adapter, which samples{" "}
          {vlaFrameSummary(backendInfo)} before calling the model.
        </p>
      </div>
      <dl>
        <div>
          <dt>Backbone</dt>
          <dd>{backendInfo?.vla?.backbone || "Cosmos Reason2-8B VLM"}</dd>
        </div>
        <div>
          <dt>Prompt role</dt>
          <dd>{backendInfo?.vla?.prompt_role || "VQA/caption question"}</dd>
        </div>
        <div>
          <dt>Boundary</dt>
          <dd>{trajectoryMode}</dd>
        </div>
      </dl>
    </section>
  );
}

function ExampleLoadStatus({ load }: { load: ExampleLoadState }) {
  const etaText = load.etaSeconds === 0 ? "ETA complete" : `ETA ${formatDuration(load.etaSeconds)}`;
  const bytesText = load.bytesTotal
    ? `${formatBytes(load.bytesLoaded)} / ${formatBytes(load.bytesTotal)}`
    : load.bytesLoaded
      ? formatBytes(load.bytesLoaded)
      : "";
  return (
    <div className="exampleLoadStatus">
      <div className="exampleLoadHeader">
        <strong>{load.phase}</strong>
        <span>{load.title}</span>
      </div>
      <div className="exampleLoadBar" aria-label={`Example load progress ${load.percent}%`}>
        <span style={{ width: `${Math.max(0, Math.min(100, load.percent))}%` }} />
      </div>
      <div className="exampleLoadMeta">
        <span>{load.detail || "Preparing media"}</span>
        {bytesText ? <span>{bytesText}</span> : null}
        <span>Elapsed {formatDuration(load.elapsedSeconds)}</span>
        <span>{etaText}</span>
      </div>
      <div className="exampleLoadSteps">
        {load.steps.map((step) => (
          <span className={`exampleLoadStep ${step.status}`} key={step.key}>
            {step.label}
          </span>
        ))}
      </div>
    </div>
  );
}

function ExperiencePanel({
  applyExample,
  backendInfo,
  copied,
  copyRequest,
  dragActive,
  exampleLoad,
  examples,
  examplesOpen,
  framesPerSecond,
  handleDrag,
  handleDrop,
  handleFile,
  inputRef,
  isRunning,
  jsonOutput,
  longConcurrency,
  longPreset,
  longProgress,
  maxTokens,
  media,
  model,
  models,
  outputTab,
  parametersOpen,
  parsedOutput,
  presencePenalty,
  reasoningEnabled,
  reasoningExpanded,
  repetitionPenalty,
  requestPreview,
  reset,
  result,
  run,
  runLongVideo,
  runMode,
  seed,
  selectedExampleId,
  setExamplesOpen,
  setFramesPerSecond,
  setLongConcurrency,
  setLongPreset,
  setMaxTokens,
  setModel,
  setOutputTab,
  setParametersOpen,
  setPresencePenalty,
  setReasoning,
  setReasoningExpanded,
  setRepetitionPenalty,
  setSeed,
  setSelectedExampleId,
  setSystemPrompt,
  setTemperature,
  setTopK,
  setTopP,
  setUserPrompt,
  status,
  streamPhase,
  systemPrompt,
  temperature,
  topK,
  topP,
  userPrompt
}: {
  applyExample: (exampleId?: string) => Promise<void>;
  backendInfo: BackendInfo | null;
  copied: boolean;
  copyRequest: () => Promise<void>;
  dragActive: boolean;
  exampleLoad: ExampleLoadState | null;
  examples: ExampleItem[];
  examplesOpen: boolean;
  framesPerSecond: number;
  handleDrag: (event: DragEvent<HTMLElement>, active: boolean) => void;
  handleDrop: (event: DragEvent<HTMLElement>) => Promise<void>;
  handleFile: (event: ChangeEvent<HTMLInputElement>) => Promise<void>;
  inputRef: RefObject<HTMLInputElement | null>;
  isRunning: boolean;
  jsonOutput: unknown;
  longConcurrency: number;
  longPreset: LongPreset;
  longProgress: LongProgressState | null;
  maxTokens: number;
  media: MediaState | null;
  model: string;
  models: string[];
  outputTab: OutputTab;
  parametersOpen: boolean;
  parsedOutput: { reasoning: string; answer: string; steps: string[] };
  presencePenalty: number;
  reasoningEnabled: boolean;
  reasoningExpanded: boolean;
  repetitionPenalty: number;
  requestPreview: unknown;
  reset: () => void;
  result: ApiResult | null;
  run: () => Promise<void>;
  runLongVideo: () => Promise<void>;
  runMode: RunMode;
  seed: number;
  selectedExampleId: string;
  setExamplesOpen: (open: boolean) => void;
  setFramesPerSecond: (value: number) => void;
  setLongConcurrency: (value: number) => void;
  setLongPreset: (value: LongPreset) => void;
  setMaxTokens: (value: number) => void;
  setModel: (value: string) => void;
  setOutputTab: (tab: OutputTab) => void;
  setParametersOpen: (open: boolean) => void;
  setPresencePenalty: (value: number) => void;
  setReasoning: (enabled: boolean) => void;
  setReasoningExpanded: (expanded: boolean) => void;
  setRepetitionPenalty: (value: number) => void;
  setSeed: (value: number) => void;
  setSelectedExampleId: (id: string) => void;
  setSystemPrompt: (value: string) => void;
  setTemperature: (value: number) => void;
  setTopK: (value: number) => void;
  setTopP: (value: number) => void;
  setUserPrompt: (value: string) => void;
  status: string;
  streamPhase: StreamPhase;
  systemPrompt: string;
  temperature: number;
  topK: number;
  topP: number;
  userPrompt: string;
}) {
  const [mobilePanel, setMobilePanel] = useState<MobilePanel>("input");
  const vlaMode = isVlaMode(model, backendInfo);
  const selectedExample = examples.find((example) => example.id === selectedExampleId);
  const selectedExampleLoaded =
    Boolean(selectedExample) &&
    media?.kind === selectedExample?.mediaKind &&
    media.name === selectedExample.mediaName &&
    (media.previewUrl === selectedExample.mediaUrl || Boolean(media.sourceUrl?.endsWith(selectedExample.mediaUrl)));
  const showLongVideoUi = Boolean(selectedExample?.longVideoEnabled && selectedExampleLoaded) || runMode === "long";

  function chooseLongPreset(preset: LongPreset) {
    setLongPreset(preset);
    setFramesPerSecond(LONG_VIDEO_PRESET_FPS[preset]);
    setLongConcurrency(LONG_VIDEO_DEFAULT_CONCURRENCY);
  }

  async function runWithOutputVisible() {
    setMobilePanel("output");
    await run();
  }

  async function runLongWithOutputVisible() {
    setMobilePanel("output");
    await runLongVideo();
  }

  function resetWithInputVisible() {
    reset();
    setMobilePanel("input");
  }

  return (
    <>
      <div className="aiNotice">
        <Info size={16} />
        <span className="noticeDesktop">
          AI models generate responses and outputs based on complex algorithms and machine learning techniques, and
          those responses or outputs may be inaccurate, harmful, biased or indecent. By testing this model, you assume
          the risk of any harm caused by any response or output of the model. Please do not upload any confidential
          information or personal data unless expressly permitted. Your use is logged for security purposes.
        </span>
        <span className="noticeMobile">AI Response Message</span>
        <button className="noticeView">View</button>
      </div>
      {vlaMode ? <VlaModeBanner backendInfo={backendInfo} /> : null}

      <div className="mobileIOTabs" role="tablist" aria-label="Input output">
        <button
          className={mobilePanel === "input" ? "active" : ""}
          aria-selected={mobilePanel === "input"}
          onClick={() => setMobilePanel("input")}
          role="tab"
          type="button"
        >
          Input
        </button>
        <button
          className={mobilePanel === "output" ? "active" : ""}
          aria-selected={mobilePanel === "output"}
          onClick={() => setMobilePanel("output")}
          role="tab"
          type="button"
        >
          Output
        </button>
      </div>

      <div className="workspace">
        <section className={mobilePanel === "input" ? "panel inputPanel mobileActivePanel" : "panel inputPanel"}>
          <div className="panelHeader">
            <h2>Input</h2>
            <button className="secondaryAction" onClick={() => setExamplesOpen(true)}>
              View Examples
              <ChevronDown size={15} />
            </button>
          </div>

          <label className="fieldLabel">Input</label>
          <input ref={inputRef} type="file" accept={ACCEPTED_MEDIA_ACCEPT} onChange={handleFile} hidden />
          {media ? (
            <div
              className={`dropzone hasMedia${dragActive ? " dragActive" : ""}`}
              onDragEnter={(event) => handleDrag(event, true)}
              onDragOver={(event) => handleDrag(event, true)}
              onDragLeave={(event) => handleDrag(event, false)}
              onDrop={handleDrop}
            >
              <span className="mediaLoaded">
                {media.kind === "image" ? <FileImage size={18} /> : <FileVideo size={18} />}
                {media.name}
              </span>
              <div className="mediaPreview">
                {media.kind === "image" ? (
                  <img src={media.previewUrl} alt="Selected input" />
                ) : (
                  <video
                    aria-label={`Preview of ${media.name}`}
                    controls
                    playsInline
                    preload="metadata"
                    src={media.previewUrl}
                  />
                )}
              </div>
              <button className="replaceMediaButton" onClick={() => inputRef.current?.click()} type="button">
                <Upload size={14} />
                Replace input
              </button>
            </div>
          ) : (
            <button
              className={`dropzone${dragActive ? " dragActive" : ""}`}
              onClick={() => inputRef.current?.click()}
              onDragEnter={(event) => handleDrag(event, true)}
              onDragOver={(event) => handleDrag(event, true)}
              onDragLeave={(event) => handleDrag(event, false)}
              onDrop={handleDrop}
              type="button"
            >
              <Upload size={22} />
              <span>Drop files here</span>
              <small>{ACCEPTED_MEDIA_HELP}</small>
            </button>
          )}

          <PromptBox
            label="User Prompt"
            hint={
              vlaMode
                ? "Ask a VQA/caption question about the visible scene. Keep important actions early in the clip."
                : "Describe the video or ask a question. Enable reasoning by asking for the <think> format."
            }
            max={4000}
            value={userPrompt}
            onChange={setUserPrompt}
            rows={7}
          />
          <PromptBox
            label="System Prompt"
            hint="Defines AI role/rules for session. Max 250 tokens."
            max={250}
            value={systemPrompt}
            onChange={setSystemPrompt}
            rows={4}
          />

          <ParameterAccordion
            framesPerSecond={framesPerSecond}
            maxTokens={maxTokens}
            open={parametersOpen}
            presencePenalty={presencePenalty}
            reasoningEnabled={reasoningEnabled}
            repetitionPenalty={repetitionPenalty}
            seed={seed}
            setFramesPerSecond={setFramesPerSecond}
            setMaxTokens={setMaxTokens}
            setOpen={setParametersOpen}
            setPresencePenalty={setPresencePenalty}
            setReasoning={setReasoning}
            setRepetitionPenalty={setRepetitionPenalty}
            setSeed={setSeed}
            setTemperature={setTemperature}
            setTopK={setTopK}
            setTopP={setTopP}
            temperature={temperature}
            topK={topK}
            topP={topP}
            vlaMode={vlaMode}
            vlaSummary={vlaFrameSummary(backendInfo)}
          />

          {showLongVideoUi ? (
            <div className="longVideoControls" aria-label="Long video analysis controls">
              <div className="longPresetTabs" role="radiogroup" aria-label="Long video preset">
                {(["fast", "balanced", "detailed"] as LongPreset[]).map((preset) => (
                  <button
                    aria-checked={longPreset === preset}
                    className={longPreset === preset ? "active" : ""}
                    key={preset}
                    onClick={() => chooseLongPreset(preset)}
                    role="radio"
                    type="button"
                  >
                    {preset[0].toUpperCase()}
                    {preset.slice(1)}
                  </button>
                ))}
              </div>
              <label className="longFpsControl" htmlFor="long-video-fps">
                <span>Long video FPS</span>
                <div className="longFpsInputs">
                  <input
                    id="long-video-fps"
                    max={12}
                    min={0.5}
                    onChange={(event) => setFramesPerSecond(Number(event.target.value))}
                    step={0.5}
                    type="range"
                    value={framesPerSecond}
                  />
                  <input
                    aria-label="Long video frames per second"
                    max={12}
                    min={0.5}
                    onChange={(event) => setFramesPerSecond(Number(event.target.value))}
                    step={0.5}
                    type="number"
                    value={framesPerSecond}
                  />
                </div>
              </label>
              <label className="longFpsControl" htmlFor="long-video-concurrency">
                <span>Chunk concurrency</span>
                <div className="longFpsInputs">
                  <input
                    id="long-video-concurrency"
                    max={16}
                    min={1}
                    onChange={(event) => setLongConcurrency(Math.max(1, Math.min(16, Number(event.target.value) || 1)))}
                    step={1}
                    type="range"
                    value={longConcurrency}
                  />
                  <input
                    aria-label="Long video chunk concurrency"
                    max={16}
                    min={1}
                    onChange={(event) => setLongConcurrency(Math.max(1, Math.min(16, Number(event.target.value) || 1)))}
                    step={1}
                    type="number"
                    value={longConcurrency}
                  />
                </div>
              </label>
              <p>
                Long Video sends timestamped frame chunks to the current NIM, max 5 images per request, with live ETA and
                a stitched timeline.
              </p>
              <p className="longVideoWarning">
                Higher FPS increases chunk count and wait time. Presets use 2, 4, and 6 FPS for Fast, Balanced,
                and Detailed. Chunk concurrency defaults to 16 on this RTX PRO 6000 based on the live NIM probes;
                lower it if other users share the endpoint.
              </p>
            </div>
          ) : null}

          <div className="runBar">
            <button className="resetButton" onClick={resetWithInputVisible} type="button">
              <RotateCcw size={16} />
              Reset
            </button>
            <button className={`runButton${isRunning ? " running" : ""}`} onClick={runWithOutputVisible} type="button">
              {isRunning ? null : <Play size={16} fill="currentColor" />}
              {isRunning ? "Quit Task" : "Run"}
            </button>
            {showLongVideoUi ? (
              <button
                className={`runButton longRunButton${isRunning && runMode === "long" ? " running" : ""}`}
                disabled={media?.kind !== "video" && !(isRunning && runMode === "long")}
                onClick={runLongWithOutputVisible}
                type="button"
              >
                {isRunning && runMode === "long" ? null : <FileVideo size={16} />}
                {isRunning && runMode === "long" ? "Quit Task" : "Long Video"}
              </button>
            ) : null}
          </div>
          <div className="statusLine" role="status">
            {exampleLoad ? <ExampleLoadStatus load={exampleLoad} /> : status}
          </div>
        </section>

        <section className={mobilePanel === "output" ? "panel outputPanel mobileActivePanel" : "panel outputPanel"}>
          <div className="panelHeader">
            <div className="outputTabs">
              <h2>Output</h2>
              <button
                className={outputTab === "preview" ? "previewPill activeOutputTab" : "previewPill"}
                onClick={() => setOutputTab("preview")}
                type="button"
              >
                Preview
              </button>
              <button
                className={outputTab === "json" ? "jsonTab activeOutputTab" : "jsonTab"}
                onClick={() => setOutputTab("json")}
                type="button"
              >
                JSON
              </button>
            </div>
          </div>
          <div className="outputBody">
            {outputTab === "json" ? (
              <div className="jsonOutput">
                <pre>{safeJson(jsonOutput)}</pre>
                <button onClick={copyRequest} type="button" aria-label="Copy JSON">
                  <Copy size={14} />
                  {copied ? "Copied" : "Copy"}
                </button>
              </div>
            ) : (
              <PreviewOutput
                isRunning={isRunning}
                longProgress={longProgress}
                media={media}
                parsedOutput={parsedOutput}
                reasoningExpanded={reasoningExpanded}
                result={result}
                setReasoningExpanded={setReasoningExpanded}
                streamPhase={streamPhase}
              />
            )}
          </div>
        </section>
      </div>

      <aside className="apiPanel">
        <div className="apiTopline">
          Backend: <strong>{vlaMode ? "Alpamayo VLA adapter" : backendInfo?.backend || "vLLM / OpenAI-compatible"}</strong>
        </div>
        <div className="apiButtons">
          <button>Upgrade</button>
          <button>Get API Key</button>
          <button onClick={copyRequest}>
            <Copy size={14} />
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
        <label className="fieldLabel">Model</label>
        <select value={model} onChange={(event) => setModel(event.target.value)}>
          {models.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
        <pre className="codeBlock">{safeJson(requestPreview)}</pre>
      </aside>

      {examplesOpen ? (
        <ExampleModal
          applyExample={applyExample}
          close={() => setExamplesOpen(false)}
          examples={examples}
          selectedExampleId={selectedExampleId}
          setSelectedExampleId={setSelectedExampleId}
        />
      ) : null}
    </>
  );
}

function ExampleModal({
  applyExample,
  close,
  examples,
  selectedExampleId,
  setSelectedExampleId
}: {
  applyExample: (exampleId?: string) => Promise<void>;
  close: () => void;
  examples: ExampleItem[];
  selectedExampleId: string;
  setSelectedExampleId: (id: string) => void;
}) {
  const [draftExampleId, setDraftExampleId] = useState(selectedExampleId);
  const [activeGroup, setActiveGroup] = useState<ExampleGroupId>(() =>
    exampleGroupId(examples.find((example) => example.id === selectedExampleId) || examples[0])
  );
  const [isApplying, setIsApplying] = useState(false);
  const visibleExamples = useMemo(
    () => examples.filter((example) => exampleGroupId(example) === activeGroup),
    [activeGroup, examples]
  );
  const groupCounts = useMemo(
    () =>
      EXAMPLE_GROUPS.reduce<Record<ExampleGroupId, number>>(
        (counts, group) => {
          counts[group.id] = examples.filter((example) => exampleGroupId(example) === group.id).length;
          return counts;
        },
        {
          build: 0,
          vss: 0,
          "av-dense-captioning": 0,
          "anomaly-id": 0,
          "embodied-reasoning": 0
        }
      ),
    [examples]
  );

  useEffect(() => {
    const selectedExample = examples.find((example) => example.id === selectedExampleId);
    if (selectedExample) {
      setDraftExampleId(selectedExampleId);
      setActiveGroup(exampleGroupId(selectedExample));
      return;
    }
    if (examples[0]) {
      setDraftExampleId(examples[0].id);
      setActiveGroup(exampleGroupId(examples[0]));
      setSelectedExampleId(examples[0].id);
    }
  }, [examples, selectedExampleId, setSelectedExampleId]);

  useEffect(() => {
    if (visibleExamples.length === 0 || visibleExamples.some((example) => example.id === draftExampleId)) return;
    setDraftExampleId(visibleExamples[0].id);
    setSelectedExampleId(visibleExamples[0].id);
  }, [draftExampleId, setSelectedExampleId, visibleExamples]);

  function handleDone() {
    if (!draftExampleId || !visibleExamples.some((example) => example.id === draftExampleId)) return;
    setIsApplying(true);
    void applyExample(draftExampleId);
  }

  return (
    <div className="nv-modal-overlay" data-state="open" role="presentation">
      <div className="nv-modal-content" role="dialog" aria-modal="true" aria-labelledby="example-modal-title">
        <div className="modalHeader">
          <h2 id="example-modal-title">Select an Example</h2>
          <button className="modalClose" aria-label="Close Modal" onClick={close} type="button">
            <X size={18} />
            <span>Close Modal</span>
          </button>
        </div>
        <div className="modalMain">
          <div className="exampleTabs" role="tablist" aria-label="Example categories">
            {EXAMPLE_GROUPS.map((group) => {
              const active = activeGroup === group.id;
              return (
                <button
                  aria-selected={active}
                  className={active ? "active" : ""}
                  key={group.id}
                  onClick={() => {
                    setActiveGroup(group.id);
                    const first = examples.find((example) => exampleGroupId(example) === group.id);
                    if (first) {
                      setDraftExampleId(first.id);
                      setSelectedExampleId(first.id);
                    }
                  }}
                  role="tab"
                  type="button"
                >
                  <span>{group.label}</span>
                  <small>{groupCounts[group.id]}</small>
                </button>
              );
            })}
          </div>
          <p>Select the input from the {EXAMPLE_GROUPS.find((group) => group.id === activeGroup)?.label} examples below:</p>
          {visibleExamples.length > 0 ? (
            <div className="exampleList" role="radiogroup" aria-label="Examples">
              {visibleExamples.map((example) => {
                const checked = draftExampleId === example.id;
                return (
                  <button
                    className={checked ? "exampleItem checked" : "exampleItem"}
                    key={example.id}
                    role="radio"
                    aria-checked={checked}
                    onClick={() => {
                      setDraftExampleId(example.id);
                      setSelectedExampleId(example.id);
                    }}
                    type="button"
                  >
                    <div className="exampleThumb">
                      {example.mediaKind === "image" ? (
                        <img src={example.mediaUrl} alt="" />
                      ) : (
                        <video src={example.mediaUrl} muted preload="metadata" />
                      )}
                    </div>
                    <div className="exampleText">
                      <strong>{example.title}</strong>
                      <span>
                        <b>User Prompt:</b> {promptForReasoning(example.userPrompt, example.reasoning)}
                      </span>
                      <span>
                        <b>Reasoning:</b> {example.reasoning ? "On" : "Off"}
                      </span>
                      <span>
                        <b>System Prompt:</b> {example.systemPrompt}
                      </span>
                      {example.judgeNote ? (
                        <span>
                          <b>LingoQA:</b> {example.judgeNote}
                        </span>
                      ) : null}
                    </div>
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="exampleEmpty">No examples have been added to this category yet.</div>
          )}
        </div>
        <div className="modalFooter">
          <button
            className="runButton"
            disabled={isApplying || visibleExamples.length === 0}
            onClick={handleDone}
            type="button"
            aria-busy={isApplying}
          >
            {isApplying ? "Loading" : "Done"}
          </button>
        </div>
      </div>
    </div>
  );
}

function SpatialTrajectoryOverlay({
  answerText,
  media,
  reasoningText
}: {
  answerText: string;
  media: MediaState | null;
  reasoningText: string;
}) {
  const [imageSize, setImageSize] = useState<ImageSize | null>(null);
  const [showReasoningTrace, setShowReasoningTrace] = useState(false);
  useEffect(() => {
    setShowReasoningTrace(false);
  }, [answerText, media?.previewUrl, reasoningText]);
  const canParseSpatial = useMemo(
    () => hasSpatialPayload(reasoningText) || hasSpatialPayload(answerText),
    [answerText, reasoningText]
  );
  const allMarks = useMemo(() => {
    if (!imageSize) return [];
    return [
      ...parseSpatialMarks(reasoningText, imageSize, "thinking"),
      ...parseSpatialMarks(answerText, imageSize, "final")
    ];
  }, [answerText, imageSize, reasoningText]);
  const thinkingMarks = allMarks.filter((mark) => mark.source === "thinking");
  const finalMarks = allMarks.filter((mark) => mark.source === "final");
  const marks = showReasoningTrace ? allMarks : finalMarks;
  const points = marks.filter((mark) => mark.kind === "point");
  const boxes = marks.filter((mark) => mark.kind === "bbox");
  const thinkingPoints = thinkingMarks.filter((mark) => mark.kind === "point");
  const finalPoints = finalMarks.filter((mark) => mark.kind === "point");
  const visibleThinkingPoints = showReasoningTrace ? thinkingPoints : [];
  const visibleThinkingMarks = showReasoningTrace ? thinkingMarks : [];
  const visibleFinalMarks = finalMarks;
  const traceState = showReasoningTrace ? "shown" : "hidden";

  if (!media || media.kind !== "image" || !canParseSpatial) return null;

  return (
    <article className="spatialOverlayCard">
      <div className="spatialTopline">
        <div>
          <p className="responseLabel">Spatial trajectory</p>
          <h3>Final path in image coordinate space</h3>
        </div>
        <div className="spatialToolbar">
          <span>
            {finalMarks.length} final · {thinkingMarks.length} trace {traceState} · {points.length} points · {boxes.length} boxes
          </span>
          {thinkingMarks.length > 0 ? (
            <button
              aria-checked={showReasoningTrace}
              className={showReasoningTrace ? "spatialTraceToggle checked" : "spatialTraceToggle"}
              onClick={() => setShowReasoningTrace((value) => !value)}
              role="switch"
              type="button"
            >
              <span />
              {showReasoningTrace ? "Hide reasoning trace" : "Show reasoning trace"}
            </button>
          ) : null}
        </div>
      </div>
      {thinkingMarks.length > 0 ? (
        <div className="spatialLegend" aria-label="Spatial overlay legend">
          <span className="finalLegend">Final answer</span>
          <span className={showReasoningTrace ? "traceLegend" : "traceLegend mutedLegend"}>Reasoning trace</span>
        </div>
      ) : null}
      <div className="spatialCanvas">
        <img
          className="spatialImage"
          src={media.previewUrl}
          alt={`${media.name} with generated point and box overlay`}
          onLoad={(event) => {
            const image = event.currentTarget;
            setImageSize({
              width: image.naturalWidth || image.clientWidth,
              height: image.naturalHeight || image.clientHeight
            });
          }}
        />
        {imageSize && marks.length > 0 ? (
          <svg
            aria-hidden="true"
            className="spatialSvg"
            preserveAspectRatio="xMidYMid meet"
            viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
          >
            {visibleThinkingPoints.length > 1 ? (
              <polyline
                className="spatialPath spatialPathThinking"
                points={visibleThinkingPoints.map((point) => `${point.centerX},${point.centerY}`).join(" ")}
              />
            ) : null}
            {finalPoints.length > 1 ? (
              <polyline
                className="spatialPath spatialPathFinal"
                points={finalPoints.map((point) => `${point.centerX},${point.centerY}`).join(" ")}
              />
            ) : null}
            {marks.map((mark) => (
              <g key={mark.id}>
                <title>
                  {mark.source === "thinking" ? "Reasoning trace" : "Final answer"} #{mark.sequence}: {mark.label}
                  {mark.detail ? ` — ${mark.detail}` : ""}
                </title>
                <rect
                  className={`${mark.kind === "point" ? "spatialPointBox" : "spatialBbox"} ${
                    mark.source === "thinking" ? "spatialThinkingMark" : "spatialFinalMark"
                  }`}
                  height={mark.height}
                  rx={Math.max(4, Math.min(mark.width, mark.height) * 0.08)}
                  width={mark.width}
                  x={mark.x}
                  y={mark.y}
                />
                {mark.kind === "point" ? (
                  <circle
                    className={`spatialPointDot ${mark.source === "thinking" ? "spatialThinkingDot" : "spatialFinalDot"}`}
                    cx={mark.centerX}
                    cy={mark.centerY}
                    r={5}
                  />
                ) : null}
                <text
                  className={`spatialPointLabel ${mark.source === "thinking" ? "spatialTraceLabel" : "spatialFinalLabel"}`}
                  x={mark.x + 8}
                  y={Math.max(18, mark.y - 8)}
                >
                  {spatialMarkSvgPrefix(mark)}
                  {mark.sequence}
                </text>
              </g>
            ))}
          </svg>
        ) : null}
      </div>
      {marks.length > 0 ? (
        <div className="spatialSequenceGroups">
          <SpatialMarkList label="Final answer" marks={visibleFinalMarks} />
          {showReasoningTrace && visibleThinkingMarks.length > 0 ? (
            <SpatialMarkList label="Reasoning trace" marks={visibleThinkingMarks} />
          ) : null}
        </div>
      ) : thinkingMarks.length > 0 && !showReasoningTrace ? (
        <p className="spatialLoading">Reasoning trace is hidden. Toggle it on to inspect trace coordinates.</p>
      ) : (
        <p className="spatialLoading">Loading image dimensions for overlay alignment...</p>
      )}
    </article>
  );
}

function spatialMarkPrefix(mark: SpatialMark) {
  return mark.source === "thinking" ? "Trace" : "Final";
}

function spatialMarkSvgPrefix(mark: SpatialMark) {
  return mark.source === "thinking" ? "R" : "F";
}

function SpatialMarkList({ label, marks }: { label: string; marks: SpatialMark[] }) {
  if (marks.length === 0) return null;
  return (
    <section className={label === "Reasoning trace" ? "spatialSequenceGroup traceGroup" : "spatialSequenceGroup"}>
      <p>{label}</p>
      <ol className="spatialSequence" aria-label={`${label} coordinate sequence`}>
        {marks.map((mark) => (
          <li className="spatialSequenceItem" key={`sequence-${mark.id}`}>
            <strong>
              {spatialMarkPrefix(mark)} #{mark.sequence}
            </strong>
            <span>{mark.label}</span>
            {mark.detail ? <em>{mark.detail}</em> : null}
            <code>
              {mark.sourceKey} {mark.coordinateMode === "cosmos-1000" ? "0-1000" : mark.coordinateMode} →{" "}
              {mark.kind === "bbox"
                ? `${Math.round(mark.x)}, ${Math.round(mark.y)}, ${Math.round(mark.width)}×${Math.round(mark.height)}`
                : `${Math.round(mark.centerX)}, ${Math.round(mark.centerY)}`}
            </code>
          </li>
        ))}
      </ol>
    </section>
  );
}

function PreviewOutput({
  isRunning,
  longProgress,
  media,
  parsedOutput,
  reasoningExpanded,
  result,
  setReasoningExpanded,
  streamPhase
}: {
  isRunning: boolean;
  longProgress: LongProgressState | null;
  media: MediaState | null;
  parsedOutput: { reasoning: string; answer: string; steps: string[] };
  reasoningExpanded: boolean;
  result: ApiResult | null;
  setReasoningExpanded: (expanded: boolean) => void;
  streamPhase: StreamPhase;
}) {
  const progressCard = longProgress ? <LongVideoProgress progress={longProgress} /> : null;

  if (result?.status === "error" || result?.error) {
    return (
      <div className="responseStack">
        {progressCard}
        <pre className="errorBox">{result.message || result.error}</pre>
      </div>
    );
  }

  if (result?.status === "skip") {
    return (
      <div className="responseStack">
        {progressCard}
        <article className="emptyOutput">
          <h3>Task stopped</h3>
          <p>{result.message || "The current inference task was stopped before completion."}</p>
        </article>
      </div>
    );
  }

  if (result?.files && result.files.length > 0) {
    return (
      <div className="resultFiles">
        {result.files.map((file) => {
          const src = file.b64 ? `data:${file.mime};base64,${file.b64}` : null;
          if (src && file.mime.startsWith("video/")) {
            return <video key={file.path} controls src={src} style={{ width: "100%" }} />;
          }
          if (src && file.mime.startsWith("image/")) {
            return <img key={file.path} src={src} alt={file.path} style={{ width: "100%" }} />;
          }
          return (
            <pre key={file.path} className="filePath">
              {file.path}
            </pre>
          );
        })}
      </div>
    );
  }

  if (result?.content || result?.reasoning) {
    const hasAnswer = Boolean(parsedOutput.answer || result.content);
    const answerText = parsedOutput.answer || result.content || "";
    const timelineItems = stitchedTimelineItems(result);
    const isLongVideoResult = Boolean(result?.long_video || longProgress);
    const collapseResponse = isLongVideoResult && hasAnswer;
    return (
      <div className="responseStack">
        {progressCard}
        {timelineItems.length > 0 ? <StitchedTimeline items={timelineItems} /> : null}
        <SpatialTrajectoryOverlay answerText={answerText} media={media} reasoningText={parsedOutput.reasoning} />
        {hasAnswer && collapseResponse ? (
          <details className="answer rawResponseDetails">
            <summary>{looksLikeJsonResponse(answerText) ? "Stitched JSON response" : "Raw stitched response"}</summary>
            <pre>{answerText || "No final response returned."}</pre>
          </details>
        ) : hasAnswer ? (
          <article className={isRunning ? "answer streamingAnswer" : "answer"}>
            <p className="responseLabel">Response</p>
            <FormattedText text={answerText || "No final response returned."} />
          </article>
        ) : null}
        {parsedOutput.reasoning ? (
          <ReasoningCard
            complete={!isRunning || streamPhase === "answer" || streamPhase === "complete"}
            expanded={reasoningExpanded}
            isStreaming={isRunning && streamPhase === "reasoning"}
            reasoning={parsedOutput.reasoning}
            steps={parsedOutput.steps}
            setExpanded={setReasoningExpanded}
          />
        ) : null}
      </div>
    );
  }

  if (isRunning) {
    return (
      <div className="responseStack">
        {progressCard}
        {progressCard ? null : <GeneratingOutput />}
      </div>
    );
  }

  return (
    <article className="emptyOutput">
      <h3>Ready for inference</h3>
      <p>Upload media or choose an example, then run the model to see the response and reasoning trace.</p>
    </article>
  );
}

function TimelineList({ empty, items }: { empty: string; items: TimelineItem[] }) {
  if (items.length === 0) return <p className="timelineEmpty">{empty}</p>;
  return (
    <ol className="longTimelineList">
      {items.map((item, index) => (
        <li key={item.id || index}>
          <strong>{item.range}</strong>
          <span>
            {item.title ? <em>{item.title}</em> : null}
            {item.caption}
          </span>
          {item.meta ? <small>{item.meta}</small> : null}
        </li>
      ))}
    </ol>
  );
}

function StitchedTimeline({ items }: { items: TimelineItem[] }) {
  return (
    <article className="stitchedTimelineCard">
      <div className="stitchedTimelineHeader">
        <div>
          <p className="responseLabel">Complete Stitched Response</p>
          <h3>Sequential timeline</h3>
        </div>
        <span>{items.length} events</span>
      </div>
      <TimelineList empty="No stitched events were returned." items={items} />
    </article>
  );
}

function LongVideoProgress({ progress }: { progress: LongProgressState }) {
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [showTimelineEvents, setShowTimelineEvents] = useState(false);
  const [timelineCollapsed, setTimelineCollapsed] = useState(true);
  const [compact, setCompact] = useState(false);
  const percent = Math.max(0, Math.min(100, Math.round(progress.percent || 0)));
  const completed = progress.completedChunks || progress.chunks.filter((chunk) => chunk.status === "done").length;
  const failed = progress.failedChunks || progress.chunks.filter((chunk) => chunk.status === "error").length;
  const running = progress.runningChunks || progress.chunks.filter((chunk) => chunk.status === "running").length;
  const total = progress.totalChunks || progress.chunks.length;
  const phase = progress.phase || "preparing";
  const timelineItems = showTimelineEvents
    ? timelineItemsFromChunks(progress.chunks, "events").slice(-24)
    : progress.partialTimeline.slice(-8).map((item) => ({
        id: `partial-${item.index}-${item.timeRange}`,
        range: item.timeRange || `Chunk ${item.index + 1}`,
        title: "",
        caption: item.summary,
        meta: `chunk ${item.index + 1}`
      }));
  const selectedChunk =
    progress.chunks.find((chunk) => chunk.index === selectedIndex) ||
    (selectedIndex === null ? null : progress.chunks[selectedIndex]) ||
    null;
  useEffect(() => {
    if (selectedIndex !== null && !progress.chunks.some((chunk) => chunk.index === selectedIndex)) {
      setSelectedIndex(null);
    }
  }, [progress.chunks, selectedIndex]);
  return (
    <article className={`longProgressCard${compact ? " compact" : ""}`} aria-live="polite">
      <div className="longProgressHeader">
        <div>
          <p className="responseLabel">Long Video Analysis</p>
          <h3>{progress.message || "Preparing timeline analysis"}</h3>
        </div>
        <div className="longProgressActions">
          <span className="longPresetBadge">{progress.preset}</span>
          <button className="longProgressCollapse" onClick={() => setCompact((value) => !value)} type="button">
            {compact ? "Show details" : "Collapse"}
          </button>
        </div>
      </div>
      <div className="longProgressMeta">
        <span>Phase: {phase.replace(/_/g, " ")}</span>
        <span>Chunks: {completed + failed}/{total || "?"}</span>
        <span>Running: {running}</span>
        <span>Elapsed: {formatDuration(progress.elapsedSeconds)}</span>
        <span>ETA: {progress.etaSeconds === 0 ? "complete" : formatDuration(progress.etaSeconds)}</span>
      </div>
      <div className="longProgressTrack" aria-label={`Long video progress ${percent}%`}>
        <span style={{ width: `${percent}%` }} />
      </div>
      {progress.steps?.length ? <LongStepList steps={progress.steps} /> : null}
      <div className="longCoverageGrid">
        <span>Duration: {progress.durationText || formatDuration(progress.durationSeconds)}</span>
        <span>Frames: {progress.frameCount ?? "scanning"}</span>
        <span>Coverage: {progress.sampleFps ? `${progress.sampleFps} fps` : "planning"}</span>
        <span>Requested FPS: {progress.requestedFps ? progress.requestedFps : "preset"}</span>
        {progress.frameLimit ? <span>Frame budget: {progress.frameLimit}</span> : null}
        <span>Concurrency: {progress.concurrency || 4}</span>
      </div>
      {!compact ? (
        <>
          {progress.warnings.length > 0 ? (
            <ul className="longWarnings">
              {progress.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          ) : null}
          {progress.chunks.length > 0 ? (
            <div className="longChunkGrid" aria-label="Chunk progress">
              {progress.chunks.map((chunk) => (
                <button
                  aria-pressed={selectedChunk?.index === chunk.index}
                  className={`longChunkPill ${chunk.status}${selectedChunk?.index === chunk.index ? " selected" : ""}`}
                  key={chunk.index}
                  onClick={() => setSelectedIndex((current) => (current === chunk.index ? null : chunk.index))}
                  title={chunk.error || chunk.summary || chunk.timeRange}
                  type="button"
                >
                  <div className="longChunkThumb" aria-hidden="true">
                    {chunk.thumbnailUrl ? <img src={chunk.thumbnailUrl} alt="" loading="lazy" /> : <span />}
                    <div className="longChunkOverlay">
                      <strong>{chunk.index + 1}</strong>
                      <span>{chunk.status}</span>
                    </div>
                  </div>
                  <small>{chunk.timeRange}</small>
                  <span className="longChunkProgress" aria-label={`Chunk ${chunk.index + 1} ${chunk.status}`}>
                    <span style={{ width: `${chunkProgressPercent(chunk.status)}%` }} />
                  </span>
                </button>
              ))}
            </div>
          ) : null}
          {selectedChunk ? <LongChunkInspector chunk={selectedChunk} /> : null}
          {progress.partialTimeline.length > 0 || progress.chunks.some((chunk) => eventsFromChunk(chunk).length > 0) ? (
            <div className={`longTimeline${timelineCollapsed ? " collapsed" : ""}`}>
              <div className="longTimelineHeader">
                <button
                  aria-expanded={!timelineCollapsed}
                  className="longTimelineTitle"
                  onClick={() => setTimelineCollapsed((value) => !value)}
                  type="button"
                >
                  <span>{showTimelineEvents ? "Live parsed events" : "Live partial timeline"}</span>
                  <small>{timelineCollapsed ? "Show timeline" : "Hide timeline"}</small>
                </button>
                {!timelineCollapsed ? (
                  <button
                    className="longTimelineMode"
                    onClick={() => setShowTimelineEvents((value) => !value)}
                    type="button"
                  >
                    {showTimelineEvents ? "Show partial timeline" : "Show events"}
                  </button>
                ) : null}
              </div>
              {!timelineCollapsed ? (
                <TimelineList
                  empty={showTimelineEvents ? "No parsed events have arrived yet." : "No chunk summaries have arrived yet."}
                  items={timelineItems}
                />
              ) : null}
            </div>
          ) : null}
        </>
      ) : null}
    </article>
  );
}

function LongStepList({ steps }: { steps: LongStepProgress[] }) {
  return (
    <div className="longStepList" aria-label="Long video analysis steps">
      {steps.map((step) => {
        const progress = Math.max(0, Math.min(100, Math.round(Number(step.progress) || 0)));
        return (
          <div className={`longStepItem ${step.status}`} key={step.key}>
            <div className="longStepHeader">
              <span>{step.label}</span>
              <strong>{step.status}</strong>
            </div>
            <div className="longStepTrack" aria-label={`${step.label} ${progress}%`}>
              <span style={{ width: `${progress}%` }} />
            </div>
            {step.detail ? <small>{step.detail}</small> : null}
          </div>
        );
      })}
    </div>
  );
}

function exampleGroupId(example?: ExampleItem): ExampleGroupId {
  return example?.group || "build";
}

function LongChunkInspector({ chunk }: { chunk: LongChunkProgress }) {
  const hasEvents = Array.isArray(chunk.events) && chunk.events.length > 0;
  const eventItems = timelineItemsFromEvents(eventsFromChunk(chunk), chunk.timeRange, chunk.summary);
  const rawText = chunk.content || chunk.summary || chunk.error || "No chunk response has arrived yet.";
  return (
    <section className="longChunkInspector" aria-label={`Chunk ${chunk.index + 1} details`}>
      <div className="longChunkInspectorHeader">
        <div>
          <p className="responseLabel">Selected Chunk</p>
          <h4>
            Chunk {chunk.index + 1}
            {chunk.timeRange ? ` · ${chunk.timeRange}` : ""}
          </h4>
        </div>
        <span className={`longChunkStatus ${chunk.status}`}>{chunk.status}</span>
      </div>
      <div className="longChunkInspectorMeta">
        <span>Frames: {chunk.frameCount ?? "done"}</span>
        <span>Elapsed: {formatDuration(chunk.elapsedSeconds)}</span>
        <span>Events: {chunk.eventsCount ?? chunk.events?.length ?? 0}</span>
      </div>
      {chunk.summary ? (
        <div className="longChunkSummary">
          <strong>Summary</strong>
          <p>{chunk.summary}</p>
        </div>
      ) : null}
      {hasEvents || eventItems.length > 0 ? (
        <div className="longChunkEvents">
          <strong>Parsed events</strong>
          <TimelineList empty="No parsed events for this chunk." items={eventItems} />
        </div>
      ) : null}
      <details className="longChunkRaw">
        <summary>Raw chunk response</summary>
        <pre>{rawText}</pre>
      </details>
    </section>
  );
}

function GeneratingOutput() {
  return (
    <article className="generatingOutput" aria-live="polite">
      <div className="generatingCenter">
        <GeneratingMark />
        <h3>Generating...</h3>
      </div>
    </article>
  );
}

function GeneratingMark() {
  return (
    <div className="generatingMark" aria-hidden="true">
      <span className="facet facetA" />
      <span className="facet facetB" />
      <span className="facet facetC" />
      <span className="facet facetD" />
      <span className="facet facetE" />
      <span className="facet facetF" />
      <span className="facet facetG" />
      <span className="facet facetH" />
      <span className="facet facetI" />
    </div>
  );
}

function ReasoningStatusIcon({ complete }: { complete: boolean }) {
  if (complete) return <CheckCircle2 size={16} />;
  return <span className="reasoningSpinner" aria-hidden="true" />;
}

function ReasoningCard({
  complete,
  expanded,
  isStreaming,
  reasoning,
  setExpanded,
  steps
}: {
  complete: boolean;
  expanded: boolean;
  isStreaming: boolean;
  reasoning: string;
  setExpanded: (expanded: boolean) => void;
  steps: string[];
}) {
  const [openSteps, setOpenSteps] = useState<Record<string, boolean>>({});
  const title = complete ? "Reasoning Complete" : "Thinking...";
  const description = complete
    ? "Below is the entire thinking process the model went through to arrive at its response."
    : "The Model is thinking. Once it has completed reasoning, it will give you a response.";

  if (!expanded) {
    return (
      <button
        className={complete ? "reasoningCollapsed" : "reasoningCollapsed thinkingCollapsed"}
        onClick={() => setExpanded(true)}
        type="button"
      >
        <ReasoningStatusIcon complete={complete} />
        <span>{title}</span>
        <ChevronDown className="pillChevron" size={15} />
      </button>
    );
  }

  const visibleSteps = steps.length > 0 ? steps : [reasoning];
  return (
    <div className="reasoningFrame">
      <article className={complete ? "reasoningCard" : "reasoningCard thinkingCard"}>
        <div className="reasoningTopline">
          <div>
            <h3>{title}</h3>
            <p>{description}</p>
          </div>
          <button onClick={() => setExpanded(false)} type="button">
            Collapse
            <ChevronDown className="collapseChevron" size={15} />
          </button>
        </div>
        <ul>
          {visibleSteps.map((step, index) => {
            const id = `reasoning-step-${index}`;
            const active = isStreaming && index === visibleSteps.length - 1;
            const open = openSteps[id] ?? active;
            return (
              <li className={open ? "open" : ""} key={id}>
                <button
                  aria-expanded={open}
                  className={active ? "reasoningStepButton activeStep" : "reasoningStepButton"}
                  onClick={() => setOpenSteps((current) => ({ ...current, [id]: !open }))}
                  type="button"
                >
                  <span className="stepStatus" aria-hidden="true">
                    {active ? <ReasoningStatusIcon complete={false} /> : <CheckCircle2 size={15} />}
                  </span>
                  <span className="stepText">{step}</span>
                  <ChevronRight className="stepChevron" size={14} />
                </button>
              </li>
            );
          })}
        </ul>
      </article>
      <button
        className={complete ? "reasoningPill" : "reasoningPill thinkingPill"}
        onClick={() => setExpanded(false)}
        type="button"
      >
        <ReasoningStatusIcon complete={complete} />
        <span>{title}</span>
        <ChevronDown className="pillChevron expanded" size={15} />
      </button>
    </div>
  );
}

function FormattedText({ text }: { text: string }) {
  const blocks = text.split(/\n{2,}/).map((block) => block.trim()).filter(Boolean);
  if (blocks.length === 0) return null;
  return (
    <>
      {blocks.map((block, index) => {
        if (/^-{3,}$/.test(block)) return <hr key={index} />;
        if (block.startsWith("###")) return <h4 key={index}>{block.replace(/^#+\s*/, "")}</h4>;
        return <p key={index}>{block}</p>;
      })}
    </>
  );
}

function ParameterAccordion({
  framesPerSecond,
  maxTokens,
  open,
  presencePenalty,
  reasoningEnabled,
  repetitionPenalty,
  seed,
  setFramesPerSecond,
  setMaxTokens,
  setOpen,
  setPresencePenalty,
  setReasoning,
  setRepetitionPenalty,
  setSeed,
  setTemperature,
  setTopK,
  setTopP,
  temperature,
  topK,
  topP,
  vlaMode,
  vlaSummary
}: {
  framesPerSecond: number;
  maxTokens: number;
  open: boolean;
  presencePenalty: number;
  reasoningEnabled: boolean;
  repetitionPenalty: number;
  seed: number;
  setFramesPerSecond: (value: number) => void;
  setMaxTokens: (value: number) => void;
  setOpen: (open: boolean) => void;
  setPresencePenalty: (value: number) => void;
  setReasoning: (enabled: boolean) => void;
  setRepetitionPenalty: (value: number) => void;
  setSeed: (value: number) => void;
  setTemperature: (value: number) => void;
  setTopK: (value: number) => void;
  setTopP: (value: number) => void;
  temperature: number;
  topK: number;
  topP: number;
  vlaMode: boolean;
  vlaSummary: string;
}) {
  const frameHelp = vlaMode
    ? `Alpamayo receives the MP4 as video_url and samples ${vlaSummary} inside the adapter. This slider only affects legacy image-frame fallback paths.`
    : PARAMETER_HELP.framesPerSecond;

  return (
    <div className="parameterAccordion">
      <button
        className="nv-accordion-trigger"
        type="button"
        aria-expanded={open}
        data-state={open ? "open" : "closed"}
        onClick={() => setOpen(!open)}
      >
        <span className="nv-accordion-label-text">View Parameters</span>
        <ChevronDown className="nv-accordion-icon" size={16} />
      </button>
      {open ? (
        <div className="nv-accordion-content" data-state="open">
          {vlaMode ? (
            <div className="vlaParameterNote">
              <strong>VLA adapter sampling</strong>
              <span>
                For Alpamayo, use short front-loaded clips. Change adapter env vars, not this FPS slider, to alter the
                effective frame window.
              </span>
            </div>
          ) : null}
          <SliderField
            help={PARAMETER_HELP.temperature}
            label="Temperature"
            min={0}
            max={1}
            step={0.05}
            value={temperature}
            onChange={setTemperature}
          />
          <SliderField
            help={PARAMETER_HELP.topP}
            label="Top P"
            min={0.01}
            max={1}
            step={0.01}
            value={topP}
            onChange={setTopP}
          />
          <SliderField
            help={PARAMETER_HELP.topK}
            label="Top K"
            min={1}
            max={100}
            step={1}
            value={topK}
            onChange={setTopK}
          />
          <SliderField
            help={PARAMETER_HELP.repetitionPenalty}
            label="Repetition Penalty"
            min={1}
            max={2}
            step={0.05}
            value={repetitionPenalty}
            onChange={setRepetitionPenalty}
          />
          <SliderField
            help={PARAMETER_HELP.presencePenalty}
            label="Presence Penalty"
            min={0}
            max={2}
            step={0.05}
            value={presencePenalty}
            onChange={setPresencePenalty}
          />
          <SliderField
            help={frameHelp}
            label={vlaMode ? "Frame fallback FPS" : "Frames per Second"}
            min={0.5}
            max={12}
            step={0.5}
            value={framesPerSecond}
            onChange={setFramesPerSecond}
          />
          <SliderField
            help={PARAMETER_HELP.maxTokens}
            label="Max Tokens"
            min={128}
            max={4096}
            step={128}
            value={maxTokens}
            onChange={setMaxTokens}
          />
          <label className="seedField">
            <span>
              Seed
              <InfoTooltip help={PARAMETER_HELP.seed} label="Seed" />
            </span>
            <input type="number" value={seed} disabled onChange={(event) => setSeed(Number(event.target.value))} />
          </label>
          <div className="reasoningSwitch">
            <button
              aria-checked={reasoningEnabled}
              className={reasoningEnabled ? "switchTrack checked" : "switchTrack"}
              onClick={() => setReasoning(!reasoningEnabled)}
              role="switch"
              type="button"
            >
              <span />
            </button>
            <span>Reasoning</span>
            <InfoTooltip help={PARAMETER_HELP.reasoning} label="Reasoning" />
          </div>
        </div>
      ) : null}
    </div>
  );
}

function SliderField({
  help,
  label,
  max,
  min,
  onChange,
  step,
  value
}: {
  help: string;
  label: string;
  max: number;
  min: number;
  onChange: (value: number) => void;
  step: number;
  value: number;
}) {
  return (
    <fieldset className="sliderField">
      <legend>
        <span>{label}</span>
        <InfoTooltip help={help} label={label} />
      </legend>
      <div className="sliderRow">
        <input
          aria-label={label}
          max={max}
          min={min}
          onChange={(event) => onChange(Number(event.target.value))}
          step={step}
          type="range"
          value={value}
        />
        <input
          aria-label={`${label} value`}
          max={max}
          min={min}
          onChange={(event) => onChange(Number(event.target.value))}
          step={step}
          type="number"
          value={value}
        />
      </div>
    </fieldset>
  );
}

function InfoTooltip({ help, label }: { help: string; label: string }) {
  return (
    <span className="infoTooltip" tabIndex={0} aria-label={`${label}: ${help}`}>
      <Info size={15} />
      <span className="tooltipBubble" role="tooltip">
        {help}
      </span>
    </span>
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
  const maxModelLen = backendInfo?.vllm?.model?.max_model_len || flagValue(backendInfo, "max_model_len");
  const parser = flagValue(backendInfo, "reasoning_parser");
  const gpuUtil = flagValue(backendInfo, "gpu_memory_utilization");
  const source = backendInfo?.source;
  const vlaMode = isVlaMode(model, backendInfo);

  return (
    <div className="runtimeBar" id="active-model-runtime-details">
      <div className="runtimeMetric">
        <span className="runtimeLabel">Live Model</span>
        <span className="runtimeValue">{model}</span>
      </div>
      <div className="runtimeMetric">
        <span className="runtimeLabel">Backend</span>
        <span className="runtimeValue">
          {displayValue(backendInfo?.backend || "vLLM")} at {displayValue(backendInfo?.vllm?.base_url || backendInfo?.base_url)}
        </span>
      </div>
      <div className="runtimeMetric">
        <span className="runtimeLabel">{vlaMode ? "Mode" : "Quantization"}</span>
        <span className="runtimeValue">{vlaMode ? vlaModeLabel(backendInfo) : quantizationLabel(backendInfo)}</span>
      </div>
      <div className="runtimeMetric">
        <span className="runtimeLabel">Model Commit</span>
        <span className="runtimeValue">
          {source?.commit_url ? (
            <a className="runtimeCommitLink" href={source.commit_url} rel="noreferrer" target="_blank">
              <code>{shortSha(source?.sha)}</code>
            </a>
          ) : (
            <code>{shortSha(source?.sha)}</code>
          )}
          <span className="runtimeSubtle"> {formatTimestamp(source?.timestamp)}</span>
        </span>
      </div>
      <div className="runtimeMetric">
        <span className="runtimeLabel">{vlaMode ? "Adapter Sampling" : "vLLM Details"}</span>
        <span className="runtimeValue">
          {vlaMode ? (
            vlaFrameSummary(backendInfo)
          ) : (
            <>
              max len {displayValue(maxModelLen)}
              {parser ? `, parser ${parser}` : ""}
              {gpuUtil ? `, GPU util ${gpuUtil}` : ""}
            </>
          )}
        </span>
      </div>
      <a className="runtimeJsonLink" href={detailsUrl} rel="noreferrer" target="_blank">
        Full JSON details
        <ExternalLink size={13} />
      </a>
    </div>
  );
}

function RuntimeDetails({ backendInfo, model }: { backendInfo: BackendInfo | null; model: string }) {
  const vlaMode = isVlaMode(model, backendInfo);
  return (
    <dl className="metadataGrid">
      <dt>Loaded model</dt>
      <dd>{model}</dd>
      <dt>Backend</dt>
      <dd>{backendInfo?.backend || "vLLM / OpenAI-compatible"}</dd>
      <dt>Backend endpoint</dt>
      <dd>{backendInfo?.vllm?.base_url || backendInfo?.base_url || "http://localhost:8000/v1"}</dd>
      {vlaMode ? (
        <>
          <dt>VLA family</dt>
          <dd>{backendInfo?.vla?.family || "Alpamayo"}</dd>
          <dt>Current VLA mode</dt>
          <dd>{vlaModeLabel(backendInfo)}</dd>
          <dt>VLA backbone</dt>
          <dd>{backendInfo?.vla?.backbone || "Cosmos Reason2-8B VLM"}</dd>
          <dt>Adapter sampling</dt>
          <dd>{vlaFrameSummary(backendInfo)}</dd>
          <dt>Trajectory/action mode</dt>
          <dd>{backendInfo?.vla?.full_trajectory_mode ? "active" : "not active"}</dd>
        </>
      ) : null}
      <dt>HF model commit SHA</dt>
      <dd>
        <code>{displayValue(backendInfo?.source?.sha)}</code>
      </dd>
      <dt>HF commit time</dt>
      <dd>{formatTimestamp(backendInfo?.source?.timestamp)}</dd>
      <dt>HF repo</dt>
      <dd>{displayValue(backendInfo?.source?.repo_id)}</dd>
      <dt>HF revision</dt>
      <dd>{displayValue(backendInfo?.source?.revision || backendInfo?.source?.branch)}</dd>
      <dt>HF commit URL</dt>
      <dd>
        {backendInfo?.source?.commit_url ? (
          <a href={backendInfo.source.commit_url} rel="noreferrer" target="_blank">
            {backendInfo.source.commit_url}
          </a>
        ) : (
          "unknown"
        )}
      </dd>
      <dt>HF cache path</dt>
      <dd>
        <code>{displayValue(backendInfo?.source?.cache_path)}</code>
      </dd>
      <dt>Source detection</dt>
      <dd>{displayValue(backendInfo?.source?.source)}</dd>
      <dt>Quantization</dt>
      <dd>{quantizationLabel(backendInfo)}</dd>
      <dt>Quantization source</dt>
      <dd>{displayValue(backendInfo?.quantization?.source)}</dd>
      <dt>vLLM model id</dt>
      <dd>{displayValue(backendInfo?.vllm?.model?.id)}</dd>
      <dt>vLLM owner</dt>
      <dd>{displayValue(backendInfo?.vllm?.model?.owned_by)}</dd>
      <dt>vLLM max model len</dt>
      <dd>{displayValue(backendInfo?.vllm?.model?.max_model_len || flagValue(backendInfo, "max_model_len"))}</dd>
      <dt>vLLM process</dt>
      <dd>{backendInfo?.vllm?.process?.pid ? `pid ${backendInfo.vllm.process.pid}` : "unknown"}</dd>
      <dt>Reasoning parser</dt>
      <dd>{displayValue(flagValue(backendInfo, "reasoning_parser"))}</dd>
      <dt>Media IO kwargs</dt>
      <dd>
        <code>{displayValue(flagValue(backendInfo, "media_io_kwargs"))}</code>
      </dd>
      <dt>vLLM dtype</dt>
      <dd>{displayValue(flagValue(backendInfo, "dtype"))}</dd>
      <dt>GPU memory utilization</dt>
      <dd>{displayValue(flagValue(backendInfo, "gpu_memory_utilization"))}</dd>
      <dt>Served model name</dt>
      <dd>{displayValue(flagValue(backendInfo, "served_model_name"))}</dd>
      <dt>Allowed media path</dt>
      <dd>
        <code>{displayValue(flagValue(backendInfo, "allowed_local_media_path"))}</code>
      </dd>
    </dl>
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
  const vlaMode = isVlaMode(model, backendInfo);
  const familyLabel = modelFamilyLabel(model);
  const imageName = nimImageForModel(model, backendInfo);
  if (tab === "Model Card") {
    return (
      <div className="staticPanel">
        <p className="staticEyebrow">Overview</p>
        <h2>{model}</h2>
        <p className="staticLead">
          {vlaMode
            ? "Alpamayo is loaded as a VLA-backed BYO-video endpoint. The current frontend path is captioning and VQA over adapter-sampled frames, with full driving trajectory/action mode called out as not active."
            : `${familyLabel} is a vision-language reasoning surface for images and videos. This Vite deployment is tuned for physical-world understanding tasks that benefit from structured reasoning, visible trace playback, and concise final answers.`}
        </p>

        <StaticSection title="ModelCard++">
          <p>
            {vlaMode
              ? "This card follows the NVIDIA Build model-card layout while reflecting the active Alpamayo adapter. It separates the user-facing VQA/caption mode from the deeper VLA trajectory mode that still needs a dedicated adapter path."
              : `This card follows the NVIDIA Build model-card layout while reflecting the active ${familyLabel} deployment shown here. It summarizes intended inputs, outputs, integration notes, and operational risks for evaluating the local OpenAI-compatible endpoint.`}
          </p>
        </StaticSection>

        <StaticSection title="Description">
          <p>
            The UI sends multimodal chat messages to a local reasoner service and renders streamed reasoning when the
            backend returns it. The experience is intended for robotics, industrial inspection, smart-city review,
            annotation, and video-understanding workflows where spatial and temporal cues matter.
          </p>
        </StaticSection>

        <StaticSection title="Input">
          <dl>
            <dt>Type</dt>
            <dd>Text with video or image</dd>
            <dt>Formats</dt>
            <dd>{ACCEPTED_MEDIA_HELP}</dd>
            <dt>Prompting</dt>
            <dd>
              {vlaMode ? (
                <>
                  Prompts are VQA/caption questions over sampled frames. Use short, front-loaded clips unless adapter
                  sampling settings are raised.
                </>
              ) : (
                <>Reasoning prompts can request a <code>&lt;think&gt;</code> trace followed by the answer.</>
              )}
            </dd>
          </dl>
        </StaticSection>

        <StaticSection title="Output">
          <dl>
            <dt>Type</dt>
            <dd>Text</dd>
            <dt>Reasoning trace</dt>
            <dd>Displayed when returned as inline <code>&lt;think&gt;</code> content or streamed reasoning deltas.</dd>
            <dt>Recommended review</dt>
            <dd>Validate conclusions against the source media before using outputs in production workflows.</dd>
          </dl>
        </StaticSection>

        <StaticSection title="Software Integration">
          <RuntimeDetails backendInfo={backendInfo} model={model} />
          <dl>
            <dt>Model type</dt>
            <dd>VLM / Reasoner</dd>
            <dt>Primary backend</dt>
            <dd>{backendInfo?.backend || "vLLM / OpenAI-compatible"}</dd>
            <dt>Endpoint</dt>
            <dd>{backendInfo?.base_url || "http://localhost:8000/v1"}</dd>
          </dl>
        </StaticSection>

        <StaticSection title="Ethical Considerations">
          <p>
            Users are responsible for evaluating whether inputs, outputs, and downstream decisions are appropriate for
            their domain. Add domain-specific guardrails, validation, logging policies, and human review before deploying
            decisions that affect people, property, or safety.
          </p>
        </StaticSection>

        <a className="staticLink" href={BUILD_REASON2_MODEL_CARD_URL} rel="noreferrer" target="_blank">
          NVIDIA Build model-card reference <ExternalLink size={16} />
        </a>
      </div>
    );
  }
  if (tab === "System Card") {
    return (
      <div className="staticPanel">
        <p className="staticEyebrow">System Card</p>
        <h2>NVIDIA Cosmos</h2>
        <p className="staticLead">
          Cosmos is NVIDIA's platform for physical AI world-model development. This placeholder adapts the Cosmos
          system-card topics for a reasoning UI that consumes images, videos, and text prompts through a local service.
        </p>

        <StaticSection title="Cosmos Model Family">
          <p>
            Cosmos models support physical AI workflows such as robot training, curation, simulation, and multimodal
            reasoning. This app focuses on the reasoner path: it analyzes supplied media and returns natural-language
            conclusions, optionally with a streamed reasoning trace.
          </p>
        </StaticSection>

        <StaticSection title="Governing Terms / Terms of Use">
          <p>
            Use is subject to the license and terms attached to the model and deployment environment. Confirm commercial
            rights, derivative-model requirements, and any guardrail obligations before production use.
          </p>
        </StaticSection>

        <StaticSection title="Specific Risk Areas and Mitigations">
          <ul>
            <li>Model output can be incomplete, incorrect, or overconfident; verify against the original media.</li>
            <li>Physical-world tasks can carry safety consequences; keep humans in the review loop for high-risk uses.</li>
            <li>Apply prompt, content, privacy, and access controls that match the deployment environment.</li>
          </ul>
        </StaticSection>

        <StaticSection title="Deployment">
          <dl>
            <dt>Supported input</dt>
            <dd>{ACCEPTED_MEDIA_HELP}</dd>
            <dt>GPU</dt>
            <dd>{backendInfo?.gpu_name || "Detected on target host"}</dd>
            <dt>vLLM launch command</dt>
            <dd>
              <code>{displayValue(backendInfo?.vllm?.process?.command)}</code>
            </dd>
            <dt>Privacy</dt>
            <dd>Do not upload confidential or personal data unless expressly permitted.</dd>
          </dl>
        </StaticSection>

        <StaticSection title="Getting Help / Support">
          <p>
            For model safety concerns, security issues, or deployment policy questions, use the NVIDIA support and AI
            concern channels linked from the source system card.
          </p>
        </StaticSection>

        <a className="staticLink" href={BUILD_REASON2_SYSTEM_CARD_URL} rel="noreferrer" target="_blank">
          NVIDIA Build system-card reference <ExternalLink size={16} />
        </a>
      </div>
    );
  }
  return (
    <div className="staticPanel">
      <p className="staticEyebrow">Linux with Docker</p>
      <h2>Deploy</h2>
      <p className="staticLead">
        Follow the NVIDIA Build deployment flow for a downloadable NIM, with the model references adapted to
        <code> {modelShortName(model)}</code>. After the service is running, test the same local OpenAI-compatible
        endpoint with a multimodal chat completion request.
      </p>

      <StaticSection title="Step 1: Generate API Key">
        <p>
          Sign in to NVIDIA Build or NGC, create an API key, and use it to authenticate against the NVIDIA container
          registry before pulling the NIM image.
        </p>
      </StaticSection>

      <StaticSection title="Step 2: Pull and Run the NIM">
        <p>
          Active image: <code>{imageName}</code>
        </p>
        <pre className="codeBlock">{DEPLOY_DOCKER_COMMAND(model, backendInfo)}</pre>
      </StaticSection>

      <StaticSection title="Step 3: Test the NIM">
        <pre className="codeBlock">{DEPLOY_CURL_COMMAND(model)}</pre>
      </StaticSection>

      <a className="staticLink" href={BUILD_REASON2_DEPLOY_URL} rel="noreferrer" target="_blank">
        NVIDIA Build deploy reference <ExternalLink size={16} />
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
