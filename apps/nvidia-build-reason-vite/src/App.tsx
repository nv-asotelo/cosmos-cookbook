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

type MediaState = {
  name: string;
  kind: "video" | "image";
  previewUrl: string;
  dataUrl: string;
};

type ImageSize = {
  width: number;
  height: number;
};

type SpatialMark = {
  id: string;
  kind: "point" | "bbox";
  sourceKey: string;
  coordinateMode: "cosmos-1000" | "unit" | "pixel";
  x: number;
  y: number;
  width: number;
  height: number;
  centerX: number;
  centerY: number;
  label: string;
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
  judgeNote?: string;
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
      'You are given the task "Move the tape into the basket". Specify the 2D trajectory your end effector should follow. Return Cosmos grounding JSON using coordinates normalized to a 0-1000 image plane, like this: {"point_2d": [x, y], "label": "gripper trajectory"}.',
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
      'Locate the bounding box of the load and determine if its size and weight of load within the forklift\'s limits. Estimate weights. Return valid JSON using Cosmos grounding coordinates, not COCO boxes: {"bbox_2d":[x1,y1,x2,y2],"label":"load","estimated_weight":"...","within_limit":true}. bbox_2d coordinates must be normalized to a 0-1000 image plane.',
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

function parseResponseJsonPayload(text: string): unknown | null {
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

function spatialLabel(record: Record<string, unknown>, fallback: string) {
  const value = record.label ?? record.category_name ?? record.name ?? record.class ?? record.description;
  return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

function spatialSequence(record: Record<string, unknown>, fallback: number) {
  const value = Number(record.sequence ?? record.step ?? record.index ?? record.id);
  return Number.isFinite(value) ? value : fallback;
}

function hasSpatialPayload(text: string) {
  const payload = parseResponseJsonPayload(text);
  return payload !== null && collectSpatialRecords(payload).length > 0;
}

function parseSpatialMarks(text: string, imageSize: ImageSize) {
  const payload = parseResponseJsonPayload(text);
  const records = payload === null ? [] : collectSpatialRecords(payload);
  const marks: SpatialMark[] = [];
  records.forEach((record, index) => {
    const sequence = spatialSequence(record, index + 1);
    const label = spatialLabel(record, `point ${sequence}`);

    for (const key of POINT_KEYS) {
      const point = normalizePoint(key, record[key], imageSize);
      if (!point) continue;
      const side = Math.max(24, Math.min(imageSize.width, imageSize.height) * 0.055);
      const [centerX, centerY, coordinateMode] = point;
      marks.push({
        id: `${key}-${index}-point`,
        kind: "point",
        sourceKey: key,
        coordinateMode,
        x: centerX - side / 2,
        y: centerY - side / 2,
        width: side,
        height: side,
        centerX,
        centerY,
        label,
        sequence
      });
      break;
    }

    for (const key of BBOX_KEYS) {
      const bbox = normalizeBbox(key, record[key], imageSize);
      if (!bbox) continue;
      const [x, y, width, height, coordinateMode] = bbox;
      marks.push({
        id: `${key}-${index}-bbox`,
        kind: "bbox",
        sourceKey: key,
        coordinateMode,
        x,
        y,
        width,
        height,
        centerX: x + width / 2,
        centerY: y + height / 2,
        label,
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
  const [result, setResult] = useState<ApiResult | null>(null);
  const [streamState, setStreamState] = useState<StreamState>(() => idleStreamState());
  const [copied, setCopied] = useState(false);
  const vlaMode = isVlaMode(model, backendInfo);
  const activeExamples = useMemo(() => (vlaMode ? ALPAMAYO_LINGOQA_EXAMPLES : EXAMPLES), [vlaMode]);

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

  async function mediaFromExample(example: ExampleItem): Promise<MediaState> {
    if (example.mediaUrl.startsWith("/")) {
      const response = await fetch(example.mediaUrl);
      if (!response.ok) {
        throw new Error(`Could not load example media (${response.status})`);
      }
      const blob = await response.blob();
      const mime = blob.type || (example.mediaKind === "image" ? "image/png" : "video/mp4");
      return {
        name: example.mediaName,
        kind: mime.startsWith("image/") ? "image" : "video",
        previewUrl: example.mediaUrl,
        dataUrl: await readBlobAsDataUrl(blob)
      };
    }

    const params = new URLSearchParams({ url: example.mediaUrl, name: example.mediaName });
    const response = await fetch(`/api/example-media?${params.toString()}`);
    if (!response.ok) {
      const error = await response.json().catch(() => null);
      throw new Error(error?.message || "Could not load example media");
    }
    const data = (await response.json()) as { dataUrl: string; mime: string; name: string };
    return {
      name: data.name,
      kind: data.mime.startsWith("image/") ? "image" : "video",
      previewUrl: example.mediaUrl,
      dataUrl: data.dataUrl
    };
  }

  async function applyExample(exampleId = selectedExampleId, options: { closeModal?: boolean } = {}) {
    const closeModal = options.closeModal ?? true;
    const example = activeExamples.find((item) => item.id === exampleId) || activeExamples[0] || EXAMPLES[0];
    setSelectedExampleId(example.id);
    if (closeModal) setExamplesOpen(false);
    setStatus("Loading example");
    try {
      setReasoningEnabled(example.reasoning);
      setUserPrompt(promptForReasoning(example.userPrompt, example.reasoning));
      setSystemPrompt(example.systemPrompt);
      applyExampleParameters(example);
      setMedia(await mediaFromExample(example));
      setResult(null);
      setStreamState(idleStreamState(model));
      setOutputTab("preview");
      setStatus("Example loaded");
    } catch (error) {
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
    setFramesPerSecond(params.framesPerSecond ?? mediaDefaults.fps);
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
          video: media?.kind === "video" ? media.dataUrl : undefined,
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
            examples={activeExamples}
            examplesOpen={examplesOpen}
            handleDrag={handleDrag}
            handleDrop={handleDrop}
            handleFile={handleFile}
            inputRef={inputRef}
            isRunning={isRunning}
            jsonOutput={jsonOutput}
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
            seed={seed}
            selectedExampleId={selectedExampleId}
            setExamplesOpen={setExamplesOpen}
            setFramesPerSecond={setFramesPerSecond}
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

function ExperiencePanel({
  applyExample,
  backendInfo,
  copied,
  copyRequest,
  dragActive,
  examples,
  examplesOpen,
  framesPerSecond,
  handleDrag,
  handleDrop,
  handleFile,
  inputRef,
  isRunning,
  jsonOutput,
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
  seed,
  selectedExampleId,
  setExamplesOpen,
  setFramesPerSecond,
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
  examples: ExampleItem[];
  examplesOpen: boolean;
  framesPerSecond: number;
  handleDrag: (event: DragEvent<HTMLElement>, active: boolean) => void;
  handleDrop: (event: DragEvent<HTMLElement>) => Promise<void>;
  handleFile: (event: ChangeEvent<HTMLInputElement>) => Promise<void>;
  inputRef: RefObject<HTMLInputElement | null>;
  isRunning: boolean;
  jsonOutput: unknown;
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
  seed: number;
  selectedExampleId: string;
  setExamplesOpen: (open: boolean) => void;
  setFramesPerSecond: (value: number) => void;
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

  async function runWithOutputVisible() {
    setMobilePanel("output");
    await run();
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

          <div className="runBar">
            <button className="resetButton" onClick={resetWithInputVisible} type="button">
              <RotateCcw size={16} />
              Reset
            </button>
            <button className={`runButton${isRunning ? " running" : ""}`} onClick={runWithOutputVisible} type="button">
              {isRunning ? null : <Play size={16} fill="currentColor" />}
              {isRunning ? "Quit Task" : "Run"}
            </button>
          </div>
          <div className="statusLine" role="status">
            {status}
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
  const [isApplying, setIsApplying] = useState(false);

  useEffect(() => {
    if (examples.some((example) => example.id === selectedExampleId)) {
      setDraftExampleId(selectedExampleId);
      return;
    }
    if (examples[0]) {
      setDraftExampleId(examples[0].id);
      setSelectedExampleId(examples[0].id);
    }
  }, [examples, selectedExampleId, setSelectedExampleId]);

  function handleDone() {
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
          <p>Select the input from the examples below:</p>
          <div className="exampleList" role="radiogroup" aria-label="Examples">
            {examples.map((example) => {
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
        </div>
        <div className="modalFooter">
          <button className="runButton" disabled={isApplying} onClick={handleDone} type="button" aria-busy={isApplying}>
            {isApplying ? "Loading" : "Done"}
          </button>
        </div>
      </div>
    </div>
  );
}

function SpatialTrajectoryOverlay({ media, text }: { media: MediaState | null; text: string }) {
  const [imageSize, setImageSize] = useState<ImageSize | null>(null);
  const canParseSpatial = useMemo(() => hasSpatialPayload(text), [text]);
  const marks = useMemo(() => (imageSize ? parseSpatialMarks(text, imageSize) : []), [imageSize, text]);
  const points = marks.filter((mark) => mark.kind === "point");
  const boxes = marks.filter((mark) => mark.kind === "bbox");

  if (!media || media.kind !== "image" || !canParseSpatial) return null;

  return (
    <article className="spatialOverlayCard">
      <div className="spatialTopline">
        <div>
          <p className="responseLabel">Response JSON boxes</p>
          <h3>Point and bbox fields rendered from the final answer only</h3>
        </div>
        <span>
          {points.length} points · {boxes.length} boxes
        </span>
      </div>
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
            {points.length > 1 ? (
              <polyline className="spatialPath" points={points.map((point) => `${point.centerX},${point.centerY}`).join(" ")} />
            ) : null}
            {marks.map((mark) => (
              <g key={mark.id}>
                <rect
                  className={mark.kind === "point" ? "spatialPointBox" : "spatialBbox"}
                  height={mark.height}
                  rx={Math.max(4, Math.min(mark.width, mark.height) * 0.08)}
                  width={mark.width}
                  x={mark.x}
                  y={mark.y}
                />
                {mark.kind === "point" ? <circle className="spatialPointDot" cx={mark.centerX} cy={mark.centerY} r={5} /> : null}
                <text className="spatialPointLabel" x={mark.x + 8} y={Math.max(18, mark.y - 8)}>
                  {mark.sequence}
                </text>
              </g>
            ))}
          </svg>
        ) : null}
      </div>
      {marks.length > 0 ? (
        <ol className="spatialSequence" aria-label="Generated coordinate sequence">
          {marks.map((mark) => (
            <li className="spatialSequenceItem" key={`sequence-${mark.id}`}>
              <strong>#{mark.sequence}</strong>
              <span>{mark.label}</span>
              <code>
                {mark.sourceKey} {mark.coordinateMode === "cosmos-1000" ? "0-1000" : mark.coordinateMode} →{" "}
                {mark.kind === "bbox"
                  ? `${Math.round(mark.x)}, ${Math.round(mark.y)}, ${Math.round(mark.width)}×${Math.round(mark.height)}`
                  : `${Math.round(mark.centerX)}, ${Math.round(mark.centerY)}`}
              </code>
            </li>
          ))}
        </ol>
      ) : (
        <p className="spatialLoading">Loading image dimensions for overlay alignment...</p>
      )}
    </article>
  );
}

function PreviewOutput({
  isRunning,
  media,
  parsedOutput,
  reasoningExpanded,
  result,
  setReasoningExpanded,
  streamPhase
}: {
  isRunning: boolean;
  media: MediaState | null;
  parsedOutput: { reasoning: string; answer: string; steps: string[] };
  reasoningExpanded: boolean;
  result: ApiResult | null;
  setReasoningExpanded: (expanded: boolean) => void;
  streamPhase: StreamPhase;
}) {
  if (result?.status === "error" || result?.error) {
    return <pre className="errorBox">{result.message || result.error}</pre>;
  }

  if (result?.status === "skip") {
    return (
      <article className="emptyOutput">
        <h3>Task stopped</h3>
        <p>{result.message || "The current inference task was stopped before completion."}</p>
      </article>
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
    return (
      <div className="responseStack">
        <SpatialTrajectoryOverlay media={media} text={answerText} />
        {hasAnswer ? (
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
    return <GeneratingOutput />;
  }

  return (
    <article className="emptyOutput">
      <h3>Ready for inference</h3>
      <p>Upload media or choose an example, then run the model to see the response and reasoning trace.</p>
    </article>
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
            min={2}
            max={8}
            step={1}
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
