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
const EMBODIED_EGO_DRILL_VIDEO = "/examples/embodied-ego-drill.mp4";
const EMBODIED_EGO_DRILL_POSTER = "/examples/embodied-ego-drill-poster.jpg";
const TENNIS_TEMPORAL_EXAMPLE_ID = "tennis-temporal-events";
const TENNIS_TEMPORAL_VIDEO = "/examples/tennis_nim_safe.mp4";
const WAREHOUSE_ROW_D_VIDEO = "/examples/warehouse_7min.mp4";
const AV_EGO_RAPID_SCENE_VIDEO = "/examples/av-ego-rapid-scene.mp4";
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
type ExampleGroupId = "build" | "vss" | "av-dense-captioning" | "embodied-reasoning";

const LONG_VIDEO_PRESET_FPS: Record<LongPreset, number> = {
  fast: 6,
  balanced: 8,
  detailed: 10
};
const LONG_VIDEO_DEFAULT_CONCURRENCY = 16;

type MediaState = {
  name: string;
  kind: "video" | "image";
  previewUrl: string;
  dataUrl: string;
  sourceUrl?: string;
  posterUrl?: string;
  planningTrace?: PlanningTrace;
};

type ImageSize = {
  width: number;
  height: number;
};

type SpatialSource = "thinking" | "final";
type SpatialRole = "perception" | "trajectory";

type PlanningTraceFrame = {
  time: string;
  phase: string;
  imageUrl: string;
  caption?: string;
  coordinateFrameIndex?: number;
  renderMode?: "perception" | "trajectory" | "raw";
};

type PlanningTrace = {
  title: string;
  subtitle?: string;
  frames: PlanningTraceFrame[];
  coordinateFrameIndex?: number;
  defaultFrameIndex?: number;
};

type SpatialMark = {
  id: string;
  kind: "point" | "bbox";
  role: SpatialRole;
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
  time?: string;
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
  posterUrl?: string;
  planningTrace?: PlanningTrace;
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

type PlanningStageApiResult = {
  phase?: string;
  mode?: string;
  time?: string;
  actual_time?: string;
  elapsed_seconds?: number;
  payload?: unknown;
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
  planning_stages?: PlanningStageApiResult[];
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
  logs: StreamLogEntry[];
};

type StreamLogEntry = {
  label: string;
  detail?: string;
  elapsedSeconds?: number;
  level?: "info" | "warn" | "error";
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
  { id: "av-dense-captioning", label: "AV" },
  { id: "embodied-reasoning", label: "Embodied" }
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

const ROBOT_ARM_TRAJECTORY_PROMPT =
  'You are given the task "Move the tape into the basket". Specify the 2D trajectory your end effector should follow in pixel space. Return the trajectory coordinates in JSON format like this: {"point_2d": [x, y], "label": "gripper trajectory"}.\n\nPrompt format:\nAnswer the question using the following format:\n<think>\nYour reasoning.\n</think>\nWrite your final answer immediately after the </think> tag.';

const ROBOT_EGO_DRILL_PROMPT =
  'You are given the task "Pick up the Black+Decker drill and place it into the yellow box". Specify the 2D trajectory your end effector should follow in pixel space. Return the trajectory coordinates in JSON format like this: {"point_2d": [x, y], "label": "gripper trajectory"}.';

const ROBOT_EGO_DRILL_TRACE: PlanningTrace = {
  title: "Robot planning trace",
  subtitle: "Perceive boxes use the 00:00 frame; Grasp trajectory uses the Grasp-frame coordinate space",
  defaultFrameIndex: 0,
  frames: [
    {
      time: "00:00.00",
      phase: "Perceive",
      imageUrl: "/examples/embodied-ego-drill-frame-01.jpg",
      coordinateFrameIndex: 0,
      renderMode: "perception",
      caption: "Establish drill, yellow box, gripper, and support surface."
    },
    {
      time: "00:01.00",
      phase: "Approach",
      imageUrl: "/examples/embodied-ego-drill-frame-02.jpg",
      renderMode: "raw",
      caption: "Move the end effector toward the drill handle."
    },
    {
      time: "00:02.00",
      phase: "Grasp",
      imageUrl: "/examples/embodied-ego-drill-frame-03.jpg",
      coordinateFrameIndex: 2,
      renderMode: "trajectory",
      caption: "Close around the drill and confirm the pickup path."
    },
    {
      time: "00:03.00",
      phase: "Transfer",
      imageUrl: "/examples/embodied-ego-drill-frame-04.jpg",
      renderMode: "raw",
      caption: "Lift and carry the drill toward the yellow box."
    },
    {
      time: "00:04.00",
      phase: "Place",
      imageUrl: "/examples/embodied-ego-drill-frame-05.jpg",
      renderMode: "raw",
      caption: "Align over the yellow box and release."
    }
  ]
};

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
    userPrompt: ROBOT_ARM_TRAJECTORY_PROMPT,
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
    id: "robot-ego-drill-planning",
    title: "Robot ego planning: drill to yellow box",
    group: "embodied-reasoning",
    mediaUrl: EMBODIED_EGO_DRILL_VIDEO,
    mediaName: "embodied-ego-drill.mp4",
    mediaKind: "video",
    userPrompt: ROBOT_EGO_DRILL_PROMPT,
    systemPrompt: "You are a helpful assistant.",
    reasoning: true,
    posterUrl: EMBODIED_EGO_DRILL_POSTER,
    planningTrace: ROBOT_EGO_DRILL_TRACE,
    parameters: {
      framesPerSecond: 6,
      maxTokens: 4096,
      repetitionPenalty: 1.2,
      temperature: 0.3,
      topP: 0.3
    }
  },
  {
    id: "robot-arm-embodied",
    title: "robot arm pick up stuff",
    group: "embodied-reasoning",
    mediaUrl: ROBOT_TAPE_IMAGE,
    mediaName: "robot_tape.png",
    mediaKind: "image",
    userPrompt: ROBOT_ARM_TRAJECTORY_PROMPT,
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
  "Answer the user's driving-scene question from visible evidence in the sampled frames.";

const AV_EGO_ACTION_EXAMPLE: ExampleItem = {
  id: "av-ego-rapid-action-plan",
  title: "AV: rapid ego-car action plan",
  group: "av-dense-captioning",
  mediaUrl: AV_EGO_RAPID_SCENE_VIDEO,
  mediaName: "av-ego-rapid-scene.mp4",
  mediaKind: "video",
  userPrompt:
    'Analyze this rapid ego-vehicle driving scene at high temporal resolution. Focus on what the ego car should do, not generic scene captioning.\n\nUse only visible evidence from the video and preserve exact timestamps in "mm:ss.ff" format. Mention road actors, traffic controls, lane markings, obstacles, and right-of-way cues only when they affect risk or ego action.\n\nUse concrete ego actions such as maintain lane, maintain speed, slow, brake, yield, stop, wait, creep forward, steer left/right within lane, proceed, or accelerate. Do not infer hidden actors or between-sample events. If the scene is static and does not change the ego action, omit it from the event list.\n\nReturn the final answer as valid JSON with this shape: {"events":[{"start":"mm:ss.ff","end":"mm:ss.ff","risk":"","ego_action":"","confidence":0.0,"caption":""}],"timeline_summary":[{"start":"mm:ss.ff","end":"mm:ss.ff","ego_policy":"","key_reason":""}],"uncertain_events":[],"sampling_limits":{"rapid_motion":"","occlusion":"","would_higher_fps_help":true}}.\n\nAnswer the question using the following format:\n\n<think>\nYour reasoning.\n</think>\n\nWrite your final answer immediately after the </think> tag.',
  systemPrompt:
    "You are an autonomous-driving video analyst. Use only visible evidence, preserve exact timestamps, separate static context from dynamic actors, and give conservative concrete ego-car actions.",
  reasoning: true,
  longVideoEnabled: true,
  parameters: {
    framesPerSecond: 8,
    maxTokens: 4096,
    presencePenalty: 0,
    repetitionPenalty: 1.0,
    temperature: 0.6,
    topK: 20,
    topP: 0.95
  }
};

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

const AV_DENSE_CAPTIONING_EXAMPLES: ExampleItem[] = [
  AV_EGO_ACTION_EXAMPLE,
  ...ALPAMAYO_LINGOQA_EXAMPLES.map((example) => ({
    ...example,
    group: "av-dense-captioning" as ExampleGroupId,
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
  }))
];

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
  const chunks: BlobPart[] = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(new Uint8Array(value));
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

function assistantContentFromOpenAi(value: unknown): string {
  if (!isObjectRecord(value) || !Array.isArray(value.choices)) return "";
  for (const choice of value.choices) {
    if (!isObjectRecord(choice)) continue;
    const message = isObjectRecord(choice.message) ? choice.message : isObjectRecord(choice.delta) ? choice.delta : null;
    const content = message?.content ?? choice.content;
    const text = openAiContentText(content);
    if (text) return text;
  }
  return "";
}

function resultContentText(result: ApiResult | null): string | undefined {
  if (!result) return undefined;
  return (
    result.content ||
    result.combined_content ||
    assistantContentFromOpenAi(result) ||
    assistantContentFromOpenAi(result.openai) ||
    assistantContentFromOpenAi(result.raw) ||
    undefined
  );
}

const POINT_KEYS = [
  "point_2d",
  "point",
  "position",
  "coordinate",
  "coordinates",
  "center",
  "center_point",
  "target_point",
  "waypoint",
  "end_effector_point"
];
const BBOX_KEYS = [
  "bbox_2d",
  "box_2d",
  "bounding_box",
  "bounding_box_2d",
  "bbox",
  "bbox2d",
  "box",
  "box2d",
  "bounds",
  "bounds_2d",
  "rect",
  "rectangle",
  "rectangle_2d"
];
const TRAJECTORY_KEYS = [
  "annotations",
  "detections",
  "end_effector_path",
  "grasp_plan",
  "grasp_trajectory",
  "objects",
  "perception",
  "plan",
  "planned_path",
  "planned_trajectory",
  "points",
  "route",
  "spatial",
  "spatial_annotations",
  "steps",
  "trajectory",
  "waypoints"
];
const COSMOS_COORD_MAX = 1000;
const INTERNAL_SPATIAL_ROLE_KEY = "__spatialRole";

function isObjectRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function finiteNumber(value: unknown): number | null {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function numberField(record: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    const value = finiteNumber(record[key]);
    if (value !== null) return value;
  }
  return null;
}

function numberArray(value: unknown): number[] | null {
  if (Array.isArray(value)) {
    const flattened = value.flatMap((item) => (Array.isArray(item) ? item : [item]));
    const numbers = flattened.map((item) => Number(item));
    return numbers.every((item) => Number.isFinite(item)) ? numbers : null;
  }
  if (isObjectRecord(value)) {
    const x1 = numberField(value, ["x1", "x_min", "xmin", "left"]);
    const y1 = numberField(value, ["y1", "y_min", "ymin", "top"]);
    const x2 = numberField(value, ["x2", "x_max", "xmax", "right"]);
    const y2 = numberField(value, ["y2", "y_max", "ymax", "bottom"]);
    if (x1 !== null && y1 !== null && x2 !== null && y2 !== null) return [x1, y1, x2, y2];

    const x = numberField(value, ["x", "left", "cx", "center_x"]);
    const y = numberField(value, ["y", "top", "cy", "center_y"]);
    const width = numberField(value, ["width", "w"]);
    const height = numberField(value, ["height", "h"]);
    if (x !== null && y !== null && width !== null && height !== null) {
      return [x, y, width, height];
    }
    if (x !== null && y !== null) return [x, y];
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

function spatialRoleForContainer(key: string, currentRole?: SpatialRole): SpatialRole | undefined {
  const lowerKey = key.toLowerCase();
  if (
    lowerKey.includes("trajectory") ||
    lowerKey.includes("waypoint") ||
    lowerKey.includes("path") ||
    lowerKey.includes("route") ||
    lowerKey.includes("grasp") ||
    lowerKey === "plan" ||
    lowerKey === "points"
  ) {
    return "trajectory";
  }
  if (
    lowerKey.includes("perception") ||
    lowerKey.includes("object") ||
    lowerKey.includes("detection") ||
    lowerKey.includes("annotation")
  ) {
    return "perception";
  }
  return currentRole;
}

function spatialRoleForRecord(record: Record<string, unknown>): SpatialRole | undefined {
  const role = record[INTERNAL_SPATIAL_ROLE_KEY];
  return role === "perception" || role === "trajectory" ? role : undefined;
}

function collectSpatialRecords(value: unknown, records: Record<string, unknown>[] = [], depth = 0, role?: SpatialRole) {
  if (depth > 8) return records;
  if (Array.isArray(value)) {
    value.forEach((item) => collectSpatialRecords(item, records, depth + 1, role));
    return records;
  }
  if (!isObjectRecord(value)) return records;

  const hasSpatialField = [...POINT_KEYS, ...BBOX_KEYS].some((key) => value[key] !== undefined);
  if (hasSpatialField) {
    if (role) value[INTERNAL_SPATIAL_ROLE_KEY] = role;
    records.push(value);
  }

  for (const [key, nested] of Object.entries(value)) {
    if ([...POINT_KEYS, ...BBOX_KEYS, INTERNAL_SPATIAL_ROLE_KEY].includes(key)) continue;
    if (TRAJECTORY_KEYS.includes(key) || Array.isArray(nested) || isObjectRecord(nested)) {
      collectSpatialRecords(nested, records, depth + 1, spatialRoleForContainer(key, role));
    }
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
  const looksLikeCosmosKey =
    lowerKey.includes("2d") ||
    lowerKey.includes("coordinate") ||
    lowerKey.includes("position") ||
    lowerKey.includes("point") ||
    lowerKey.includes("waypoint");
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

function normalizeBboxPointFallback(
  key: string,
  value: unknown,
  imageSize: ImageSize
): [number, number, "cosmos-1000" | "unit" | "pixel"] | null {
  const numbers = numberArray(value);
  if (!numbers || numbers.length !== 2) return null;
  return normalizePoint(key, value, imageSize);
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
      "object_label",
      "object_name",
      "category_name",
      "name",
      "class",
      "object",
      "target",
      "action",
      "action_label",
      "description",
      "phase",
      "phase_label",
      "stage",
      "step_name",
      "step_label"
    ]) || fallback
  );
}

function spatialDetail(record: Record<string, unknown>, label: string) {
  const detail = stringField(record, [
    "caption",
    "reason",
    "rationale",
    "thought",
    "observation",
    "note",
    "description",
    "action"
  ]);
  return detail && detail !== label ? detail : undefined;
}

function spatialTime(record: Record<string, unknown>) {
  return stringField(record, ["time", "timestamp", "start"]);
}

function spatialSequence(record: Record<string, unknown>, fallback: number) {
  const value = Number(record.sequence ?? record.step ?? record.index ?? record.id);
  return Number.isFinite(value) ? value : fallback;
}

function coordinateTracePairs(text: string) {
  const pairs: Array<{ x: number; y: number; index: number; endIndex: number; raw: string }> = [];
  const coordinatePattern = /\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]/g;
  for (const match of text.matchAll(coordinatePattern)) {
    const x = Number(match[1]);
    const y = Number(match[2]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const index = match.index || 0;
    pairs.push({ x, y, index, endIndex: index + match[0].length, raw: match[0] });
  }
  return pairs;
}

function coordinateListItems(text: string) {
  const items = new Map<number, { label: string; phrase: string }>();
  const listPattern = /([A-Za-z][A-Za-z0-9 +&/.'’+-]{1,120}?)\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]/g;
  for (const match of text.matchAll(listPattern)) {
    const matchIndex = match.index || 0;
    const coordinateOffset = match[0].lastIndexOf("[");
    const coordinateIndex = matchIndex + Math.max(0, coordinateOffset);
    const sentenceStart = Math.max(
      text.lastIndexOf(".", matchIndex),
      text.lastIndexOf(";", matchIndex),
      text.lastIndexOf("\n", matchIndex),
      text.lastIndexOf("<think>", matchIndex)
    );
    const sentencePrefix = text.slice(Math.max(0, sentenceStart + 1), matchIndex);
    const appearsInObjectList = /(?:locate|position|positions|item|items|object|objects|visible|coordinates?)\b/i.test(sentencePrefix);
    if (!appearsInObjectList) continue;

    const rawLabel = match[1].split(/[:\n]/).pop() || "";
    const label = trimTraceLabel(rawLabel.replace(/^(?:and|then|so|also|next)\s+/i, ""));
    if (!label || /^(?:based|this|that|i|after|before|then|so|current|next|position)$/i.test(label)) continue;
    if (label.split(/\s+/).length > 8) continue;

    const phrase = `${label} [${match[2]}, ${match[3]}]`;
    items.set(coordinateIndex, { label, phrase });
  }
  return items;
}

function hasCoordinateTracePayload(text: string) {
  const pairs = coordinateTracePairs(text);
  return (
    pairs.length > 0 &&
    /point_2d|trajectory|waypoint|gripper|end.?effector|coordinate|located at|positioned at|basket|tape|drill|box/i.test(text)
  );
}

function hasSpatialPayload(text: string) {
  return parseJsonPayloads(text).some((payload) => collectSpatialRecords(payload).length > 0) || hasCoordinateTracePayload(text);
}

function trimTraceLabel(label: string) {
  return label
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .replace(/^(?:the|a|an|my|its|this|that)\s+/i, "")
    .replace(/\s+(?:located|positioned|sitting|placed|shown|visible)$/i, "")
    .trim();
}

function tailTraceLabel(rawLabel: string) {
  const clause = rawLabel.split(/[.;:\n]/).pop() || rawLabel;
  const pieces = clause
    .split(/\b(?:over to|towards?|to|inside|within|near|around|at)\s+/i)
    .map((part) => part.trim())
    .filter(Boolean);
  const label = trimTraceLabel(pieces.at(-1) || clause);
  if (!label) return "";
  const words = label.split(" ");
  return words.slice(Math.max(0, words.length - 6)).join(" ");
}

function localCoordinateContext(before: string) {
  const previousCoordinate = before.lastIndexOf("]");
  const previousSentence = Math.max(before.lastIndexOf("."), before.lastIndexOf(";"), before.lastIndexOf("\n"));
  const start = Math.max(previousCoordinate, previousSentence);
  return before.slice(Math.max(0, start + 1));
}

function coordinateTracePhrase(text: string, pair: { index: number; endIndex: number; raw?: string }) {
  const before = text.slice(0, pair.index);
  const after = text.slice(pair.endIndex);
  const sentenceStart = Math.max(
    before.lastIndexOf("."),
    before.lastIndexOf(";"),
    before.lastIndexOf("\n"),
    before.lastIndexOf("<think>")
  );
  const previousCoordinate = before.lastIndexOf("]");
  const likelyListItem = previousCoordinate >= 0 && pair.index - previousCoordinate < 24;
  const start = likelyListItem
    ? Math.max(previousCoordinate + 1, pair.index - 28)
    : Math.max(0, sentenceStart + 1, pair.index - 140);
  const sentenceEndRelative = after.search(/[.;\n]/);
  const nextCoordinateRelative = after.search(/,\s*\[/);
  const endCandidates = [text.length, pair.endIndex + 140];
  if (sentenceEndRelative >= 0) endCandidates.push(pair.endIndex + sentenceEndRelative);
  if (nextCoordinateRelative >= 0 && nextCoordinateRelative < 40) endCandidates.push(pair.endIndex);
  const end = Math.max(pair.endIndex, Math.min(...endCandidates));
  const phrase = text
    .slice(start, end)
    .replace(/\s+/g, " ")
    .replace(/^[:,\s]+/, "")
    .trim();
  return phrase || pair.raw;
}

function hasLocalGripperPositionCue(before: string) {
  return /current\s+(?:position|pose)|my\s+(?:left\s+)?gripper|end.?effector/i.test(localCoordinateContext(before).slice(-90));
}

function traceObjectLabel(before: string) {
  const localBefore = localCoordinateContext(before);
  if (hasLocalGripperPositionCue(before)) {
    return "gripper position";
  }
  const anchored = localBefore.match(
    /([a-z][a-z0-9 +&/.'-]{2,100}?)\s+(?:located|positioned|sitting|placed|centered)\s+(?:at|near|inside|within|around)?\s*$/i
  );
  const anchoredLabel = tailTraceLabel(anchored?.[1] || "");
  if (anchoredLabel) return anchoredLabel;
  const match = localBefore.match(
    /(?:to|toward|towards|over to|at|of|the|a|an)\s+([a-z][a-z0-9 +&/.-]{2,70}?)(?:\s+(?:located|positioned|sitting|placed|is|at|around|near))?\s*$/i
  );
  return tailTraceLabel(match?.[1] || "");
}

function traceCoordinateRole(text: string, pairIndex: number, before: string): SpatialRole {
  const objectLabel = traceObjectLabel(before);
  const localTraceCue = /next\s+steps|trajectory|waypoints|point_2d/i.test(before.slice(-90));
  if (objectLabel && objectLabel !== "gripper position" && !localTraceCue) return "perception";

  const traceStartCandidates = [
    text.toLowerCase().lastIndexOf("next steps", pairIndex),
    text.toLowerCase().lastIndexOf("trajectory", pairIndex),
    text.toLowerCase().lastIndexOf("waypoints", pairIndex),
    text.toLowerCase().lastIndexOf("point_2d", pairIndex)
  ].filter((index) => index >= 0);
  const traceStart = traceStartCandidates.length > 0 ? Math.max(...traceStartCandidates) : -1;
  if (traceStart >= 0 && pairIndex - traceStart < 240) return "trajectory";
  if (/current\s+(?:position|pose)|next\s+steps|trajectory|waypoint|gripper trajectory|point_2d/i.test(before)) {
    return "trajectory";
  }
  return "perception";
}

function parseCoordinateTraceMarks(text: string, imageSize: ImageSize, source: SpatialSource, sequenceStart: number) {
  if (!hasCoordinateTracePayload(text)) return [];
  const side = Math.max(24, Math.min(imageSize.width, imageSize.height) * 0.055);
  const listItems = coordinateListItems(text);
  let sequence = sequenceStart;
  const seen = new Set<string>();
  return coordinateTracePairs(text)
    .map((pair, index) => {
      const before = text.slice(Math.max(0, pair.index - 150), pair.index);
      const listItem = listItems.get(pair.index);
      const inferredRole = traceCoordinateRole(text, pair.index, before);
      const role = listItem ? "perception" : inferredRole;
      if (role === "perception" && !listItem) return null;
      const label =
        role === "perception" && listItem
          ? listItem.label
          : hasLocalGripperPositionCue(before)
            ? "gripper start"
            : "gripper trajectory";
      const [centerX, centerY, coordinateMode] = scalePair("reasoning point_2d", pair.x, pair.y, imageSize);
      const clampedX = Math.max(0, Math.min(imageSize.width, centerX));
      const clampedY = Math.max(0, Math.min(imageSize.height, centerY));
      const tracePhrase = listItem?.phrase || coordinateTracePhrase(text, pair);
      const duplicateKey = `${role}:${label.toLowerCase()}:${Math.round(clampedX)}:${Math.round(clampedY)}`;
      if (seen.has(duplicateKey)) return null;
      seen.add(duplicateKey);
      const mark: SpatialMark = {
        id: `${source}-trace-coordinate-${pair.index}-${index}`,
        kind: "point",
        role,
        sourceKey: role === "perception" ? "reasoning object point" : "reasoning trajectory point",
        coordinateMode,
        source,
        x: Math.max(0, Math.min(imageSize.width - side, clampedX - side / 2)),
        y: Math.max(0, Math.min(imageSize.height - side, clampedY - side / 2)),
        width: side,
        height: side,
        centerX: clampedX,
        centerY: clampedY,
        label,
        detail: tracePhrase,
        sequence
      };
      sequence += 1;
      return mark;
    })
    .filter((mark): mark is SpatialMark => Boolean(mark))
    .slice(0, 80);
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
    const time = spatialTime(record);
    const recordRole = spatialRoleForRecord(record);
    let addedPoint = false;

    for (const key of POINT_KEYS) {
      const point = normalizePoint(key, record[key], imageSize);
      if (!point) continue;
      const side = Math.max(24, Math.min(imageSize.width, imageSize.height) * 0.055);
      const [centerX, centerY, coordinateMode] = point;
      marks.push({
        id: `${source}-${key}-${index}-point`,
        kind: "point",
        role: recordRole || "trajectory",
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
        time,
        sequence
      });
      addedPoint = true;
      break;
    }

    for (const key of BBOX_KEYS) {
      const bbox = normalizeBbox(key, record[key], imageSize);
      if (!bbox) continue;
      const [x, y, width, height, coordinateMode] = bbox;
      if (recordRole === "trajectory") {
        const side = Math.max(24, Math.min(imageSize.width, imageSize.height) * 0.055);
        const centerX = x + width / 2;
        const centerY = y + height / 2;
        marks.push({
          id: `${source}-${key}-${index}-bbox-trajectory-center`,
          kind: "point",
          role: "trajectory",
          sourceKey: `${key} center`,
          coordinateMode,
          source,
          x: centerX - side / 2,
          y: centerY - side / 2,
          width: side,
          height: side,
          centerX,
          centerY,
          label,
          detail: detail ? `${detail} Trajectory bbox rendered as its center point.` : "Trajectory bbox rendered as its center point.",
          time,
          sequence
        });
        break;
      }
      marks.push({
        id: `${source}-${key}-${index}-bbox`,
        kind: "bbox",
        role: recordRole || "perception",
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
        time,
        sequence
      });
      break;
    }

    if (!addedPoint) {
      for (const key of BBOX_KEYS) {
        const point = normalizeBboxPointFallback(key, record[key], imageSize);
        if (!point) continue;
        const side = Math.max(24, Math.min(imageSize.width, imageSize.height) * 0.055);
        const [centerX, centerY, coordinateMode] = point;
        const fallbackDetail = detail
          ? `${detail} Bbox value had two coordinates, so it is shown as a point marker.`
          : "Bbox value had two coordinates, so it is shown as a point marker.";
        marks.push({
          id: `${source}-${key}-${index}-bbox-point`,
          kind: "point",
          role: recordRole || "perception",
          sourceKey: `${key} point`,
          coordinateMode,
          source,
          x: centerX - side / 2,
          y: centerY - side / 2,
          width: side,
          height: side,
          centerX,
          centerY,
          label,
          detail: fallbackDetail,
          time,
          sequence
        });
        break;
      }
    }
  });
  if (source === "thinking" || records.length === 0) {
    marks.push(...parseCoordinateTraceMarks(text, imageSize, source, fallbackSequence));
  }
  return marks.slice(0, 160);
}

function makeStreamState(model: string): StreamState {
  return {
    phase: "waiting_first_token",
    reasoning: "",
    answer: "",
    schema: "plain_content",
    created: Math.floor(Date.now() / 1000),
    model,
    logs: []
  };
}

function idleStreamState(model = DEFAULT_MODEL): StreamState {
  return {
    phase: "idle",
    reasoning: "",
    answer: "",
    schema: "plain_content",
    created: Math.floor(Date.now() / 1000),
    model,
    logs: []
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

function openAiContentText(content: unknown): string {
  if (typeof content === "string") return content.trim();
  if (!Array.isArray(content)) return "";
  return content
    .map((part) => {
      if (typeof part === "string") return part;
      if (isObjectRecord(part) && typeof part.text === "string") return part.text;
      return "";
    })
    .filter(Boolean)
    .join("\n\n")
    .trim();
}

function openAiContentImageCount(content: unknown): number {
  if (!Array.isArray(content)) return 0;
  return content.filter(
    (part) =>
      isObjectRecord(part) &&
      (part.type === "image_url" || part.type === "input_image" || isObjectRecord(part.image_url))
  ).length;
}

function payloadMessages(payload: unknown): Record<string, unknown>[] {
  if (!isObjectRecord(payload) || !Array.isArray(payload.messages)) return [];
  return payload.messages.filter(isObjectRecord);
}

type PlanningStagePromptView = {
  key: string;
  title: string;
  mode: string;
  time: string;
  elapsed: string;
  imageCount: number;
  systemPrompt: string;
  userPrompt: string;
};

function planningStagePromptViews(result: ApiResult | null): PlanningStagePromptView[] {
  const stages = Array.isArray(result?.planning_stages) ? result.planning_stages : [];
  return stages
    .map((stage, index) => {
      const messages = payloadMessages(stage.payload);
      const systemMessage = messages.find((message) => message.role === "system");
      const userMessage = messages.find((message) => message.role === "user");
      const systemPrompt = systemMessage ? openAiContentText(systemMessage.content) : "";
      const userPrompt = userMessage ? openAiContentText(userMessage.content) : "";
      if (!systemPrompt && !userPrompt) return null;
      const phase = typeof stage.phase === "string" && stage.phase ? stage.phase : `Stage ${index + 1}`;
      const mode = typeof stage.mode === "string" && stage.mode ? stage.mode : "planning";
      const time = typeof stage.time === "string" && stage.time ? stage.time : "unknown time";
      const elapsed =
        typeof stage.elapsed_seconds === "number" && Number.isFinite(stage.elapsed_seconds)
          ? `${stage.elapsed_seconds.toFixed(1)}s`
          : "";
      return {
        key: `${phase}-${mode}-${time}-${index}`,
        title: phase,
        mode,
        time,
        elapsed,
        imageCount: userMessage ? openAiContentImageCount(userMessage.content) : 0,
        systemPrompt,
        userPrompt
      };
    })
    .filter((view): view is PlanningStagePromptView => Boolean(view));
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

function isStaticTimelineItem(item: TimelineItem) {
  const title = item.title.trim().toLowerCase();
  const caption = item.caption.trim().toLowerCase();
  return (
    title === "static" ||
    title === "static scene" ||
    title === "stationary" ||
    title === "none" ||
    title === "no change" ||
    title.includes("stationary") ||
    caption.startsWith("the scene remains static") ||
    caption.includes("no visible movement") ||
    caption.includes("remains stationary") ||
    caption.includes("remain stationary") ||
    caption.includes("is stationary") ||
    caption.includes("are stationary")
  );
}

function isSummaryTimelineItem(item: TimelineItem) {
  return item.title.trim().toLowerCase() === "summary";
}

function rangeParts(range: string) {
  const [start, end] = range.split(/\s+-\s+/);
  return {
    start: start || range || "unknown",
    end: end || start || range || "unknown"
  };
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
  const lower = modelName.toLowerCase();
  const vendor = lower.includes("qwen")
    ? "qwen"
    : lower.includes("gemma")
      ? "google"
      : lower.includes("mistral") || lower.includes("ministral")
        ? "mistralai"
        : lower.includes("kimi")
          ? "moonshotai"
          : "nvidia";
  return slug ? `nvcr.io/nim/${vendor}/${slug}:latest` : "<NIM_IMAGE>";
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
  const activeContent = useMemo(() => resultContentText(activeResult), [activeResult]);
  const parsedOutput = useMemo(
    () => parseReasoning(activeContent, activeResult?.reasoning),
    [activeContent, activeResult?.reasoning]
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
          sourceUrl,
          posterUrl: example.posterUrl,
          planningTrace: example.planningTrace
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
        sourceUrl,
        posterUrl: example.posterUrl,
        planningTrace: example.planningTrace
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
      sourceUrl: example.mediaUrl,
      posterUrl: example.posterUrl,
      planningTrace: example.planningTrace
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
    setFramesPerSecond(params.framesPerSecond ?? (example.longVideoEnabled ? LONG_VIDEO_PRESET_FPS[longPreset] : mediaDefaults.fps));
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
    const planningFrames =
      media?.planningTrace?.frames
        .map((frame, index) => ({
          index,
          phase: frame.phase,
          time: frame.time,
          renderMode: planningFrameRenderMode(frame)
        }))
        .filter((frame) => frame.renderMode === "perception" || frame.renderMode === "trajectory") || [];
    const streamEndpoint = planningFrames.length > 0 ? "/api/reason/planning/stream" : "/api/reason/stream";

    try {
      const response = await fetch(streamEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          prompt: effectivePrompt,
          systemPrompt,
          model,
          planningFrames: planningFrames.length > 0 ? planningFrames : undefined,
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
            const data = event.data as { phase?: StreamPhase; note?: string };
            if (data.phase) {
              setStreamState((current) => ({ ...current, phase: data.phase || current.phase }));
              setStatus(data.note || statusForPhase(data.phase));
            }
          } else if (event.event === "planning_stage") {
            const data = event.data as { phase?: string; mode?: string; elapsedSeconds?: number };
            const phaseLabel = data.phase || (data.mode === "trajectory" ? "Grasp" : "Perceive");
            setStatus(`${phaseLabel} pass complete${Number.isFinite(data.elapsedSeconds) ? ` in ${formatDuration(data.elapsedSeconds)}` : ""}`);
          } else if (event.event === "log") {
            const data = event.data as StreamLogEntry;
            const logEntry: StreamLogEntry = {
              label: data.label || "Backend log",
              detail: data.detail,
              elapsedSeconds: data.elapsedSeconds,
              level: data.level || "info"
            };
            setStreamState((current) => ({
              ...current,
              logs: [...(current.logs || []), logEntry].slice(-16)
            }));
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
            const summary = data.summary;
            if (summary) {
              const index = data.index ?? Date.now();
              setLongProgress((current) => {
                const base = current || { preset: longPreset, warnings: [], chunks: [], partialTimeline: [] };
                return {
                  ...base,
                  preset: longPreset,
                  partialTimeline: [
                    ...base.partialTimeline.filter((item) => item.index !== index),
                    { index, timeRange: data.timeRange, summary }
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
            streamCreated={streamState.created}
            streamLogs={streamState.logs}
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
  streamCreated,
  streamLogs,
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
  streamCreated: number;
  streamLogs: StreamLogEntry[];
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
  const selectedExampleLoaded = Boolean(
    selectedExample &&
    media &&
    media?.kind === selectedExample?.mediaKind &&
    media.name === selectedExample.mediaName &&
    (media.previewUrl === selectedExample.mediaUrl || Boolean(media.sourceUrl?.endsWith(selectedExample.mediaUrl)))
  );
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
                    poster={media.posterUrl}
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
            label={media?.planningTrace ? "Robot Task" : "User Prompt"}
            hint={
              media?.planningTrace
                ? "Shared task brief. Vite splits this into separate Perceive and Grasp NIM calls with frame-specific prompts."
                : vlaMode
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
                Higher FPS increases chunk count and wait time. Presets use 6, 8, and 10 FPS for Fast, Balanced,
                and Detailed. Fast is tuned for dense captioning speed, but very brief actions may still be missed.
                Chunk concurrency defaults to 16 on this RTX PRO 6000 based on the live NIM probes; lower it if other
                users share the endpoint.
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
                streamCreated={streamCreated}
                streamLogs={streamLogs}
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
                        <video src={example.mediaUrl} poster={example.posterUrl} muted preload="metadata" />
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

function planningFrameRenderMode(frame?: PlanningTraceFrame) {
  if (!frame) return "raw";
  if (frame.renderMode) return frame.renderMode;
  const phase = frame.phase.toLowerCase();
  if (phase.includes("perceive")) return "perception";
  if (phase.includes("grasp") || phase.includes("plan")) return "trajectory";
  return "raw";
}

function planningFrameCoordinateIndex(planningTrace: PlanningTrace | undefined, frameIndex: number) {
  if (!planningTrace) return 0;
  const selectedIndex = Math.max(0, Math.min(planningTrace.frames.length - 1, frameIndex));
  const explicitIndex = planningTrace.frames[selectedIndex]?.coordinateFrameIndex;
  const coordinateIndex = typeof explicitIndex === "number" ? explicitIndex : selectedIndex;
  return Math.max(0, Math.min(planningTrace.frames.length - 1, coordinateIndex));
}

function marksForPlanningFrame(marks: SpatialMark[], frame?: PlanningTraceFrame) {
  const renderMode = planningFrameRenderMode(frame);
  if (renderMode === "perception") return marks.filter((mark) => mark.role === "perception");
  if (renderMode === "trajectory") return marks.filter((mark) => mark.role === "trajectory");
  return [];
}

function planningLabel(mark: SpatialMark) {
  return `${mark.label || ""} ${mark.detail || ""}`.toLowerCase();
}

function derivedTrajectoryMarksFromPerception(finalMarks: SpatialMark[], imageSize: ImageSize) {
  const perceptionBoxes = finalMarks.filter((mark) => mark.role === "perception" && mark.kind === "bbox");
  if (perceptionBoxes.length < 2) return [];
  const drill =
    perceptionBoxes.find((mark) => /drill|decker|cordless|power|tool/.test(planningLabel(mark))) ||
    perceptionBoxes.find((mark) => !/box|bin|crate|basket|container|yellow|green/.test(planningLabel(mark)));
  const container =
    perceptionBoxes.find((mark) => /box|bin|crate|basket|container|yellow|green/.test(planningLabel(mark))) ||
    perceptionBoxes.find((mark) => mark !== drill);
  if (!drill || !container) return [];
  const side = Math.max(24, Math.min(imageSize.width, imageSize.height) * 0.055);
  const liftY = Math.max(side, Math.min(drill.centerY, container.y) - imageSize.height * 0.1);
  const releaseY = Math.max(container.y + side, Math.min(container.y + container.height - side / 2, container.centerY));
  const points = [
    {
      label: "approach drill handle",
      x: drill.centerX,
      y: drill.centerY,
      detail: "Derived fallback from the detected drill box because no usable Grasp point_2d waypoints were returned."
    },
    {
      label: "grasp drill handle",
      x: drill.centerX,
      y: drill.centerY,
      detail: "Close around the visible drill handle."
    },
    {
      label: "lift clear",
      x: drill.centerX,
      y: liftY,
      detail: "Lift above the work surface before transfer."
    },
    {
      label: "move over yellow box",
      x: container.centerX,
      y: container.y + container.height * 0.28,
      detail: "Move toward the detected container opening."
    },
    {
      label: "release into box",
      x: container.centerX,
      y: releaseY,
      detail: "Release inside the detected container."
    }
  ];
  return points.map((point, index) => ({
    id: `final-derived-trajectory-${index}`,
    kind: "point" as const,
    role: "trajectory" as const,
    sourceKey: "derived point_2d",
    coordinateMode: "pixel" as const,
    source: "final" as const,
    x: Math.max(0, Math.min(imageSize.width - side, point.x - side / 2)),
    y: Math.max(0, Math.min(imageSize.height - side, point.y - side / 2)),
    width: side,
    height: side,
    centerX: Math.max(0, Math.min(imageSize.width, point.x)),
    centerY: Math.max(0, Math.min(imageSize.height, point.y)),
    label: point.label,
    detail: point.detail,
    time: "00:02.00",
    sequence: index + 1
  }));
}

function trajectoryMarksForPlanning(marks: SpatialMark[], imageSize: ImageSize | null) {
  const trajectory = marks.filter((mark) => mark.role === "trajectory");
  const trajectoryPoints = trajectory.filter((mark) => mark.kind === "point");
  const usableWaypoints = trajectoryPoints.filter((mark) => {
    const text = `${mark.sourceKey} ${mark.label} ${mark.detail || ""}`.toLowerCase();
    return (
      text.includes("point") ||
      /waypoint|trajectory|end.?effector|approach|grasp|lift|move|release|place|handle|path/.test(text)
    );
  });
  if (usableWaypoints.length >= 2) return usableWaypoints;
  const fallback = imageSize ? derivedTrajectoryMarksFromPerception(marks, imageSize) : [];
  return fallback.length > 0 ? fallback : trajectory;
}

function cinematicMarksForPlanningFrame(
  marks: SpatialMark[],
  frame: PlanningTraceFrame | undefined,
  frameIndex: number,
  imageSize: ImageSize | null
) {
  const renderMode = planningFrameRenderMode(frame);
  if (renderMode === "perception") return marks.filter((mark) => mark.role === "perception");
  const trajectory = trajectoryMarksForPlanning(marks, imageSize);
  if (renderMode === "trajectory") return trajectory;
  if (!frame || frameIndex <= 0 || trajectory.length === 0) return [];
  const phase = frame.phase.toLowerCase();
  const pointMarks = trajectory.filter((mark) => mark.kind === "point");
  const otherMarks = trajectory.filter((mark) => mark.kind !== "point");
  const pointCount = phase.includes("approach")
    ? 1
    : phase.includes("transfer")
      ? Math.max(3, Math.ceil(pointMarks.length * 0.75))
      : phase.includes("place")
        ? pointMarks.length
        : 0;
  return pointCount > 0 ? [...otherMarks, ...pointMarks.slice(0, pointCount)] : [];
}

function spatialMarkDisplayLabel(mark: SpatialMark, compact = false) {
  const label = (mark.label || (mark.role === "perception" ? "object" : "waypoint")).trim();
  const maxLength = compact ? 18 : 34;
  if (label.length <= maxLength) return label;
  return `${label.slice(0, maxLength - 1).trim()}…`;
}

function isConnectedTrajectoryPoint(mark: SpatialMark) {
  if (mark.kind !== "point" || mark.role !== "trajectory") return false;
  const text = `${mark.label} ${mark.sourceKey} ${mark.detail || ""}`.toLowerCase();
  return /gripper|end.?effector|trajectory|waypoint|path|approach|grasp|lift|move|release|place|start/.test(text);
}

function SpatialMarksSvg({
  className = "",
  compact = false,
  imageSize,
  marks,
  showPath = true
}: {
  className?: string;
  compact?: boolean;
  imageSize: ImageSize;
  marks: SpatialMark[];
  showPath?: boolean;
}) {
  const sortedMarks = [...marks].sort((a, b) => {
    if (a.source !== b.source) return a.source === "thinking" ? -1 : 1;
    return a.sequence - b.sequence;
  });
  const thinkingPoints = sortedMarks.filter((mark) => mark.source === "thinking" && isConnectedTrajectoryPoint(mark));
  const finalPoints = sortedMarks.filter((mark) => mark.source === "final" && isConnectedTrajectoryPoint(mark));
  return (
    <svg
      aria-hidden="true"
      className={`spatialSvg${compact ? " spatialSvgCompact" : ""}${className ? ` ${className}` : ""}`}
      preserveAspectRatio="xMidYMid meet"
      viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
    >
      {showPath && thinkingPoints.length > 1 ? (
        <polyline
          className="spatialPath spatialPathThinking"
          points={thinkingPoints.map((point) => `${point.centerX},${point.centerY}`).join(" ")}
        />
      ) : null}
      {showPath && finalPoints.length > 1 ? (
        <polyline
          className="spatialPath spatialPathFinal"
          points={finalPoints.map((point) => `${point.centerX},${point.centerY}`).join(" ")}
        />
      ) : null}
      {sortedMarks.map((mark) => {
        const labelY = mark.kind === "bbox" ? Math.max(20, mark.y + 24) : Math.max(18, mark.y - 8);
        const semanticClass =
          mark.kind === "point"
            ? isConnectedTrajectoryPoint(mark)
              ? "spatialTrajectoryMark"
              : "spatialObjectMark"
            : mark.role === "perception"
              ? "spatialObjectMark"
              : "spatialTrajectoryMark";
        return (
          <g key={mark.id}>
            <title>
              {mark.source === "thinking" ? "Reasoning trace" : "Final answer"} #{mark.sequence}: {mark.label}
              {mark.detail ? ` — ${mark.detail}` : ""}
            </title>
            <rect
              className={`${mark.kind === "point" ? "spatialPointBox" : "spatialBbox"} ${
                mark.source === "thinking" ? "spatialThinkingMark" : "spatialFinalMark"
              } ${semanticClass}`}
              height={mark.height}
              rx={Math.max(4, Math.min(mark.width, mark.height) * 0.08)}
              width={mark.width}
              x={mark.x}
              y={mark.y}
            />
            {mark.kind === "point" ? (
              <circle
                className={`spatialPointDot ${mark.source === "thinking" ? "spatialThinkingDot" : "spatialFinalDot"} ${
                  isConnectedTrajectoryPoint(mark) ? "spatialTrajectoryDot" : "spatialObjectDot"
                }`}
                cx={mark.centerX}
                cy={mark.centerY}
                r={compact ? 12 : 5}
              />
            ) : null}
            <text
              className={`spatialPointLabel ${mark.source === "thinking" ? "spatialTraceLabel" : "spatialFinalLabel"} ${
                isConnectedTrajectoryPoint(mark) ? "spatialTrajectoryLabel" : "spatialObjectLabel"
              }`}
              x={mark.x + 8}
              y={labelY}
            >
              {spatialMarkDisplayLabel(mark, compact)}
            </text>
          </g>
        );
      })}
    </svg>
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
  const [showFinalAnswer, setShowFinalAnswer] = useState(true);
  const [showReasoningTrace, setShowReasoningTrace] = useState(false);
  const [selectedPlanningFrame, setSelectedPlanningFrame] = useState(0);
  const spatialImageRef = useRef<HTMLImageElement | null>(null);
  const planningTrace = media?.planningTrace;
  const selectedFrameIndex = planningTrace
    ? Math.max(0, Math.min(planningTrace.frames.length - 1, selectedPlanningFrame))
    : 0;
  const selectedFrame = planningTrace?.frames[selectedFrameIndex];
  const selectedFrameMode = planningFrameRenderMode(selectedFrame);
  const selectedCoordinateFrameIndex = planningFrameCoordinateIndex(planningTrace, selectedFrameIndex);
  const selectedCoordinateFrame = planningTrace?.frames[selectedCoordinateFrameIndex];
  const imageSource =
    media?.kind === "image"
      ? media.previewUrl
      : planningTrace
        ? selectedFrameMode === "raw"
          ? selectedFrame?.imageUrl
          : selectedCoordinateFrame?.imageUrl
        : undefined;

  function updateImageSizeFromElement(image: HTMLImageElement | null) {
    if (!image) return;
    const width = image.naturalWidth || image.clientWidth;
    const height = image.naturalHeight || image.clientHeight;
    if (width > 0 && height > 0) setImageSize({ width, height });
  }

  useEffect(() => {
    const hasReasoningTrace = hasSpatialPayload(reasoningText);
    setShowReasoningTrace(hasReasoningTrace);
    setShowFinalAnswer(!hasReasoningTrace);
    setSelectedPlanningFrame(planningTrace?.defaultFrameIndex ?? 0);
  }, [answerText, media?.previewUrl, planningTrace?.title, reasoningText]);

  useEffect(() => {
    setImageSize(null);
    const image = spatialImageRef.current;
    if (image?.complete) updateImageSizeFromElement(image);
  }, [imageSource]);

  const canParseSpatial = useMemo(
    () => hasSpatialPayload(reasoningText) || hasSpatialPayload(answerText),
    [answerText, reasoningText]
  );
  const parsedMarks = useMemo(() => {
    if (!imageSize) return [];
    return [
      ...parseSpatialMarks(reasoningText, imageSize, "thinking"),
      ...parseSpatialMarks(answerText, imageSize, "final")
    ];
  }, [answerText, imageSize, reasoningText]);
  const parsedThinkingMarks = parsedMarks.filter((mark) => mark.source === "thinking");
  const parsedFinalMarks = parsedMarks.filter((mark) => mark.source === "final");
  const derivedFinalTrajectoryMarks = useMemo(() => {
    if (!planningTrace || !imageSize) return [];
    const hasUsableTrajectoryPoints =
      parsedFinalMarks.filter((mark) => {
        if (mark.role !== "trajectory" || mark.kind !== "point") return false;
        const text = `${mark.sourceKey} ${mark.label} ${mark.detail || ""}`.toLowerCase();
        return (
          text.includes("point") ||
          /waypoint|trajectory|end.?effector|approach|grasp|lift|move|release|place|handle|path/.test(text)
        );
      }).length >= 2;
    return hasUsableTrajectoryPoints ? [] : derivedTrajectoryMarksFromPerception(parsedFinalMarks, imageSize);
  }, [imageSize, parsedFinalMarks, planningTrace]);
  const thinkingMarks = parsedThinkingMarks;
  const finalMarks = [...parsedFinalMarks, ...derivedFinalTrajectoryMarks];
  const marks = [
    ...(showReasoningTrace ? thinkingMarks : []),
    ...(showFinalAnswer ? finalMarks : [])
  ];
  const visibleMarks = planningTrace ? cinematicMarksForPlanningFrame(marks, selectedFrame, selectedFrameIndex, imageSize) : marks;
  const points = visibleMarks.filter((mark) => mark.kind === "point");
  const boxes = visibleMarks.filter((mark) => mark.kind === "bbox");
  const visibleThinkingMarks = planningTrace
    ? showReasoningTrace
      ? cinematicMarksForPlanningFrame(thinkingMarks, selectedFrame, selectedFrameIndex, imageSize)
      : []
    : showReasoningTrace
      ? thinkingMarks
      : [];
  const visibleFinalMarks = showFinalAnswer
    ? planningTrace
      ? cinematicMarksForPlanningFrame(finalMarks, selectedFrame, selectedFrameIndex, imageSize)
      : finalMarks
    : [];
  const selectedOverlayLabel =
    selectedFrameMode === "perception"
      ? "Perceive boxes"
      : selectedFrameMode === "trajectory"
        ? "Grasp trajectory"
        : visibleMarks.length > 0
          ? "Plan progression"
          : "Raw execution frame";
  const traceState = showReasoningTrace ? "shown" : "hidden";
  const finalState = showFinalAnswer ? "shown" : "hidden";
  const cardTitle = planningTrace ? planningTrace.title : "Final path in image coordinate space";
  const cardSubtitle = planningTrace?.subtitle || (planningTrace ? "Select a keyframe to inspect the overlaid plan." : "");

  if (!media || !imageSource || !canParseSpatial) return null;

  return (
    <article className="spatialOverlayCard">
      <div className="spatialTopline">
        <div>
          <p className="responseLabel">{planningTrace ? "Robot planning trace" : "Spatial trajectory"}</p>
          <h3>{cardTitle}</h3>
          {cardSubtitle ? <p className="spatialSubtitle">{cardSubtitle}</p> : null}
          </div>
          <div className="spatialToolbar">
            <span>
              {planningTrace ? `${selectedOverlayLabel} · ` : ""}
              {finalMarks.length} final {finalState} · {thinkingMarks.length} trace {traceState} · {points.length} points · {boxes.length} boxes shown
            </span>
          {finalMarks.length > 0 ? (
            <button
              aria-checked={showFinalAnswer}
              className={showFinalAnswer ? "spatialTraceToggle spatialFinalToggle checked" : "spatialTraceToggle spatialFinalToggle"}
              onClick={() =>
                setShowFinalAnswer((value) => {
                  const nextValue = !value;
                  if (nextValue) setShowReasoningTrace(false);
                  return nextValue;
                })
              }
              role="switch"
              type="button"
            >
              <span />
              {showFinalAnswer ? "Final answer on" : "Final answer off"}
            </button>
          ) : null}
          {thinkingMarks.length > 0 ? (
            <button
              aria-checked={showReasoningTrace}
              className={showReasoningTrace ? "spatialTraceToggle checked" : "spatialTraceToggle"}
              onClick={() =>
                setShowReasoningTrace((value) => {
                  const nextValue = !value;
                  if (nextValue) setShowFinalAnswer(false);
                  return nextValue;
                })
              }
              role="switch"
              type="button"
            >
              <span />
              {showReasoningTrace ? "Reasoning trace on" : "Reasoning trace off"}
            </button>
          ) : null}
        </div>
      </div>
      {thinkingMarks.length > 0 || finalMarks.length > 0 ? (
        <div className="spatialLegend" aria-label="Spatial overlay legend">
          {finalMarks.length > 0 ? (
            <button
              aria-checked={showFinalAnswer}
              className={showFinalAnswer ? "spatialLegendToggle finalLegend" : "spatialLegendToggle finalLegend mutedLegend"}
              onClick={() =>
                setShowFinalAnswer((value) => {
                  const nextValue = !value;
                  if (nextValue) setShowReasoningTrace(false);
                  return nextValue;
                })
              }
              role="switch"
              type="button"
            >
              {showFinalAnswer ? "Final answer on" : "Final answer off"}
            </button>
          ) : null}
          {thinkingMarks.length > 0 ? (
            <button
              aria-checked={showReasoningTrace}
              className={showReasoningTrace ? "spatialLegendToggle traceLegend" : "spatialLegendToggle traceLegend mutedLegend"}
              onClick={() =>
                setShowReasoningTrace((value) => {
                  const nextValue = !value;
                  if (nextValue) setShowFinalAnswer(false);
                  return nextValue;
                })
              }
              role="switch"
              type="button"
            >
              {showReasoningTrace ? "Reasoning trace on" : "Reasoning trace off"}
            </button>
          ) : null}
        </div>
      ) : null}
      {planningTrace ? (
        <div className="planningFilmstrip" aria-label="Robot planning trace keyframes">
          {planningTrace.frames.map((frame, index) => {
            const selected = index === selectedFrameIndex;
            const frameMode = planningFrameRenderMode(frame);
            const frameMarks = imageSize ? cinematicMarksForPlanningFrame(marks, frame, index, imageSize) : [];
            const hasFramePath = frameMarks.filter((mark) => mark.kind === "point").length > 1;
            const frameImageUrl = frame.imageUrl;
            const frameBadge =
              frameMode === "perception"
                ? "boxes"
                : frameMode === "trajectory"
                  ? "trajectory"
                  : frameMarks.length > 0
                    ? "trace"
                    : "";
            return (
              <button
                aria-pressed={selected}
                className={selected ? "planningFrame active" : "planningFrame"}
                key={`${frame.time}-${frame.phase}`}
                onClick={() => setSelectedPlanningFrame(index)}
                type="button"
              >
                <div className="planningFrameMedia">
                  <img src={frameImageUrl} alt="" />
                  {imageSize && frameMarks.length > 0 ? (
                    <SpatialMarksSvg
                      className="planningFrameOverlay"
                      compact
                      imageSize={imageSize}
                      marks={frameMarks}
                      showPath={hasFramePath}
                    />
                  ) : null}
                </div>
                <span>{frame.time}</span>
                <strong>{frame.phase}</strong>
                {frameBadge ? <em>{frameBadge}</em> : null}
                {frame.caption ? <small>{frame.caption}</small> : null}
              </button>
            );
          })}
        </div>
      ) : null}
      <div className="spatialCanvas">
        <img
          className="spatialImage"
          ref={spatialImageRef}
          src={imageSource}
          alt={
            selectedCoordinateFrame
              ? `${media.name} ${selectedCoordinateFrame.phase} coordinate frame with generated point and box overlay`
              : `${media.name} with generated point and box overlay`
          }
          onLoad={(event) => updateImageSizeFromElement(event.currentTarget)}
        />
        {imageSize && visibleMarks.length > 0 ? (
          <SpatialMarksSvg imageSize={imageSize} marks={visibleMarks} showPath={selectedFrameMode === "trajectory" || !planningTrace} />
        ) : null}
      </div>
      {selectedCoordinateFrame ? (
        <p className="planningCoordinateNote">
          <strong>
            Coordinate frame: {selectedCoordinateFrame.phase} · {selectedCoordinateFrame.time}
          </strong>
          {selectedFrameMode === "perception"
            ? "Perceive renders object boxes on this frame."
            : selectedFrameMode === "trajectory"
              ? "Grasp renders the planned 2D trajectory in this frame's coordinate space."
              : visibleMarks.length > 0
                ? "This cinematic execution tile projects the plan progression onto the step frame; exact trajectory coordinates are measured on the Grasp frame."
                : "This execution step is shown as a raw frame without overlays."}
        </p>
      ) : null}
      {selectedFrame?.caption ? (
        <p className="planningFrameCaption">
          <strong>
            {selectedFrame.phase} · {selectedFrame.time}
          </strong>
          {selectedFrame.caption}
          <span>{selectedOverlayLabel}</span>
        </p>
      ) : null}
      {visibleMarks.length > 0 ? (
        <div className="spatialSequenceGroups">
          <SpatialMarkList label="Final answer" marks={visibleFinalMarks} />
          {showReasoningTrace && visibleThinkingMarks.length > 0 ? (
            <SpatialMarkList label="Reasoning trace" marks={visibleThinkingMarks} />
          ) : null}
        </div>
      ) : planningTrace && marks.length > 0 ? (
        <p className="spatialLoading">This execution step is shown as a raw frame. Select Perceive for boxes or Grasp for the planned 2D path.</p>
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
            {mark.time ? <small>{mark.time}</small> : null}
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

function PlanningStagePromptInspector({ result }: { result: ApiResult | null }) {
  const stages = useMemo(() => planningStagePromptViews(result), [result]);
  if (stages.length === 0) return null;

  return (
    <details className="stagePromptInspector">
      <summary>
        <span>Stage prompts</span>
        <em>{stages.length} NIM calls</em>
      </summary>
      <p>
        These are the exact per-step text prompts sent to NIM. Image data URLs are redacted, but the
        image count and coordinate-frame timestamp are preserved.
      </p>
      <div className="stagePromptGrid">
        {stages.map((stage) => (
          <section className="stagePromptCard" key={stage.key}>
            <div className="stagePromptHeader">
              <div>
                <strong>{stage.title}</strong>
                <span>{stage.mode}</span>
              </div>
              <code>{stage.time}</code>
            </div>
            <div className="stagePromptMeta" aria-label={`${stage.title} request metadata`}>
              <span>{stage.imageCount} image</span>
              {stage.elapsed ? <span>{stage.elapsed}</span> : null}
            </div>
            {stage.userPrompt ? (
              <>
                <p className="stagePromptLabel">User prompt</p>
                <pre>{stage.userPrompt}</pre>
              </>
            ) : null}
            {stage.systemPrompt ? (
              <details className="stageSystemPromptDetails">
                <summary>System prompt</summary>
                <pre>{stage.systemPrompt}</pre>
              </details>
            ) : null}
          </section>
        ))}
      </div>
    </details>
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
  streamCreated,
  streamLogs,
  streamPhase
}: {
  isRunning: boolean;
  longProgress: LongProgressState | null;
  media: MediaState | null;
  parsedOutput: { reasoning: string; answer: string; steps: string[] };
  reasoningExpanded: boolean;
  result: ApiResult | null;
  setReasoningExpanded: (expanded: boolean) => void;
  streamCreated: number;
  streamLogs: StreamLogEntry[];
  streamPhase: StreamPhase;
}) {
  const progressCard = longProgress ? <LongVideoProgress progress={longProgress} /> : null;
  const standardProgressCard =
    isRunning && !longProgress ? (
      <StandardRunProgress created={streamCreated} logs={streamLogs} media={media} phase={streamPhase} />
    ) : null;

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

  const resultText = resultContentText(result);
  if (result?.content || result?.reasoning || resultText) {
    const hasAnswer = Boolean(parsedOutput.answer || result?.content || resultText);
    const answerText = parsedOutput.answer || result?.content || resultText || "";
    const timelineItems = stitchedTimelineItems(result);
    const isLongVideoResult = Boolean(result?.long_video || longProgress);
    const isSpatialResult = hasSpatialPayload(answerText) || hasSpatialPayload(parsedOutput.reasoning);
    const collapseResponse = (isLongVideoResult || isSpatialResult) && hasAnswer;
    return (
      <div className="responseStack">
        {progressCard}
        {standardProgressCard}
        {timelineItems.length > 0 ? <StitchedTimeline items={timelineItems} /> : null}
        <SpatialTrajectoryOverlay answerText={answerText} media={media} reasoningText={parsedOutput.reasoning} />
        <PlanningStagePromptInspector result={result} />
        {hasAnswer && collapseResponse ? (
          <details className="answer rawResponseDetails">
            <summary>
              {isLongVideoResult
                ? looksLikeJsonResponse(answerText)
                  ? "Stitched JSON response"
                  : "Raw stitched response"
                : looksLikeJsonResponse(answerText)
                  ? "Spatial JSON response"
                  : "Raw spatial response"}
            </summary>
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
        {progressCard ? null : standardProgressCard || <GeneratingOutput />}
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

function DynamicTimelineList({ empty, items }: { empty: string; items: TimelineItem[] }) {
  if (items.length === 0) return <p className="timelineEmpty">{empty}</p>;
  return (
    <ol className="dynamicTimelineList">
      {items.map((item, index) => {
        const isEndpoint = index === 0 || index === items.length - 1;
        return (
          <li className={isEndpoint ? "endpoint" : "middle"} key={item.id || index}>
            <span className="dynamicTimelineRail" aria-hidden="true">
              <span className={isEndpoint ? "dynamicTimelineDot" : "dynamicTimelineTick"} />
            </span>
            <div>
              <strong>{item.range}</strong>
              <span>
                {item.title ? <em>{item.title}</em> : null}
                {item.caption}
              </span>
              {item.meta ? <small>{item.meta}</small> : null}
            </div>
          </li>
        );
      })}
    </ol>
  );
}

function SummaryTimelineList({ empty, items }: { empty: string; items: TimelineItem[] }) {
  if (items.length === 0) return <p className="timelineEmpty">{empty}</p>;
  return (
    <div className="summaryTimelineList">
      {items.map((item, index) => {
        const range = rangeParts(item.range);
        return (
          <article className="summaryTimelineItem" key={item.id || index}>
            <div className="summaryTimelineRange">
              <span>
                <small>Start</small>
                {range.start}
              </span>
              <span>
                <small>End</small>
                {range.end}
              </span>
            </div>
            <div className="summaryTimelineBody">
              <strong>Summary</strong>
              <p>{item.caption}</p>
              {item.meta ? <small>{item.meta}</small> : null}
            </div>
          </article>
        );
      })}
    </div>
  );
}

function TimelineOverviewStrip({ items }: { items: TimelineItem[] }) {
  if (items.length === 0) return null;
  const firstRange = rangeParts(items[0].range);
  const lastRange = rangeParts(items[items.length - 1].range);
  return (
    <section className="timelineOverviewStrip" aria-label="Stitched timeline overview">
      <div className="timelineOverviewHeader">
        <span>Timeline overview</span>
        <small>
          {firstRange.start} - {lastRange.end} · {items.length} shown
        </small>
      </div>
      <div className="timelineOverviewRail">
        {items.map((item, index) => {
          const isEndpoint = index === 0 || index === items.length - 1;
          const range = rangeParts(item.range);
          return (
            <div
              className={`timelineOverviewEvent${isStaticTimelineItem(item) ? " static" : " dynamic"}${
                isEndpoint ? " endpoint" : ""
              }`}
              key={item.id || index}
              title={`${item.range} ${item.title} ${item.caption}`}
            >
              <span className="timelineOverviewMarker" />
              <strong>{range.start}</strong>
              <em>{item.title || "event"}</em>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function StitchedTimeline({ items }: { items: TimelineItem[] }) {
  const timelineSignature = useMemo(
    () => items.map((item) => `${item.range}:${item.title}:${item.caption}`).join("|"),
    [items]
  );
  const dynamicItems = items.filter((item) => !isStaticTimelineItem(item));
  const staticItems = items.filter(isStaticTimelineItem);
  const [showDynamic, setShowDynamic] = useState(true);
  const [showStatic, setShowStatic] = useState(false);
  const [showDynamicDetails, setShowDynamicDetails] = useState(false);
  useEffect(() => {
    setShowDynamic(true);
    setShowStatic(false);
    setShowDynamicDetails(false);
  }, [timelineSignature]);
  const visibleItems = items.filter((item) => (isStaticTimelineItem(item) ? showStatic : showDynamic));
  const summaryItems = dynamicItems.filter(isSummaryTimelineItem);
  const dynamicEventItems = dynamicItems.filter((item) => !isSummaryTimelineItem(item));
  const summaryFirst = showDynamic && !showStatic && summaryItems.length > 0;
  const overviewItems = summaryFirst ? summaryItems : visibleItems;
  const title =
    summaryFirst
      ? "Summary timeline"
      : showDynamic && showStatic
        ? "Filtered timeline"
        : showStatic
          ? "Static timeline"
          : showDynamic
            ? "Dynamic timeline"
            : "Timeline filters";
  return (
    <article className="stitchedTimelineCard">
      <div className="stitchedTimelineHeader">
        <div>
          <p className="responseLabel">Complete Stitched Response</p>
          <h3>{title}</h3>
        </div>
        <div className="stitchedTimelineActions">
          <span>{visibleItems.length}/{items.length} shown</span>
          <button
            aria-pressed={showDynamic}
            className="stitchedTimelineToggle"
            disabled={dynamicItems.length === 0}
            onClick={() => setShowDynamic((value) => !value)}
            type="button"
          >
            Dynamic <small>{dynamicItems.length}</small>
          </button>
          <button
            aria-pressed={showStatic}
            className="stitchedTimelineToggle"
            disabled={staticItems.length === 0}
            onClick={() => setShowStatic((value) => !value)}
            type="button"
          >
            Static <small>{staticItems.length}</small>
          </button>
        </div>
      </div>
      {overviewItems.length > 0 ? <TimelineOverviewStrip items={overviewItems} /> : null}
      {summaryFirst ? (
        <>
          <SummaryTimelineList empty="No summary events were returned." items={summaryItems} />
          {dynamicEventItems.length > 0 ? (
            <button
              aria-expanded={showDynamicDetails}
              className="summaryDetailsToggle"
              onClick={() => setShowDynamicDetails((value) => !value)}
              type="button"
            >
              {showDynamicDetails ? "Hide dynamic events" : `Show ${dynamicEventItems.length} dynamic events`}
            </button>
          ) : null}
          {showDynamicDetails ? (
            <DynamicTimelineList empty="No dynamic events were returned." items={dynamicEventItems} />
          ) : null}
        </>
      ) : visibleItems.length > 0 ? (
        <DynamicTimelineList empty="No events match the current timeline filters." items={visibleItems} />
      ) : (
        <TimelineList empty="Turn on Dynamic or Static to show stitched events." items={visibleItems} />
      )}
    </article>
  );
}

function LiveLongTimeline({
  collapsed,
  items,
  setCollapsed,
  setShowEvents,
  showEvents
}: {
  collapsed: boolean;
  items: TimelineItem[];
  setCollapsed: (value: boolean | ((current: boolean) => boolean)) => void;
  setShowEvents: (value: boolean | ((current: boolean) => boolean)) => void;
  showEvents: boolean;
}) {
  return (
    <article className={`liveTimelineCard${collapsed ? " collapsed" : ""}`}>
      <div className="liveTimelineHeader">
        <div>
          <p className="responseLabel">Live Timeline</p>
          <h3>{showEvents ? "Parsed events" : "Partial timeline"}</h3>
        </div>
        <div className="liveTimelineActions">
          <button
            aria-expanded={!collapsed}
            className="liveTimelineToggle"
            onClick={() => setCollapsed((value) => !value)}
            type="button"
          >
            {collapsed ? "Show timeline" : "Hide timeline"}
          </button>
          {!collapsed ? (
            <button className="liveTimelineMode" onClick={() => setShowEvents((value) => !value)} type="button">
              {showEvents ? "Show partial timeline" : "Show events"}
            </button>
          ) : null}
        </div>
      </div>
      {!collapsed ? (
        <TimelineList
          empty={showEvents ? "No parsed events have arrived yet." : "No chunk summaries have arrived yet."}
          items={items}
        />
      ) : null}
    </article>
  );
}

function LiveTimelineDisclosure({
  items,
  setShowEvents,
  showEvents
}: {
  items: TimelineItem[];
  setShowEvents: (value: boolean | ((current: boolean) => boolean)) => void;
  showEvents: boolean;
}) {
  return (
    <details className="completedChunksDisclosure liveTimelineDisclosure">
      <summary>
        <span>{showEvents ? "Live parsed events" : "Partial timeline"}</span>
        <small>{items.length}</small>
      </summary>
      <div className="liveTimelineInline">
        <button className="liveTimelineMode" onClick={() => setShowEvents((value) => !value)} type="button">
          {showEvents ? "Show partial timeline" : "Show events"}
        </button>
        <TimelineList
          empty={showEvents ? "No parsed events were returned." : "No chunk summaries were returned."}
          items={items}
        />
      </div>
    </details>
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
  const completedChunks = progress.chunks.filter((chunk) => chunk.status === "done");
  const activeChunks = progress.chunks.filter((chunk) => chunk.status !== "done");
  const renderChunkButton = (chunk: LongChunkProgress, collapseDone = false) => {
    const selected = selectedChunk?.index === chunk.index;
    const collapsedDone = collapseDone && chunk.status === "done" && !selected;
    return (
      <button
        aria-pressed={selected}
        className={`longChunkPill ${chunk.status}${selected ? " selected" : ""}${
          collapsedDone ? " collapsedDone" : ""
        }`}
        key={chunk.index}
        onClick={() => setSelectedIndex((current) => (current === chunk.index ? null : chunk.index))}
        title={chunk.error || chunk.summary || chunk.timeRange}
        type="button"
      >
        {collapsedDone ? (
          <div className="longChunkCompact">
            <strong>{chunk.index + 1}</strong>
            <span>{chunk.status}</span>
            <small>{chunk.timeRange}</small>
          </div>
        ) : (
          <>
            <div className="longChunkThumb" aria-hidden="true">
              {chunk.thumbnailUrl ? <img src={chunk.thumbnailUrl} alt="" loading="lazy" /> : <span />}
              <div className="longChunkOverlay">
                <strong>{chunk.index + 1}</strong>
                <span>{chunk.status}</span>
              </div>
            </div>
            <small>{chunk.timeRange}</small>
          </>
        )}
        <span className="longChunkProgress" aria-label={`Chunk ${chunk.index + 1} ${chunk.status}`}>
          <span style={{ width: `${chunkProgressPercent(chunk.status)}%` }} />
        </span>
      </button>
    );
  };
  useEffect(() => {
    if (selectedIndex !== null && !progress.chunks.some((chunk) => chunk.index === selectedIndex)) {
      setSelectedIndex(null);
    }
  }, [progress.chunks, selectedIndex]);
  const hasTimeline = progress.partialTimeline.length > 0 || progress.chunks.some((chunk) => eventsFromChunk(chunk).length > 0);
  const chunksFinished = total > 0 && completed + failed >= total && running === 0;
  const complete = phase === "complete" || (chunksFinished && percent >= 100);
  const compactComplete = compact && complete;
  useEffect(() => {
    if (complete) setCompact(true);
  }, [complete]);
  return (
    <>
      <article className={`longProgressCard${compact ? " compact" : ""}${compactComplete ? " compactComplete" : ""}`} aria-live="polite">
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
        {!compactComplete ? (
          <div className="longProgressMeta">
            <span>Phase: {phase.replace(/_/g, " ")}</span>
            <span>Chunks: {completed + failed}/{total || "?"}</span>
            <span>Running: {running}</span>
            <span>Elapsed: {formatDuration(progress.elapsedSeconds)}</span>
            <span>ETA: {progress.etaSeconds === 0 ? "complete" : formatDuration(progress.etaSeconds)}</span>
          </div>
        ) : null}
        <div className="longProgressTrack" aria-label={`Long video progress ${percent}%`}>
          <span style={{ width: `${percent}%` }} />
        </div>
        {!compactComplete && progress.steps?.length ? <LongStepList steps={progress.steps} /> : null}
        {!compactComplete ? (
          <div className="longCoverageGrid">
            <span>Duration: {progress.durationText || formatDuration(progress.durationSeconds)}</span>
            <span>Frames: {progress.frameCount ?? "scanning"}</span>
            <span>Coverage: {progress.sampleFps ? `${progress.sampleFps} fps` : "planning"}</span>
            <span>Requested FPS: {progress.requestedFps ? progress.requestedFps : "preset"}</span>
            {progress.frameLimit ? <span>Frame budget: {progress.frameLimit}</span> : null}
            <span>Concurrency: {progress.concurrency || 4}</span>
          </div>
        ) : null}
        {!compact ? (
          <>
            {progress.warnings.length > 0 ? (
              <ul className="longWarnings">
                {progress.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            ) : null}
            {activeChunks.length > 0 ? (
              <div className="longChunkGrid" aria-label="Active chunk progress">
                {activeChunks.map((chunk) => renderChunkButton(chunk))}
              </div>
            ) : null}
            {completedChunks.length > 0 ? (
              <details className="completedChunksDisclosure">
                <summary>
                  <span>Completed chunks</span>
                  <small>{completedChunks.length}/{total || completedChunks.length}</small>
                </summary>
                <div className="longChunkGrid completedChunkGrid" aria-label="Completed chunks">
                  {completedChunks.map((chunk) => renderChunkButton(chunk, true))}
                </div>
              </details>
            ) : null}
            {chunksFinished && hasTimeline ? (
              <LiveTimelineDisclosure
                items={timelineItems}
                setShowEvents={setShowTimelineEvents}
                showEvents={showTimelineEvents}
              />
            ) : null}
            {selectedChunk ? <LongChunkInspector chunk={selectedChunk} /> : null}
          </>
        ) : null}
      </article>
      {!compact && hasTimeline && !chunksFinished ? (
        <LiveLongTimeline
          collapsed={timelineCollapsed}
          items={timelineItems}
          setCollapsed={setTimelineCollapsed}
          setShowEvents={setShowTimelineEvents}
          showEvents={showTimelineEvents}
        />
      ) : null}
    </>
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

function StandardRunProgress({
  created,
  logs,
  media,
  phase
}: {
  created: number;
  logs: StreamLogEntry[];
  media: MediaState | null;
  phase: StreamPhase;
}) {
  const isPlanningRun = Boolean(media?.planningTrace);
  const percent = standardRunPercent(phase);
  const steps = standardRunSteps(phase, isPlanningRun);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (phase === "complete" || phase === "error" || phase === "stopped") return undefined;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [phase]);
  const latestElapsed = Math.max(
    0,
    ...logs.map((log) => (Number.isFinite(Number(log.elapsedSeconds)) ? Number(log.elapsedSeconds) : 0))
  );
  const runElapsed = Math.max(0, now / 1000 - created);
  const activeElapsed = Math.max(latestElapsed, runElapsed);
  return (
    <article className="standardProgressCard" aria-live="polite">
      <div className="longProgressHeader">
        <div>
          <p className="responseLabel">{isPlanningRun ? "Robot planning inference" : "Inference"}</p>
          <h3>{standardRunMessage(phase, isPlanningRun)}</h3>
          {logs.length > 0 ? <p className="standardProgressSubline">Latest backend log at {formatDuration(activeElapsed)}</p> : null}
        </div>
        <span className="longPresetBadge">{phase.replace(/_/g, " ")}</span>
      </div>
      <div className="longProgressTrack" aria-label={`Inference progress ${percent}%`}>
        <span style={{ width: `${percent}%` }} />
      </div>
      <LongStepList steps={steps} />
      {logs.length > 0 ? (
        <details className="standardLogPanel" open={isPlanningRun}>
          <summary>
            <span>Backend logs</span>
            <em>{logs.length}</em>
          </summary>
          <ol>
            {logs.map((log, index) => (
              <li className={log.level === "warn" ? "warn" : log.level === "error" ? "error" : ""} key={`${log.label}-${index}`}>
                <code>{Number.isFinite(Number(log.elapsedSeconds)) ? formatDuration(log.elapsedSeconds) : "--"}</code>
                <strong>{log.label}</strong>
                {log.detail ? <span>{log.detail}</span> : null}
              </li>
            ))}
          </ol>
        </details>
      ) : null}
    </article>
  );
}

function standardRunPercent(phase: StreamPhase) {
  if (phase === "preparing_media") return 15;
  if (phase === "waiting_first_token") return 42;
  if (phase === "reasoning") return 65;
  if (phase === "answer") return 86;
  if (phase === "complete") return 100;
  if (phase === "error" || phase === "stopped") return 100;
  return 8;
}

function standardRunMessage(phase: StreamPhase, isPlanningRun: boolean) {
  if (phase === "preparing_media") return "Preparing media";
  if (phase === "waiting_first_token") return isPlanningRun ? "Preparing frame fallback and waiting for the NIM" : "Waiting for the NIM";
  if (phase === "reasoning") return isPlanningRun ? "Perceiving objects and planning" : "Streaming reasoning";
  if (phase === "answer") return isPlanningRun ? "Receiving boxes and trajectory JSON" : "Streaming response";
  if (phase === "complete") return "Complete";
  if (phase === "error") return "Backend error";
  if (phase === "stopped") return "Stopped";
  return "Starting inference";
}

function standardRunSteps(phase: StreamPhase, isPlanningRun: boolean): LongStepProgress[] {
  const activeIndex =
    phase === "preparing_media"
      ? 0
      : phase === "waiting_first_token"
        ? 1
        : phase === "reasoning"
          ? 2
          : phase === "answer"
            ? 3
            : phase === "complete"
              ? 5
              : phase === "error" || phase === "stopped"
                ? 4
                : 0;
  const labels = isPlanningRun
    ? [
        ["media", "Prepare media", "Use native video first; fall back to timestamped image frames if the NIM rejects decode."],
        ["nim", "Call Super NIM", "Send the staged Perceive/Grasp prompt and wait for the first token."],
        ["perceive", "Perceive object boxes", "Parse bbox_2d object labels for the 00:00.00 coordinate frame."],
        ["grasp", "Plan grasp trajectory", "Parse point_2d waypoints for the Grasp-frame coordinate space."],
        ["render", "Render planning trace", "Draw boxes on Perceive and the trajectory on Grasp; execution frames stay raw."]
      ]
    : [
        ["media", "Prepare media", "Package the current image or video request."],
        ["nim", "Call NIM", "Send the request and wait for the first token."],
        ["reason", "Stream reasoning", "Receive the optional thinking trace."],
        ["answer", "Stream answer", "Receive the final response."],
        ["render", "Render output", "Format response JSON, timeline, or spatial overlays."]
      ];
  return labels.map(([key, label, detail], index) => {
    const status =
      phase === "error"
        ? index === activeIndex
          ? "error"
          : index < activeIndex
            ? "done"
            : "pending"
        : phase === "stopped"
          ? index < activeIndex
            ? "done"
            : "pending"
          : index < activeIndex
            ? "done"
            : index === activeIndex
              ? "running"
              : "pending";
    return {
      key,
      label,
      detail,
      status,
      progress: status === "done" ? 100 : status === "running" ? 65 : 0
    } as LongStepProgress;
  });
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
