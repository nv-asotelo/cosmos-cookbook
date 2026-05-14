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
import { ChangeEvent, DragEvent, RefObject, useEffect, useMemo, useRef, useState } from "react";

const HERO_IMAGE = "https://assets.ngc.nvidia.com/products/api-catalog/images/cosmos-reason2-8b.jpg";
const COSMOS3_INFO_URL =
  (typeof import.meta !== "undefined" && (import.meta as { env?: Record<string, string> }).env?.VITE_COSMOS3_INFO_URL) ||
  "/api/active-model";
const DEFAULT_MODEL =
  (typeof import.meta !== "undefined" && (import.meta as { env?: Record<string, string> }).env?.VITE_MODEL_NAME) ||
  "Detecting model...";
const DEFAULT_USER_PROMPT = "";
const DEFAULT_SYSTEM_PROMPT = "";
const DEFAULT_TEMPERATURE = 0.6;
const DEFAULT_TOP_P = 0.3;
const DEFAULT_MAX_TOKENS = 4096;
const DEFAULT_FRAMES_PER_SECOND = 6;
const DEFAULT_REPETITION_PENALTY = 1.2;
const DEFAULT_SEED = 42;
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

type SectionTab = "Experience" | "Model Card" | "System Card" | "Deploy";
type OutputTab = "preview" | "json";

type MediaState = {
  name: string;
  kind: "video" | "image";
  previewUrl: string;
  dataUrl: string;
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
  error?: string;
  payload?: unknown;
  raw?: unknown;
};

const EXAMPLES: ExampleItem[] = [
  {
    id: "race-car",
    title: "race car footage",
    mediaUrl: "https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason2/cr2_drift.mp4",
    mediaName: "race-car-footage.mp4",
    mediaKind: "video",
    userPrompt: "Describe the video. Add timestamps in mm:ss format.",
    systemPrompt: "You are a helpful assistant.",
    reasoning: true
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
    reasoning: false
  },
  {
    id: "mail-package",
    title: "mail package",
    mediaUrl: "https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason2/cr2_mail_package.mp4",
    mediaName: "mail-package.mp4",
    mediaKind: "video",
    userPrompt: "Is the person allowed to pick up the packages?",
    systemPrompt: "You are a helpful assistant.",
    reasoning: true
  },
  {
    id: "warehouse",
    title: "warehouse",
    mediaUrl: "https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason2/cr2_warehouse.mp4",
    mediaName: "warehouse.mp4",
    mediaKind: "video",
    userPrompt: "Which worker picked up the dropped box?",
    systemPrompt: "You are a helpful warehouse monitoring system.",
    reasoning: true
  },
  {
    id: "construction",
    title: "construction worker road sign",
    mediaUrl: "https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason2/cr2_construction_car.mp4",
    mediaName: "construction-worker-road-sign.mp4",
    mediaKind: "video",
    userPrompt: "What's the next immediate action for the Ego vehicle?",
    systemPrompt: "You are a helpful assistant.",
    reasoning: true
  },
  {
    id: "robot-arm",
    title: "robot arm pick up stuff",
    mediaUrl: HERO_IMAGE,
    mediaName: "robot-arm-pick-up-stuff.jpg",
    mediaKind: "image",
    userPrompt:
      'You are given the task "Move the tape into the basket". Specify the 2D trajectory your end effector should follow in pixel space. Return the trajectory coordinates in JSON format like this: {"point_2d": [x, y], "label": "gripper trajectory"}.',
    systemPrompt: "You are a helpful assistant.",
    reasoning: true
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
    reasoning: true
  }
];

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
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
  return /\.(mp4|jpg|jpeg|png)$/i.test(file.name);
}

function parseReasoning(content?: string) {
  const text = content || "";
  const match = text.match(/<think>([\s\S]*?)<\/think>/i);
  const reasoning = match?.[1]?.trim() || "";
  const answer = match ? text.slice((match.index || 0) + match[0].length).trim() : text.trim();
  const steps = reasoning
    .split(/\n{2,}/)
    .map((step) => step.trim())
    .filter(Boolean);
  return { reasoning, answer, steps };
}

function safeJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

export default function App() {
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [activeTab, setActiveTab] = useState<SectionTab>("Experience");
  const [outputTab, setOutputTab] = useState<OutputTab>("preview");
  const [examplesOpen, setExamplesOpen] = useState(false);
  const [selectedExampleId, setSelectedExampleId] = useState(EXAMPLES[0].id);
  const [parametersOpen, setParametersOpen] = useState(false);
  const [reasoningExpanded, setReasoningExpanded] = useState(true);
  const [dragActive, setDragActive] = useState(false);
  const [media, setMedia] = useState<MediaState | null>(null);
  const [userPrompt, setUserPrompt] = useState(DEFAULT_USER_PROMPT);
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT);
  const [reasoningEnabled, setReasoningEnabled] = useState(true);
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [models, setModels] = useState<string[]>([DEFAULT_MODEL]);
  const [backendInfo, setBackendInfo] = useState<{
    checkpoint?: string;
    display_name?: string;
    cosmos3_version?: string;
    backend?: string;
    gpu_name?: string;
    vram_free_gib?: number;
    vram_total_gib?: number;
    base_url?: string;
  } | null>(null);
  const [temperature, setTemperature] = useState(DEFAULT_TEMPERATURE);
  const [topP, setTopP] = useState(DEFAULT_TOP_P);
  const [maxTokens, setMaxTokens] = useState(DEFAULT_MAX_TOKENS);
  const [framesPerSecond, setFramesPerSecond] = useState(DEFAULT_FRAMES_PER_SECOND);
  const [repetitionPenalty, setRepetitionPenalty] = useState(DEFAULT_REPETITION_PENALTY);
  const [seed, setSeed] = useState(DEFAULT_SEED);
  const [status, setStatus] = useState("Ready");
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<ApiResult | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch(COSMOS3_INFO_URL, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled || !d) return;
        const name = (d.checkpoint as string | undefined) || (d.display_name as string | undefined);
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
          setModels(data.models);
          setModel(data.models[0]);
        }
      })
      .catch(() => undefined);
  }, []);

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
                    : { type: "video_url", video_url: { url: "data:video/mp4;base64,<payload>" } }
                ]
              : []),
            { type: "text", text: effectivePrompt }
          ]
        }
      ],
      temperature,
      top_p: topP,
      max_tokens: maxTokens,
      repetition_penalty: repetitionPenalty,
      frames_per_second: framesPerSecond,
      seed
    }),
    [
      effectivePrompt,
      framesPerSecond,
      maxTokens,
      media,
      model,
      repetitionPenalty,
      seed,
      systemPrompt,
      temperature,
      topP
    ]
  );

  const parsedOutput = useMemo(() => parseReasoning(result?.content), [result?.content]);
  const jsonOutput = useMemo(
    () => ({
      status,
      model,
      reasoning: parsedOutput.reasoning || null,
      response: parsedOutput.answer || null,
      request: requestPreview,
      raw: result?.raw || result || null
    }),
    [model, parsedOutput.answer, parsedOutput.reasoning, requestPreview, result, status]
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
    setOutputTab("preview");
    setStatus("Media loaded");
  }

  async function handleFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) await setFileMedia(file);
  }

  async function handleDrop(event: DragEvent<HTMLButtonElement>) {
    event.preventDefault();
    setDragActive(false);
    const file = event.dataTransfer.files?.[0];
    if (file) await setFileMedia(file);
  }

  function handleDrag(event: DragEvent<HTMLButtonElement>, active: boolean) {
    event.preventDefault();
    setDragActive(active);
  }

  async function mediaFromExample(example: ExampleItem): Promise<MediaState> {
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

  async function applyExample() {
    const example = EXAMPLES.find((item) => item.id === selectedExampleId) || EXAMPLES[0];
    setStatus("Loading example");
    try {
      setMedia(await mediaFromExample(example));
      setReasoningEnabled(example.reasoning);
      setUserPrompt(promptForReasoning(example.userPrompt, example.reasoning));
      setSystemPrompt(example.systemPrompt);
      setResult(null);
      setOutputTab("preview");
      setExamplesOpen(false);
      setStatus("Example loaded");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Example failed to load");
    }
  }

  function setReasoning(next: boolean) {
    setReasoningEnabled(next);
    setUserPrompt((current) => promptForReasoning(current, next));
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
    setMaxTokens(DEFAULT_MAX_TOKENS);
    setFramesPerSecond(DEFAULT_FRAMES_PER_SECOND);
    setRepetitionPenalty(DEFAULT_REPETITION_PENALTY);
    setSeed(DEFAULT_SEED);
    setParametersOpen(false);
    setReasoningExpanded(true);
    setOutputTab("preview");
    setStatus("Ready");
    setResult(null);
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
    setOutputTab("preview");
    setReasoningExpanded(true);
    setStatus("Running inference");

    try {
      const response = await fetch("/api/reason", {
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
            max_tokens: maxTokens,
            frames_per_second: framesPerSecond,
            repetition_penalty: repetitionPenalty,
            seed
          }
        })
      });
      const data = (await response.json()) as ApiResult;
      setResult(data);
      setStatus(response.ok ? "Complete" : "Backend error");
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        setResult({ status: "skip", message: "Task stopped by user" });
        setStatus("Stopped");
      } else {
        setResult({ error: error instanceof Error ? error.message : "Request failed" });
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
              <span>Downloadable</span>
            </div>
          </div>
          <p>
            Vision language model that excels in understanding the physical world using structured reasoning on videos
            or images.
          </p>
          <div className="tagRow">
            {HERO_TAGS.map((tag, index) => (
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
            reasoningEnabled={reasoningEnabled}
            reasoningExpanded={reasoningExpanded}
            repetitionPenalty={repetitionPenalty}
            requestPreview={requestPreview}
            reset={reset}
            result={result}
            run={run}
            seed={seed}
            selectedExampleId={selectedExampleId}
            setExamplesOpen={setExamplesOpen}
            setFramesPerSecond={setFramesPerSecond}
            setMaxTokens={setMaxTokens}
            setModel={setModel}
            setOutputTab={setOutputTab}
            setParametersOpen={setParametersOpen}
            setReasoning={setReasoning}
            setReasoningExpanded={setReasoningExpanded}
            setRepetitionPenalty={setRepetitionPenalty}
            setSeed={setSeed}
            setSelectedExampleId={setSelectedExampleId}
            setSystemPrompt={setSystemPrompt}
            setTemperature={setTemperature}
            setTopP={setTopP}
            setUserPrompt={setUserPrompt}
            status={status}
            systemPrompt={systemPrompt}
            temperature={temperature}
            topP={topP}
            userPrompt={userPrompt}
            framesPerSecond={framesPerSecond}
            applyExample={applyExample}
          />
        ) : (
          <StaticTab tab={activeTab} model={model} backendInfo={backendInfo} requestPreview={requestPreview} />
        )}
      </section>
    </main>
  );
}

function ExperiencePanel({
  applyExample,
  backendInfo,
  copied,
  copyRequest,
  dragActive,
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
  setReasoning,
  setReasoningExpanded,
  setRepetitionPenalty,
  setSeed,
  setSelectedExampleId,
  setSystemPrompt,
  setTemperature,
  setTopP,
  setUserPrompt,
  status,
  systemPrompt,
  temperature,
  topP,
  userPrompt
}: {
  applyExample: () => Promise<void>;
  backendInfo: { backend?: string; base_url?: string } | null;
  copied: boolean;
  copyRequest: () => Promise<void>;
  dragActive: boolean;
  examplesOpen: boolean;
  framesPerSecond: number;
  handleDrag: (event: DragEvent<HTMLButtonElement>, active: boolean) => void;
  handleDrop: (event: DragEvent<HTMLButtonElement>) => Promise<void>;
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
  setReasoning: (enabled: boolean) => void;
  setReasoningExpanded: (expanded: boolean) => void;
  setRepetitionPenalty: (value: number) => void;
  setSeed: (value: number) => void;
  setSelectedExampleId: (id: string) => void;
  setSystemPrompt: (value: string) => void;
  setTemperature: (value: number) => void;
  setTopP: (value: number) => void;
  setUserPrompt: (value: string) => void;
  status: string;
  systemPrompt: string;
  temperature: number;
  topP: number;
  userPrompt: string;
}) {
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

      <div className="mobileIOTabs" role="tablist" aria-label="Input output">
        <button className="active">Input</button>
        <button>Output</button>
      </div>

      <div className="workspace">
        <section className="panel inputPanel">
          <div className="panelHeader">
            <h2>Input</h2>
            <button className="secondaryAction" onClick={() => setExamplesOpen(true)}>
              View Examples
              <ChevronDown size={15} />
            </button>
          </div>

          <label className="fieldLabel">Input</label>
          <button
            className={`dropzone${dragActive ? " dragActive" : ""}${media ? " hasMedia" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragEnter={(event) => handleDrag(event, true)}
            onDragOver={(event) => handleDrag(event, true)}
            onDragLeave={(event) => handleDrag(event, false)}
            onDrop={handleDrop}
            type="button"
          >
            <input ref={inputRef} type="file" accept=".mp4,.jpg,.jpeg,.png" onChange={handleFile} hidden />
            {media ? (
              <>
                <span className="mediaLoaded">
                  {media.kind === "image" ? <FileImage size={18} /> : <FileVideo size={18} />}
                  {media.name}
                </span>
                <span className="mediaPreview">
                  {media.kind === "image" ? (
                    <img src={media.previewUrl} alt="Selected input" />
                  ) : (
                    <video src={media.previewUrl} muted playsInline />
                  )}
                </span>
              </>
            ) : (
              <>
                <Upload size={22} />
                <span>Drop files here</span>
                <small>.mp4, .jpg, .jpeg, .png</small>
              </>
            )}
          </button>

          <PromptBox
            label="User Prompt"
            hint="Describe the video or ask a question. Enable reasoning by asking for the <think> format."
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
            reasoningEnabled={reasoningEnabled}
            repetitionPenalty={repetitionPenalty}
            seed={seed}
            setFramesPerSecond={setFramesPerSecond}
            setMaxTokens={setMaxTokens}
            setOpen={setParametersOpen}
            setReasoning={setReasoning}
            setRepetitionPenalty={setRepetitionPenalty}
            setSeed={setSeed}
            setTemperature={setTemperature}
            setTopP={setTopP}
            temperature={temperature}
            topP={topP}
          />

          <div className="runBar">
            <button className="resetButton" onClick={reset} type="button">
              <RotateCcw size={16} />
              Reset
            </button>
            <button className={`runButton${isRunning ? " running" : ""}`} onClick={run} type="button">
              {isRunning ? null : <Play size={16} fill="currentColor" />}
              {isRunning ? "Quit Task" : "Run"}
            </button>
          </div>
          <div className="statusLine" role="status">
            {status}
          </div>
        </section>

        <section className="panel outputPanel">
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
                parsedOutput={parsedOutput}
                reasoningEnabled={reasoningEnabled}
                reasoningExpanded={reasoningExpanded}
                result={result}
                setReasoningExpanded={setReasoningExpanded}
              />
            )}
          </div>
        </section>
      </div>

      <aside className="apiPanel">
        <div className="apiTopline">
          Backend: <strong>{backendInfo?.backend || "vLLM / OpenAI-compatible"}</strong>
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
  selectedExampleId,
  setSelectedExampleId
}: {
  applyExample: () => Promise<void>;
  close: () => void;
  selectedExampleId: string;
  setSelectedExampleId: (id: string) => void;
}) {
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
            {EXAMPLES.map((example) => {
              const checked = selectedExampleId === example.id;
              return (
                <button
                  className={checked ? "exampleItem checked" : "exampleItem"}
                  key={example.id}
                  role="radio"
                  aria-checked={checked}
                  onClick={() => setSelectedExampleId(example.id)}
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
                  </div>
                </button>
              );
            })}
          </div>
        </div>
        <div className="modalFooter">
          <button className="runButton" onClick={applyExample} type="button">
            Done
          </button>
        </div>
      </div>
    </div>
  );
}

function PreviewOutput({
  parsedOutput,
  reasoningEnabled,
  reasoningExpanded,
  result,
  setReasoningExpanded
}: {
  parsedOutput: { reasoning: string; answer: string; steps: string[] };
  reasoningEnabled: boolean;
  reasoningExpanded: boolean;
  result: ApiResult | null;
  setReasoningExpanded: (expanded: boolean) => void;
}) {
  if (result?.status === "error" || result?.error) {
    return <pre className="errorBox">{result.message || result.error}</pre>;
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

  if (result?.content) {
    return (
      <div className="responseStack">
        {reasoningEnabled && parsedOutput.reasoning ? (
          <ReasoningCard
            expanded={reasoningExpanded}
            reasoning={parsedOutput.reasoning}
            steps={parsedOutput.steps}
            setExpanded={setReasoningExpanded}
          />
        ) : null}
        <article className="answer">
          <p className="responseLabel">Response</p>
          <FormattedText text={parsedOutput.answer || result.content} />
        </article>
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

function ReasoningCard({
  expanded,
  reasoning,
  setExpanded,
  steps
}: {
  expanded: boolean;
  reasoning: string;
  setExpanded: (expanded: boolean) => void;
  steps: string[];
}) {
  if (!expanded) {
    return (
      <button className="reasoningCollapsed" onClick={() => setExpanded(true)} type="button">
        <CheckCircle2 size={16} />
        <span>Reasoning Complete</span>
        <ChevronDown size={15} />
      </button>
    );
  }

  const visibleSteps = steps.length > 0 ? steps : [reasoning];
  return (
    <article className="reasoningCard">
      <div className="reasoningTopline">
        <div>
          <h3>Reasoning Complete</h3>
          <p>Below is the entire thinking process the model went through to arrive at its response.</p>
        </div>
        <button onClick={() => setExpanded(false)} type="button">
          Collapse
          <ChevronDown size={15} />
        </button>
      </div>
      <ul>
        {visibleSteps.map((step, index) => (
          <li key={`${step}-${index}`}>
            <CheckCircle2 size={15} />
            <span>{step}</span>
            <ChevronRight size={14} />
          </li>
        ))}
      </ul>
    </article>
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
  reasoningEnabled,
  repetitionPenalty,
  seed,
  setFramesPerSecond,
  setMaxTokens,
  setOpen,
  setReasoning,
  setRepetitionPenalty,
  setSeed,
  setTemperature,
  setTopP,
  temperature,
  topP
}: {
  framesPerSecond: number;
  maxTokens: number;
  open: boolean;
  reasoningEnabled: boolean;
  repetitionPenalty: number;
  seed: number;
  setFramesPerSecond: (value: number) => void;
  setMaxTokens: (value: number) => void;
  setOpen: (open: boolean) => void;
  setReasoning: (enabled: boolean) => void;
  setRepetitionPenalty: (value: number) => void;
  setSeed: (value: number) => void;
  setTemperature: (value: number) => void;
  setTopP: (value: number) => void;
  temperature: number;
  topP: number;
}) {
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
          <SliderField label="Temperature" min={0} max={1} step={0.05} value={temperature} onChange={setTemperature} />
          <SliderField label="Top P" min={0.01} max={1} step={0.01} value={topP} onChange={setTopP} />
          <SliderField
            label="Repetition Penalty"
            min={1}
            max={2}
            step={0.05}
            value={repetitionPenalty}
            onChange={setRepetitionPenalty}
          />
          <SliderField
            label="Frames per Second"
            min={2}
            max={8}
            step={1}
            value={framesPerSecond}
            onChange={setFramesPerSecond}
          />
          <SliderField label="Max Tokens" min={128} max={4096} step={128} value={maxTokens} onChange={setMaxTokens} />
          <label className="seedField">
            <span>Seed</span>
            <input type="number" value={seed} disabled onChange={(event) => setSeed(Number(event.target.value))} />
          </label>
          <label className="reasoningSwitch">
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
            <Info size={15} />
          </label>
        </div>
      ) : null}
    </div>
  );
}

function SliderField({
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
    <fieldset className="sliderField">
      <legend>
        <span>{label}</span>
        <Info size={15} />
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

function StaticTab({
  backendInfo,
  model,
  requestPreview,
  tab
}: {
  backendInfo: { backend?: string; base_url?: string; gpu_name?: string } | null;
  model: string;
  requestPreview: unknown;
  tab: SectionTab;
}) {
  if (tab === "Model Card") {
    return (
      <div className="staticPanel">
        <h2>{model}</h2>
        <p>
          Cosmos3 Nano Reasoner is a VLM reasoning surface for video and image understanding. It does not expose a
          generation tower in this deployment.
        </p>
        <dl>
          <dt>Model type</dt>
          <dd>VLM / Reasoner</dd>
          <dt>Primary backend</dt>
          <dd>{backendInfo?.backend || "vLLM / OpenAI-compatible"}</dd>
          <dt>Endpoint</dt>
          <dd>{backendInfo?.base_url || "http://localhost:8000/v1"}</dd>
        </dl>
      </div>
    );
  }
  if (tab === "System Card") {
    return (
      <div className="staticPanel">
        <h2>System Card</h2>
        <p>
          This app sends image or video inputs to the local OpenAI-compatible Reasoner endpoint using multimodal chat
          messages. Reasoning can be toggled from the parameter accordion.
        </p>
        <dl>
          <dt>Supported input</dt>
          <dd>.mp4, .jpg, .jpeg, .png</dd>
          <dt>GPU</dt>
          <dd>{backendInfo?.gpu_name || "Detected on target host"}</dd>
          <dt>Privacy</dt>
          <dd>Do not upload confidential or personal data unless expressly permitted.</dd>
        </dl>
      </div>
    );
  }
  return (
    <div className="staticPanel">
      <h2>Deploy</h2>
      <p>Use the BYO-video skill to launch this model-matched Reasoner UI and optional batch inference companion.</p>
      <pre className="codeBlock">{safeJson(requestPreview)}</pre>
    </div>
  );
}
