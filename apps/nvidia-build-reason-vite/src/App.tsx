import {
  ChevronDown,
  Copy,
  ExternalLink,
  FileVideo,
  HelpCircle,
  Info,
  Menu,
  Play,
  RotateCcw,
  Search,
  Upload
} from "lucide-react";
import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";

const HERO_IMAGE = "https://assets.ngc.nvidia.com/products/api-catalog/images/cosmos-reason2-8b.jpg";
const SAMPLE_VIDEO = "/examples/race-car.mp4";
const COSMOS3_INFO_URL =
  (typeof import.meta !== "undefined" && (import.meta as { env?: Record<string, string> }).env?.VITE_COSMOS3_INFO_URL) ||
  "/api/active-model";
const DEFAULT_MODEL =
  (typeof import.meta !== "undefined" && (import.meta as { env?: Record<string, string> }).env?.VITE_MODEL_NAME) ||
  "Detecting model…";
const DEFAULT_USER_PROMPT = "";
const DEFAULT_SYSTEM_PROMPT = "";
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
const REASONING_STEPS = [
  "Okay, let's see. The user wants me to describe the video content shown across the sequence.",
  "Starting from the first frame at 00:00-00:02, there's an outdoor racing event on a track.",
  "Moving to 00:02-00:06, inside the cockpit of another vehicle, the driver prepares for a maneuver.",
  "At 00:06-00:08, this appears to be a continuation of the previous shot focusing on steering control.",
  "The next segment, 00:08-00:09, shifts back outside where we see a person near the vehicle.",
  "From 00:09-00:11, the same red race car seen earlier now performs a controlled drift.",
  "Between 00:11-00:13, another yellow-and-black car executes similar maneuvers beside it.",
  "In 00:13-00:17, aerial footage captures both cars drifting side by side, leaving tire smoke.",
  "Then, between 00:17-00:19, close-up shots show detailed views of the drivers and vehicles."
];

type MediaState = {
  name: string;
  kind: "video" | "image";
  previewUrl: string;
  dataUrl: string;
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

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function sampleToMedia(): Promise<MediaState> {
  const response = await fetch(SAMPLE_VIDEO);
  const blob = await response.blob();
  const file = new File([blob], "Race Car.mp4", { type: "video/mp4" });
  return {
    name: file.name,
    kind: "video",
    previewUrl: SAMPLE_VIDEO,
    dataUrl: await readFileAsDataUrl(file)
  };
}

export default function App() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [media, setMedia] = useState<MediaState | null>(null);
  const [userPrompt, setUserPrompt] = useState(DEFAULT_USER_PROMPT);
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT);
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
  } | null>(null);

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
  const [temperature, setTemperature] = useState(0.6);
  const [topP, setTopP] = useState(0.9);
  const [maxTokens, setMaxTokens] = useState(4096);
  const [framesPerSecond, setFramesPerSecond] = useState(4);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1);
  const [seed, setSeed] = useState(42);
  const [status, setStatus] = useState("Ready");
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<ApiResult | null>(null);
  const [copied, setCopied] = useState(false);

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
            { type: "text", text: userPrompt || "Describe the provided media." }
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
    [framesPerSecond, maxTokens, media, model, repetitionPenalty, seed, systemPrompt, temperature, topP, userPrompt]
  );

  async function handleFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    const kind = file.type.startsWith("image/") ? "image" : "video";
    setMedia({
      name: file.name,
      kind,
      previewUrl: URL.createObjectURL(file),
      dataUrl: await readFileAsDataUrl(file)
    });
    setResult(null);
    setStatus("Media loaded");
  }

  async function loadExample() {
    setStatus("Loading example");
    setMedia(await sampleToMedia());
    setResult(null);
    setStatus("Example loaded");
  }

  function reset() {
    setMedia(null);
    setUserPrompt(DEFAULT_USER_PROMPT);
    setSystemPrompt(DEFAULT_SYSTEM_PROMPT);
    setTemperature(0.6);
    setTopP(0.9);
    setMaxTokens(4096);
    setFramesPerSecond(4);
    setRepetitionPenalty(1);
    setSeed(42);
    setStatus("Ready");
    setResult(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  async function run() {
    setIsRunning(true);
    setResult(null);
    setStatus("Running inference");

    try {
      const response = await fetch("/api/reason", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: userPrompt,
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
      setResult({ error: error instanceof Error ? error.message : "Request failed" });
      setStatus("Request failed");
    } finally {
      setIsRunning(false);
    }
  }

  async function copyRequest() {
    await navigator.clipboard.writeText(JSON.stringify(requestPreview, null, 2));
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
        {["Experience", "Model Card", "System Card", "Deploy"].map((tab, index) => (
          <button className={index === 0 ? "active" : ""} key={tab}>
            {tab}
          </button>
        ))}
      </div>

      <section className="experience">
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
              <button className="secondaryAction" onClick={loadExample}>
                View Examples
                <ChevronDown size={15} />
              </button>
            </div>

            <label className="fieldLabel">Input</label>
            <button className="dropzone" onClick={() => inputRef.current?.click()}>
              <input ref={inputRef} type="file" accept=".mp4,.jpg,.jpeg,.png" onChange={handleFile} hidden />
              {media ? (
                <span className="mediaLoaded">
                  <FileVideo size={18} />
                  {media.name}
                </span>
              ) : (
                <>
                  <Upload size={22} />
                  <span>Drop files here</span>
                  <small>.mp4, .jpg, .jpeg, .png</small>
                </>
              )}
            </button>

            {media ? (
              <div className="previewFrame">
                {media.kind === "image" ? <img src={media.previewUrl} alt="Selected input" /> : <video src={media.previewUrl} controls />}
              </div>
            ) : null}

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

            <div className="parameterGrid">
              <NumberField label="Temp" value={temperature} min={0} max={1} step={0.1} onChange={setTemperature} />
              <NumberField label="Top P" value={topP} min={0} max={1} step={0.1} onChange={setTopP} />
              <NumberField label="FPS" value={framesPerSecond} min={2} max={8} step={1} onChange={setFramesPerSecond} />
              <NumberField label="Max Tokens" value={maxTokens} min={128} max={4096} step={128} onChange={setMaxTokens} />
            </div>

            <div className="runBar">
              <button className="resetButton" onClick={reset}>
                <RotateCcw size={16} />
                Reset
              </button>
              <button className="runButton" onClick={run} disabled={isRunning}>
                <Play size={16} fill="currentColor" />
                {isRunning ? "Running" : "Run"}
              </button>
            </div>
          </section>

          <section className="panel outputPanel">
            <div className="panelHeader">
              <div className="outputTabs">
                <h2>Output</h2>
                <button className="previewPill">Preview</button>
                <button className="jsonTab">JSON</button>
              </div>
            </div>
            <div className="outputBody">
              {result?.status === "error" || result?.error ? (
                <pre className="errorBox">{result.message || result.error}</pre>
              ) : result?.content ? (
                <article className="answer">{result.content}</article>
              ) : result?.files && result.files.length > 0 ? (
                <div className="resultFiles">
                  {result.files.map((file) => {
                    const src = file.b64 ? `data:${file.mime};base64,${file.b64}` : null;
                    if (src && file.mime.startsWith("video/")) {
                      return <video key={file.path} controls src={src} style={{ width: "100%" }} />;
                    }
                    if (src && file.mime.startsWith("image/")) {
                      return <img key={file.path} src={src} alt={file.path} style={{ width: "100%" }} />;
                    }
                    return <pre key={file.path} className="filePath">{file.path}</pre>;
                  })}
                </div>
              ) : (
                <ReasoningPreview />
              )}
            </div>
          </section>
        </div>

        <aside className="apiPanel">
          <div className="apiTopline">Using free API for development</div>
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
          <pre className="codeBlock">{JSON.stringify(requestPreview, null, 2)}</pre>
        </aside>
      </section>
    </main>
  );
}

function ReasoningPreview() {
  return (
    <article className="reasoningCard">
      <div className="reasoningTopline">
        <h3>Reasoning Complete</h3>
        <button>
          Collapse
          <ChevronDown size={15} />
        </button>
      </div>
      <p>Below is the entire thinking process the model went through to arrive at its response.</p>
      <ul>
        {REASONING_STEPS.map((step) => (
          <li key={step}>
            <span className="checkRing">✓</span>
            <span>{step}</span>
            <ChevronDown size={14} />
          </li>
        ))}
      </ul>
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
