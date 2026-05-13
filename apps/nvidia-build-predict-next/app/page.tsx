"use client";

import {
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
  Upload
} from "lucide-react";
import { ChangeEvent, CSSProperties, useEffect, useMemo, useRef, useState } from "react";

const HERO_IMAGE = "https://assets.ngc.nvidia.com/products/api-catalog/images/cosmos-predict1-5b.jpg";
const SAMPLE_VIDEO = "/examples/race-car.mp4";
// Standing order: header model name is auto-detected from the live backend on
// page load; env vars are fallbacks only. See cosmos3_info_server.py.
const COSMOS3_INFO_URL =
  process.env.NEXT_PUBLIC_COSMOS3_INFO_URL || "http://10.57.233.111:8088/active-model";
const DEFAULT_MODEL = process.env.NEXT_PUBLIC_MODEL_NAME || "Detecting model…";
const DEFAULT_PROMPT = "A first person view from a robot working in a chemical plant.";
const HERO_TAGS = [
  "physical ai",
  "world foundation model",
  "robotics",
  "simulation",
  "synthetic data generation",
  "video-to-world",
  "image-to-world",
  "future state generation"
];
const MODEL_CHOICES = [
  "nvidia/cosmos-predict1-5b",
  "nvidia/cosmos-predict1-7b-video2world",
  "nvidia/cosmos-predict2-5-2b",
  "nvidia/cosmos-predict2-5-14b"
];
const COLLECTIONS = ["cosmos-predict1", "cosmos-predict25", "cosmos3", "nvidia-cosmos-2", "cosmos"];
const VIDEO_PARAMS = {
  height: 704,
  width: 1280,
  frames_count: 121,
  frames_per_sec: 24
};
const PROGRESS_FRAME_COUNT = 6;
const PROGRESS_TOTAL_FRAMES = VIDEO_PARAMS.frames_count;

type WorldMode = "Video-to-World" | "Image-to-World";
type SchemaMode = "local_nim" | "build_openapi";
type MediaState = {
  name: string;
  kind: "video" | "image";
  previewUrl: string;
  dataUrl: string;
};
type ApiResult = {
  videoDataUrl?: string;
  assetUrl?: string;
  error?: string;
  diagnostic?: Record<string, unknown>;
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

export default function Page() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [collection, setCollection] = useState("cosmos-predict1");
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [models, setModels] = useState<string[]>(MODEL_CHOICES);
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
  const [worldMode, setWorldMode] = useState<WorldMode>("Video-to-World");
  const [schemaMode, setSchemaMode] = useState<SchemaMode>("local_nim");
  const [media, setMedia] = useState<MediaState | null>(null);
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [guidanceScale, setGuidanceScale] = useState(7);
  const [steps, setSteps] = useState(35);
  const [seed, setSeed] = useState(-1);
  const [inputImageIndex, setInputImageIndex] = useState(0);
  const [status, setStatus] = useState("Ready");
  const [isRunning, setIsRunning] = useState(false);
  const [progressPercent, setProgressPercent] = useState(0);
  const [progressFrames, setProgressFrames] = useState<string[]>([]);
  const [result, setResult] = useState<ApiResult | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    // Standing order: never override the model name from /api/models — the
    // live-backend probe in the earlier useEffect is the source of truth.
    // This effect only enriches the dropdown choices.
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

  useEffect(() => {
    if (!isRunning) return () => undefined;

    const interval = window.setInterval(() => {
      setProgressPercent((current) => {
        if (current < 28) return current + 7;
        if (current < 68) return current + 4;
        if (current < 90) return current + 2;
        return Math.min(current + 0.8, 96);
      });
    }, 950);

    return () => window.clearInterval(interval);
  }, [isRunning]);

  const requestPreview = useMemo(() => {
    if (schemaMode === "build_openapi") {
      return {
        endpoint: "POST https://ai.api.nvidia.com/v1/infer",
        collection,
        model,
        payload: {
          prompt,
          input_image_index: inputImageIndex,
          seed: seed >= 0 ? seed : null
        },
        response: { asset_url: "https://..." }
      };
    }
    return {
      endpoint: "POST /generate",
      collection,
      model,
      payload: {
        name: "ui-<auto>",
        model,
        prompt,
        vision_path: media ? "<written to /tmp/uploads/...>" : "<upload required>",
        num_frames: VIDEO_PARAMS.frames_count,
        resolution: 720,
        aspect_ratio: "16,9",
        fps: VIDEO_PARAMS.frames_per_sec,
        num_steps: steps,
        guidance: guidanceScale,
        seed: seed >= 0 ? seed : "random"
      }
    };
  }, [collection, guidanceScale, inputImageIndex, media, model, prompt, schemaMode, seed, steps, worldMode]);

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
    setProgressPercent(0);
    setStatus("Conditioning media loaded");
  }

  async function loadExample() {
    setStatus("Loading example");
    setWorldMode("Video-to-World");
    setMedia(await sampleToMedia());
    setResult(null);
    setProgressPercent(0);
    setStatus("Example loaded");
  }

  function reset() {
    setCollection("cosmos-predict1");
    setModel(DEFAULT_MODEL);
    setWorldMode("Video-to-World");
    setSchemaMode("local_nim");
    setMedia(null);
    setPrompt(DEFAULT_PROMPT);
    setGuidanceScale(7);
    setSteps(35);
    setSeed(-1);
    setInputImageIndex(0);
    setStatus("Ready");
    setProgressPercent(0);
    setResult(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  async function run() {
    setIsRunning(true);
    setResult(null);
    setProgressPercent(4);
    setStatus("Predicting frames");
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mediaDataUrl: media?.dataUrl,
          mediaKind: media?.kind,
          worldMode,
          prompt,
          model,
          guidanceScale,
          steps,
          seed,
          inputImageIndex
        })
      });
      const data = (await response.json()) as ApiResult;
      setResult(data);
      setProgressPercent(100);
      setStatus(response.ok ? "Complete" : "Backend error");
    } catch (error) {
      setResult({ error: error instanceof Error ? error.message : "Request failed" });
      setProgressPercent(100);
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

  const accepts = worldMode === "Image-to-World" ? ".jpg,.jpeg,.png" : ".mp4";

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
            World foundation model that generates future video from image or short-video conditioning for physical AI
            simulation and synthetic data workflows.
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

        <div className="workspace">
          <section className="panel inputPanel">
            <div className="panelHeader">
              <h2>Input</h2>
              <button className="secondaryAction" onClick={loadExample}>
                View Examples
              </button>
            </div>

            <label className="fieldLabel">World Creation Mode</label>
            <div className="modeGrid">
              {(["Video-to-World", "Image-to-World"] as const).map((mode) => (
                <button
                  className={worldMode === mode ? "active" : ""}
                  key={mode}
                  onClick={() => {
                    setWorldMode(mode);
                    setMedia(null);
                    setProgressPercent(0);
                    if (inputRef.current) inputRef.current.value = "";
                  }}
                >
                  {mode === "Video-to-World" ? <FileVideo size={16} /> : <ImageIcon size={16} />}
                  <span>{mode}</span>
                  <small>{mode === "Video-to-World" ? "First 9 frames condition motion" : "First frame condition"}</small>
                </button>
              ))}
            </div>

            <label className="fieldLabel">Input</label>
            <button className="dropzone predictDropzone" onClick={() => inputRef.current?.click()}>
              <input ref={inputRef} type="file" accept={accepts} onChange={handleFile} hidden />
              {media ? (
                <span className="mediaLoaded">
                  {media.kind === "video" ? <FileVideo size={18} /> : <ImageIcon size={18} />}
                  {media.name}
                </span>
              ) : (
                <>
                  <Upload size={22} />
                  <span>{worldMode === "Video-to-World" ? "Drop source video here" : "Drop source image here"}</span>
                  <small>{accepts}</small>
                </>
              )}
            </button>

            {media ? (
              <div className="previewFrame">
                {media.kind === "image" ? (
                  <img src={media.previewUrl} alt="Selected input" />
                ) : (
                  <video src={media.previewUrl} controls />
                )}
              </div>
            ) : null}

            <PromptBox
              label="Prompt"
              hint="Describe the future world state to generate."
              max={1000}
              value={prompt}
              onChange={setPrompt}
              rows={4}
            />

            <div className="parameterGrid predictParams">
              <NumberField label="Guidance" value={guidanceScale} min={1} max={10} step={0.5} onChange={setGuidanceScale} />
              <NumberField label="Steps" value={steps} min={1} max={50} step={1} onChange={setSteps} />
              <NumberField label="Seed" value={seed} min={-1} max={2147483647} step={1} onChange={setSeed} />
              <NumberField label="Image Index" value={inputImageIndex} min={0} max={1} step={1} onChange={setInputImageIndex} />
            </div>

            <div className="runBar">
              <button className="resetButton" onClick={reset}>
                <RotateCcw size={16} />
                Reset
              </button>
              <button className="runButton" onClick={run} disabled={isRunning}>
                <Play size={16} fill="currentColor" />
                {isRunning ? "Generating" : "Generate"}
              </button>
            </div>
          </section>

          <section className="panel outputPanel">
            <div className="panelHeader">
              <div className="outputTabs">
                <h2>Output</h2>
                <button className="previewPill">Preview</button>
              </div>
              <span className="statusPill">{status}</span>
            </div>
            <div className="outputBody">
              {isRunning ? (
                <GenerationProgress media={media} frames={progressFrames} progress={progressPercent} />
              ) : result?.error ? (
                <FailureReport result={result} />
              ) : result?.videoDataUrl ? (
                <video className="resultVideo" src={result.videoDataUrl} controls />
              ) : result?.assetUrl ? (
                <a className="assetLink" href={result.assetUrl} target="_blank">
                  Open generated asset
                  <ExternalLink size={15} />
                </a>
              ) : (
                <WorldPreview />
              )}
            </div>
          </section>
        </div>

        <aside className="apiPanel">
          <div className="apiTopline">API request</div>
          <div className="apiButtons">
            <button onClick={copyRequest}>
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
                <option value="local_nim">Cosmos3 Ray Serve /generate</option>
                <option value="build_openapi">NVIDIA Build OpenAPI preview</option>
              </select>
            </label>
          </div>
          <pre className="codeBlock">{JSON.stringify(requestPreview, null, 2)}</pre>
        </aside>
      </section>
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
  progress
}: {
  media: MediaState | null;
  frames: string[];
  progress: number;
}) {
  const generatedFrames = Math.max(1, Math.min(PROGRESS_TOTAL_FRAMES, Math.round((progress / 100) * PROGRESS_TOTAL_FRAMES)));
  const displayFrames = frames.length > 0 ? frames : Array.from({ length: PROGRESS_FRAME_COUNT }, () => "");
  const fallbackLabel = media?.kind === "image" ? "Image condition" : "Video condition";

  return (
    <article className="generationProgress" aria-live="polite">
      <div className="progressHeader">
        <p>
          <strong>The autoregressive model is working:</strong> Predicting future frames for you...
        </p>
        <span>{generatedFrames} / {PROGRESS_TOTAL_FRAMES} frames</span>
      </div>
      <div className="progressTrack" aria-label="Generation progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress)} role="progressbar">
        <span style={{ width: `${Math.max(4, progress)}%` }} />
      </div>
      <div className="generatedFrameStrip">
        {displayFrames.map((src, index) => {
          const threshold = (index / PROGRESS_FRAME_COUNT) * 86;
          const readiness = Math.max(0, Math.min(1, (progress - threshold) / 34));
          const frameNumber = Math.min(PROGRESS_TOTAL_FRAMES, Math.max(1, Math.round(((index + 1) / PROGRESS_FRAME_COUNT) * PROGRESS_TOTAL_FRAMES)));
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

function WorldPreview() {
  return (
    <article className="worldPreview">
      <div className="worldFrameGrid">
        {["Condition", "Latent rollout", "Future frames"].map((label, index) => (
          <div className="worldFrame" key={label}>
            <span>{String(index + 1).padStart(2, "0")}</span>
            <strong>{label}</strong>
          </div>
        ))}
      </div>
      <h3>Generated Future World</h3>
      <p>
        Upload a short conditioning clip or image, then generate a future-state video. The local NIM path returns a
        base64 MP4; the hosted Build schema returns an asset URL.
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
