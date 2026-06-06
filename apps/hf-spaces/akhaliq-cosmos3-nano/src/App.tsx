import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

type ModeId =
  | "text_to_video"
  | "text_to_image"
  | "image_to_video"
  | "text_to_video_with_sound";

type MediaKind = "image" | "video";

type GeneratedMedia = {
  kind: MediaKind;
  url: string;
  title: string;
  prompt: string;
  hasSound: boolean;
};

type UploadedAsset = {
  file: File;
  previewUrl: string;
  remotePath?: string;
};

const LOCAL_API_BASE = import.meta.env.VITE_COSMOS_LOCAL_API_BASE || "";

const DEFAULT_PROMPTS: Record<ModeId, string> = {
  text_to_video: "A bernese mountain dog becomes the world's AI Token King",
  text_to_image: "A bernese mountain dog becomes the world's AI Token King",
  image_to_video: "This dog becomes the world's AI Token King",
  text_to_video_with_sound:
    "This dog howls as he becomes the world's AI Token King"
};

const MODE_COPY: Record<
  ModeId,
  {
    label: string;
    resultTitle: string;
    loadingTitle: string;
    loadingSubcopy: string;
    endpoint: string;
    requiresImage?: boolean;
    mediaKind: MediaKind;
    hasSound?: boolean;
  }
> = {
  text_to_video: {
    label: "Text -> Video",
    resultTitle: "Text to Video",
    loadingTitle: "Generating Video...",
    loadingSubcopy: "Generating 121 frames at 24 FPS. This may take a few minutes.",
    endpoint: "text_to_video",
    mediaKind: "video"
  },
  text_to_image: {
    label: "Text -> Image",
    resultTitle: "Text to Image",
    loadingTitle: "Generating Image...",
    loadingSubcopy: "Creating your image with Cosmos3 Nano...",
    endpoint: "text_to_image",
    mediaKind: "image"
  },
  image_to_video: {
    label: "Image -> Video",
    resultTitle: "Image to Video",
    loadingTitle: "Generating Video...",
    loadingSubcopy: "Animating the attached image with Cosmos3 Nano...",
    endpoint: "image_to_video",
    requiresImage: true,
    mediaKind: "video"
  },
  text_to_video_with_sound: {
    label: "Video + Sound",
    resultTitle: "Video + Sound",
    loadingTitle: "Generating Video + Sound...",
    loadingSubcopy: "Generating video with synchronized audio...",
    endpoint: "text_to_video_with_sound",
    mediaKind: "video",
    hasSound: true
  }
};

const EXAMPLE_IMAGES = [
  {
    title: "Mountain meadow",
    url: "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1600&q=80"
  },
  {
    title: "Cozy living room",
    url: "https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?auto=format&fit=crop&w=1600&q=80"
  },
  {
    title: "Industrial floor",
    url: "https://images.unsplash.com/photo-1581092160607-ee22621dd758?auto=format&fit=crop&w=1600&q=80"
  }
];

function fileData(path: string, file?: File) {
  return {
    path,
    orig_name: file?.name || path.split("/").pop() || "image",
    meta: { _type: "gradio.FileData" }
  };
}

async function callLocalBackend(mode: ModeId, payload: Record<string, unknown>) {
  const request = await fetch(`${LOCAL_API_BASE}/api/local-generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode, payload })
  });

  if (!request.ok) {
    let message = `Local generation failed with HTTP ${request.status}.`;
    try {
      const errorPayload = await request.json();
      if (typeof errorPayload?.error === "string") message = errorPayload.error;
    } catch {
      // Keep the HTTP-level message.
    }
    throw new Error(message);
  }

  return (await request.json()) as {
    url: string;
    path?: string;
    kind?: MediaKind;
    hasSound?: boolean;
  };
}

async function readFileAsDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("Could not read the uploaded image."));
    reader.readAsDataURL(file);
  });
}

function PaperPlaneIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="icon">
      <path
        d="M20.7 3.3 3.8 10.6c-.9.4-.8 1.7.1 2l6 1.7 1.7 6c.3.9 1.6 1 2 .1l7.1-16.9c.3-.7-.3-1.4-1-1.2Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinejoin="round"
      />
      <path
        d="m10.1 14.1 5.3-5.3"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

function DownloadIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="icon small">
      <path
        d="M12 3v11m0 0 4-4m-4 4-4-4M5 19h14"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
      />
    </svg>
  );
}

export default function App() {
  const [mode, setMode] = useState<ModeId>("text_to_video");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPTS.text_to_video);
  const [negativePrompt, setNegativePrompt] = useState("");
  const [showNegative, setShowNegative] = useState(false);
  const [uploadedAsset, setUploadedAsset] = useState<UploadedAsset | null>(null);
  const [generatedMedia, setGeneratedMedia] = useState<GeneratedMedia | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isMuted, setIsMuted] = useState(true);
  const [notice, setNotice] = useState<string | null>(null);
  const [galleryOpen, setGalleryOpen] = useState(false);
  const [softGlow, setSoftGlow] = useState(true);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const runIdRef = useRef(0);

  const modeSpec = MODE_COPY[mode];
  const activeTitle = generatedMedia?.title || "Cosmos3 Nano";
  const activeSubtitle =
    generatedMedia?.prompt ||
    "Generate physically plausible worlds from text, images, and sound - powered by a unified world foundation model.";

  const hasGeneratedVideo = generatedMedia?.kind === "video";
  const canDownload = Boolean(generatedMedia?.url);

  const backgroundImage = useMemo(() => {
    if (generatedMedia) return null;
    if (uploadedAsset && mode === "image_to_video") return uploadedAsset.previewUrl;
    return EXAMPLE_IMAGES[mode === "text_to_image" ? 0 : mode === "image_to_video" ? 1 : 2].url;
  }, [generatedMedia, mode, uploadedAsset]);

  useEffect(() => {
    if (!notice) return undefined;
    const timer = window.setTimeout(() => setNotice(null), 4200);
    return () => window.clearTimeout(timer);
  }, [notice]);

  useEffect(() => {
    if (!videoRef.current) return;
    videoRef.current.muted = isMuted;
  }, [isMuted, generatedMedia]);

  useEffect(() => {
    return () => {
      if (uploadedAsset?.previewUrl) URL.revokeObjectURL(uploadedAsset.previewUrl);
    };
  }, [uploadedAsset]);

  function chooseMode(nextMode: ModeId) {
    setMode(nextMode);
    setPrompt((current) =>
      current === DEFAULT_PROMPTS[mode] ? DEFAULT_PROMPTS[nextMode] : current
    );
  }

  function handleFile(file: File | undefined) {
    if (!file) return;
    if (uploadedAsset?.previewUrl) URL.revokeObjectURL(uploadedAsset.previewUrl);
    setUploadedAsset({
      file,
      previewUrl: URL.createObjectURL(file)
    });
    setMode("image_to_video");
    setPrompt((current) =>
      current === DEFAULT_PROMPTS[mode] ? DEFAULT_PROMPTS.image_to_video : current
    );
  }

  async function handleGenerate(event: FormEvent) {
    event.preventDefault();

    const trimmedPrompt = prompt.trim();
    if (!trimmedPrompt) {
      setNotice("Describe the world you want to create.");
      return;
    }

    if (modeSpec.requiresImage && !uploadedAsset) {
      setNotice("Add an image before running Image to Video.");
      fileInputRef.current?.click();
      return;
    }

    const thisRun = runIdRef.current + 1;
    runIdRef.current = thisRun;
    setIsGenerating(true);
    setNotice(null);

    try {
      let payload: Record<string, unknown>;

      if (mode === "text_to_image") {
        payload = {
          prompt: trimmedPrompt,
          height: 720,
          width: 1280,
          num_inference_steps: 25
        };
      } else if (mode === "image_to_video") {
        const mediaDataUrl = await readFileAsDataUrl((uploadedAsset as UploadedAsset).file);
        if (runIdRef.current !== thisRun) return;
        payload = {
          image_path: fileData("uploaded-image.png", uploadedAsset?.file),
          mediaDataUrl,
          prompt: trimmedPrompt,
          negative_prompt: negativePrompt.trim(),
          num_frames: 121,
          height: 544,
          width: 960,
          fps: 24,
          num_inference_steps: 25
        };
      } else {
        payload = {
          prompt: trimmedPrompt,
          negative_prompt: negativePrompt.trim(),
          num_frames: 121,
          height: 544,
          width: 960,
          fps: 24,
          num_inference_steps: 25
        };
      }

      const result = await callLocalBackend(mode, payload);
      if (runIdRef.current !== thisRun) return;

      const url = typeof result?.url === "string" ? result.url : result?.path;
      if (!url || typeof url !== "string") {
        throw new Error("The local backend completed but did not return a media URL.");
      }

      setGeneratedMedia({
        kind: result.kind || modeSpec.mediaKind,
        url,
        title: modeSpec.resultTitle,
        prompt: trimmedPrompt,
        hasSound: Boolean(modeSpec.hasSound && result.hasSound)
      });
      setIsMuted(!Boolean(modeSpec.hasSound && result.hasSound));
    } catch (error) {
      console.error(error);
      setNotice(error instanceof Error ? error.message : "Generation failed.");
    } finally {
      if (runIdRef.current === thisRun) setIsGenerating(false);
    }
  }

  function reset() {
    runIdRef.current += 1;
    setIsGenerating(false);
    setGeneratedMedia(null);
    setMode("text_to_video");
    setPrompt(DEFAULT_PROMPTS.text_to_video);
    setNegativePrompt("");
    setShowNegative(false);
    setNotice(null);
  }

  function cancelGeneration() {
    runIdRef.current += 1;
    setIsGenerating(false);
  }

  function toggleSound() {
    if (!hasGeneratedVideo) return;
    setIsMuted((current) => !current);
    window.setTimeout(() => {
      videoRef.current?.play().catch(() => {
        setNotice("Use the video area once to allow playback.");
      });
    }, 0);
  }

  return (
    <main className={`appShell ${softGlow ? "softGlow" : ""}`}>
      <section className="mediaStage" aria-label="Cosmos3 Nano workspace">
        {generatedMedia?.kind === "video" ? (
          <video
            ref={videoRef}
            className="stageMedia"
            src={generatedMedia.url}
            autoPlay
            loop
            playsInline
            muted={isMuted}
            onClick={toggleSound}
          />
        ) : null}

        {generatedMedia?.kind === "image" ? (
          <img className="stageMedia" src={generatedMedia.url} alt="Generated result" />
        ) : null}

        {!generatedMedia && backgroundImage ? (
          <img className="stageMedia placeholderMedia" src={backgroundImage} alt="" />
        ) : null}

        <div className="stageShade" />
        <div className="grain" />

        <header className="brandHeader">
          <div className="brandMark" aria-hidden="true">
            <span className="nvEye" />
          </div>
          <div className="brandText">
            <strong>Cosmos3</strong>
            <span>Nano</span>
          </div>
        </header>

        <div className="topActions" aria-label="View options">
          <button
            className="iconButton"
            type="button"
            aria-label="Open examples"
            onClick={() => setGalleryOpen((current) => !current)}
          >
            <span className="gridIcon" />
          </button>
          <button
            className="iconButton"
            type="button"
            aria-label="Toggle glow"
            onClick={() => setSoftGlow((current) => !current)}
          >
            <span className="sunIcon" />
          </button>
        </div>

        {galleryOpen ? (
          <div className="exampleTray" role="dialog" aria-label="Prompt examples">
            {Object.entries(DEFAULT_PROMPTS).map(([id, value]) => (
              <button
                key={id}
                type="button"
                className="exampleTile"
                onClick={() => {
                  setMode(id as ModeId);
                  setPrompt(value);
                  setGalleryOpen(false);
                }}
              >
                <span>{MODE_COPY[id as ModeId].resultTitle}</span>
                <small>{value}</small>
              </button>
            ))}
          </div>
        ) : null}

        {hasGeneratedVideo ? (
          <button className="soundPill" type="button" onClick={toggleSound}>
            <span aria-hidden="true">{isMuted ? "Muted" : "Sound"}</span>
            {isMuted ? "Click to unmute" : "Click to mute"}
          </button>
        ) : null}

        <section
          className={`heroCopy ${generatedMedia ? "resultHero" : "idleHero"}`}
          aria-live="polite"
        >
          <h1>{activeTitle}</h1>
          <p>{activeSubtitle}</p>
        </section>

        {generatedMedia ? (
          <div className="resultActions">
            <button className="secondaryAction" type="button" onClick={reset}>
              <span aria-hidden="true">+</span>
              New
            </button>
            <a
              className={`downloadAction ${canDownload ? "" : "disabled"}`}
              href={generatedMedia.url}
              download
              aria-disabled={!canDownload}
              onClick={(event) => {
                if (!canDownload) event.preventDefault();
              }}
            >
              <DownloadIcon />
              Download
            </a>
          </div>
        ) : null}

        <form className="composer" onSubmit={handleGenerate}>
          {uploadedAsset ? (
            <div className="uploadChip">
              <img src={uploadedAsset.previewUrl} alt="" />
              <span>{uploadedAsset.file.name}</span>
              <button
                type="button"
                aria-label="Remove uploaded image"
                onClick={() => setUploadedAsset(null)}
              >
                ×
              </button>
            </div>
          ) : null}

          <div className="promptRow">
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder="Describe the world you want to create..."
              rows={2}
              aria-label="Prompt"
            />
            <button
              className="sendButton"
              type="submit"
              aria-label="Generate"
              disabled={isGenerating}
            >
              <PaperPlaneIcon />
            </button>
          </div>

          {showNegative ? (
            <textarea
              className="negativeField"
              value={negativePrompt}
              onChange={(event) => setNegativePrompt(event.target.value)}
              placeholder="Add details to avoid..."
              rows={2}
              aria-label="Negative prompt"
            />
          ) : null}

          <div className="modeBar" role="group" aria-label="Generation mode">
            {(Object.keys(MODE_COPY) as ModeId[]).map((id) => (
              <button
                key={id}
                type="button"
                className={`modeChip ${mode === id ? "active" : ""}`}
                onClick={() => chooseMode(id)}
              >
                {MODE_COPY[id].label}
              </button>
            ))}
            <span className="divider" />
            <input
              ref={fileInputRef}
              className="hiddenInput"
              type="file"
              accept="image/*"
              onChange={(event) => handleFile(event.target.files?.[0])}
            />
            <button
              type="button"
              className={`modeChip imageAttach ${mode === "image_to_video" ? "active" : ""}`}
              onClick={() => fileInputRef.current?.click()}
            >
              + Image
            </button>
            <button
              type="button"
              className={`modeChip ${showNegative ? "active" : ""}`}
              onClick={() => setShowNegative((current) => !current)}
            >
              - Negative
            </button>
          </div>
        </form>

        {isGenerating ? (
          <div className="generatingOverlay" role="status" aria-live="assertive">
            <div className="spinner" />
            <strong>{modeSpec.loadingTitle}</strong>
            <span>{modeSpec.loadingSubcopy}</span>
            <button type="button" onClick={cancelGeneration}>
              Cancel
            </button>
          </div>
        ) : null}

        {notice ? <div className="noticeToast">{notice}</div> : null}
      </section>
    </main>
  );
}
