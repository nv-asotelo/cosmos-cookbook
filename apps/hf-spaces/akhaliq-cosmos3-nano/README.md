---
title: Cosmos3 Nano
emoji: 🌌
colorFrom: gray
colorTo: green
sdk: static
app_build_command: npm install && npm run build
app_file: dist/index.html
fullWidth: true
pinned: false
---

# Cosmos3 Nano Vite View

A Vite + React interface modeled after the public `akhaliq/Cosmos3-Nano`
Hugging Face Space experience. The UI is full-screen and media-first: generated
images or videos become the background, videos loop, and video-with-sound outputs
can be muted or unmuted in place.

The horde staging view is served with `server.mjs`, a small same-origin Node
proxy. The browser calls `/api/local-generate`, and the server forwards the
request to the local Cosmos3 Diffusers adapter, normally
`http://127.0.0.1:8010/generate`. This keeps the demo on the local GPU path and
avoids Hugging Face ZeroGPU quota limits.

## Local Development

```bash
npm install
npm run dev
```

## Build

```bash
npm run typecheck
npm run build
```

The static build is emitted to `dist/`.

## Local GPU Server

```bash
PORT=5185 DIFFUSERS_BASE_URL=http://127.0.0.1:8010 npm run serve:local
```
