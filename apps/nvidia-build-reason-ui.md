# NVIDIA Build Reason UI Comparison

Two local front-end implementations mirror the Build.NVIDIA `cosmos-reason2-8b`
experience page so page owners can compare framework fit without using Gradio.

## Apps

- `apps/nvidia-build-reason-next` - Next.js App Router with local API routes.
- `apps/nvidia-build-reason-vite` - Vite React with a small Express API proxy.

Both apps use the same visible page structure:

- NVIDIA-style top navigation
- `cosmos-reason2-8b` hero
- Experience / Model Card / System Card / Deploy tabs
- media upload area for `.mp4`, `.jpg`, `.jpeg`, `.png`
- user/system prompts
- request parameter controls
- API request preview panel
- output panel

## Backend

Both apps proxy to an OpenAI-compatible vLLM endpoint.

Default environment:

```bash
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=EMPTY
MODEL_NAME=nvidia/Cosmos3-Nano-Reasoner
```

For the current Horde instance:

```bash
VLLM_BASE_URL=http://10.57.233.111:8000/v1
VLLM_API_KEY=EMPTY
MODEL_NAME=nvidia/Cosmos3-Nano-Reasoner
```

## Run Locally

Next.js:

```bash
cd apps/nvidia-build-reason-next
VLLM_BASE_URL=http://10.57.233.111:8000/v1 \
VLLM_API_KEY=EMPTY \
MODEL_NAME=nvidia/Cosmos3-Nano-Reasoner \
npm run dev -- -H 0.0.0.0 -p 3000
```

Vite React:

```bash
cd apps/nvidia-build-reason-vite
VLLM_BASE_URL=http://10.57.233.111:8000/v1 \
VLLM_API_KEY=EMPTY \
MODEL_NAME=nvidia/Cosmos3-Nano-Reasoner \
PORT=5173 \
npm run dev
```

## QA

The checked screenshots live under:

```text
outputs/build-reason-ui-qa/
```

Build checks:

```bash
cd apps/nvidia-build-reason-next && npm run typecheck && npm run build
cd apps/nvidia-build-reason-vite && npm run typecheck && npm run build
```
