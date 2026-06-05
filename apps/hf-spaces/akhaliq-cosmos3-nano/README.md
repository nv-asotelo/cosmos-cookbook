---
title: Cosmos3 Nano
emoji: 🌌
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 6.15.2
python_version: '3.12'
app_file: app.py
pinned: false
suggested_hardware: a100-large
---

# Cosmos3 Nano — World Foundation Model

NVIDIA Cosmos 3 Nano: Generate physically plausible images and videos from text, images, and sound using a unified world foundation model.

## Features

- **Text → Image** — Single-frame generation from text prompts
- **Text → Video** — Multi-frame video generation with quality-control negative prompts
- **Image → Video** — Animate a conditioning image into video
- **Video + Sound** — Generate video with synchronized audio

## Architecture

Built with `gradio.Server` for a custom cinematic frontend with Gradio's queuing backend:
- Full-screen media display with glassmorphism controls
- `@app.api()` endpoints with GPU queuing and concurrency management
- `@spaces.GPU` for ZeroGPU allocation on HF Spaces

## Model

Uses [nvidia/Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) via the `Cosmos3OmniPipeline` from 🤗 Diffusers.
