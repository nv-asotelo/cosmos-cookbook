"""Wrapper for cosmos3.ray.gradio that adds a real upload widget for i2v.

Upstream (`cosmos3.ray.gradio`) drives i2v via a `vision_path` field in the
"Extra Arguments" JSON — no drag-and-drop widget. This wrapper rebuilds the
same UI but adds a `gr.Image(type="filepath")` component to the left column.
When the user uploads an image, the saved filepath is written into the
extra_input JSON as `vision_path`. Everything else (generate(), examples,
components, COMPONENTS, EXCLUDE_FIELDS) is reused unchanged from upstream.

This module lives in this repo (not in nvidia-cosmos/cosmos3-internal). To get
the widget merged upstream, open a PR against that repo — only after Alex
gives an explicit go (Upstream PR Policy).

Run on the same host as Ray Serve, with the cosmos3 venv:
    cd ~/cosmos3
    uv run --no-sync python /tmp/cosmos3_upload_gradio.py \
        --host 0.0.0.0 --port 8080 \
        --server-host localhost --server-port 8000 \
        --server-output-dir outputs/ray_serve
"""

import json
from functools import partial
from pathlib import Path

import gradio as gr

from cosmos3.args import OmniSampleOverrides
from cosmos3.common.args import tyro_cli
from cosmos3.ray.gradio import (
    Args,
    COMPONENTS,
    EXCLUDE_FIELDS,
    INPUTS_DIR,
    build_components,
    generate,
    get_info,
    load_input,
)


def update_extra_with_vision(image_path, current_json):
    """When the user uploads an image, inject its path into extra_input.vision_path.

    Leaves all other extra-arg keys alone. Clearing the upload removes vision_path.
    """
    try:
        data = json.loads(current_json) if current_json else {}
    except json.JSONDecodeError:
        data = {}
    if image_path:
        data["vision_path"] = str(image_path)
    else:
        data.pop("vision_path", None)
    return json.dumps(data, indent=2)


def ui_builder(args: Args) -> gr.Blocks:
    info = get_info(args)
    available_models = info["models"]
    if len(available_models) == 0:
        raise ValueError("No models available")

    default_example = "t2i"
    examples: dict[str, Path] = {}
    for p in INPUTS_DIR.rglob("*.json"):
        if "internal" in p.parts:
            continue
        if p.stem in examples:
            raise ValueError(f"Duplicate example file: {p}")
        examples[p.stem] = p
    if default_example not in examples:
        default_example = next(iter(sorted(examples.keys())), "")

    with gr.Blocks(title="Cosmos3 Omni Generator (with upload)") as ui:
        gr.Markdown("# Cosmos3 Omni Generator")
        gr.Markdown(
            "Upload an image to drive image-to-video. The uploaded file's path "
            "is injected into **Extra Arguments → `vision_path`** automatically. "
            "Clear the upload to revert to the example's default `vision_path`."
        )
        with gr.Accordion("Environment", open=False):
            gr.JSON(value=info["environment"])

        example_dropdown = gr.Dropdown(
            choices=["", *sorted(examples.keys())],
            value=default_example,
            label="Input preset",
        )

        with gr.Row():
            with gr.Column(scale=1):
                model_input = gr.Dropdown(
                    value=available_models[0],
                    choices=available_models,
                    label="Model",
                )
                generate_btn = gr.Button("Generate", variant="primary")

                image_upload = gr.Image(
                    type="filepath",
                    label="Conditioning image (drag-and-drop or click to upload)",
                    sources=["upload", "clipboard"],
                    height=240,
                )

                components = build_components(OmniSampleOverrides, COMPONENTS)

                with gr.Accordion("Extra Arguments", open=False):
                    extra_json = OmniSampleOverrides(name="").model_dump_json(
                        indent=2,
                        exclude={*COMPONENTS, *EXCLUDE_FIELDS},
                    )
                    extra_input = gr.Code(
                        extra_json,
                        language="json",
                        lines=10,
                    )

            with gr.Column(scale=1):
                media_output = gr.Gallery(label="Media", allow_preview=True)

                with gr.Accordion("Request", open=False):
                    request_output = gr.JSON()

                with gr.Accordion("Response", open=False):
                    response_output = gr.JSON()

        load_input_kwargs = dict(
            fn=partial(load_input, examples=examples),
            inputs=[example_dropdown],
            outputs=[model_input, *components.values(), extra_input],
        )
        example_dropdown.change(**load_input_kwargs)
        ui.load(**load_input_kwargs)

        image_upload.change(
            fn=update_extra_with_vision,
            inputs=[image_upload, extra_input],
            outputs=[extra_input],
        )

        generate_btn.click(
            fn=partial(generate, args=args),
            inputs=[model_input, *components.values(), extra_input],
            outputs=[media_output, request_output, response_output],
        )

    return ui


def main():
    args = tyro_cli(Args, description=__doc__)
    ui = ui_builder(args)
    # queue(): keep the generated result on the server until the client picks it up,
    # so a transient browser disconnect (VPN flap mid-inference) doesn't lose the run.
    # default_concurrency_limit=1 because we have one GPU; max_size buffers a few
    # browser refreshes without dropping in-flight work.
    ui.queue(default_concurrency_limit=1, max_size=8)
    # share=True publishes a *.gradio.live tunnel — its long-poll reconnect tolerates
    # short VPN drops far better than a direct LAN IP. The LAN URL still works for
    # anyone on the same network.
    ui.launch(
        server_name=args.host,
        server_port=args.port,
        share=True,
        allowed_paths=[str(args.server_output_dir), "/tmp"],
    )


if __name__ == "__main__":
    main()
