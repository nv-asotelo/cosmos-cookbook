#!/usr/bin/env python3
"""
cosmos_deploy_monitor.py — Local deployment monitoring agent for /byo-video pipeline.

Manages the pipeline on an existing Brev instance: BUILDING wait → script deploy →
setup launch → Gradio live detection. Outputs a formatted checklist each poll cycle
so every Monitor notification shows full pipeline state.

If the instance goes UNHEALTHY, waits 3 min, then rotates to the next provider
in the priority list. When all providers are exhausted, exits 2 so the main agent
can AskUserQuestion.

Exit codes:
  0  DONE     — Gradio live. URL in /tmp/cosmos_deploy_state.json.
  1  ERROR    — Unrecoverable. Message in state file.
  2  INPUT    — Needs user decision. Options in state file.

Usage:
  python3 cosmos_deploy_monitor.py \\
    --instance c3r-nano \\
    --model-id nvidia/Cosmos3-Nano-Reasoner \\
    --model-size C3-8B \\
    --backend vllm \\
    --rate 3.70 \\
    --provider gpu-h100-sxm.1gpu-16vcpu-200gb \\
    --provider-label "H100 SXM"
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import time

STATE_FILE  = "/tmp/cosmos_deploy_state.json"
LIVE_FLAG   = "/tmp/gradio_live.flag"
SETUP_LOG   = "/tmp/byo_video_setup.log"
SCRIPTS_DIR = os.path.expanduser("~/.claude/scripts")

PROVIDER_PRIORITY = [
    {"type": "gpu-h100-sxm.1gpu-16vcpu-200gb", "label": "H100 SXM",       "rate": 3.70, "gpu": "H100"},
    {"type": "hyperstack_A100",                 "label": "A100-80GB",       "rate": 2.20, "gpu": "A100"},
    {"type": "hyperstack_H100",                 "label": "H100 Hyperstack", "rate": 3.70, "gpu": "H100"},
    {"type": "scaleway_A40",                    "label": "A40",             "rate": 1.10, "gpu": "A40"},
]

EXPECTED_SECS = {
    "vllm": {
        "2B": 900, "8B": 1200, "32B": 2400,
        "C3-2B": 900, "C3-8B": 1200, "C3-32B": 2400,
        "NEM-12B": 1200,
        "QW3-2B": 720, "QW3-8B": 900, "QW3-32B": 1800,
    },
    "hf": {"2B": 600, "8B": 720},
}

# Display labels for the 10 steps (script uses 1-7 + 9 + 9b + 10)
STEP_LABELS = {
    1: "GPU detect + VRAM tier",
    2: "HF auth + token validate",
    3: "NGC API key",
    4: "uv package manager",
    5: "cosmos-reason2 repo",
    6: "uv sync + CUDA libs",
    7: "PyAV + Gradio + requests",
    9: "Model weights download",
    10: "Gradio launch",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_local(args_list, timeout=30):
    result = subprocess.run(args_list, capture_output=True, text=True, timeout=timeout)
    return result.stdout.strip(), result.returncode


def brev_ls(instance):
    """Return (status, build, shell) or (None, None, None) if not found."""
    out, _ = _run_local(["brev", "ls"], timeout=20)
    for line in out.splitlines():
        if instance in line:
            parts = line.split()
            if len(parts) >= 4:
                return parts[1], parts[2], parts[3]
    return None, None, None


def brev_exec(instance, remote_cmd, timeout=60):
    """Run a command on a Brev instance. Returns (stdout, returncode)."""
    result = subprocess.run(
        ["brev", "exec", instance, remote_cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.stdout.strip(), result.returncode


def brev_create(instance, provider):
    gpu_flag = provider["gpu"]
    out, rc = _run_local(
        ["brev", "create", instance, "--gpu-name", gpu_flag, "--type", provider["type"]],
        timeout=60,
    )
    return out, rc


def brev_delete(instance):
    _run_local(["brev", "delete", instance, "--force"], timeout=30)


def write_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def elapsed_fmt(start):
    s = int(time.time() - start)
    return f"{s // 60}m {s % 60:02d}s"


def eta_fmt(start, model_size, backend):
    elapsed = int(time.time() - start)
    total   = EXPECTED_SECS.get(backend, {}).get(model_size, 1200)
    remain  = max(0, total - elapsed)
    if remain == 0:
        return "any moment"
    return f"~{remain // 60}m"


def parse_setup_log(log_text):
    """Return (done_steps: set[int], current_step: int|None, has_error: bool, last_line: str)."""
    done    = set()
    current = None
    has_err = False
    lines   = [l for l in log_text.splitlines() if l.strip()]

    for line in lines:
        if "[✓] Step" in line:
            try:
                raw = line.split("Step")[1].split(":")[0].strip()
                done.add(int(raw.rstrip("abcdefghij")))
            except (ValueError, IndexError):
                pass
        elif "[→] Step" in line:
            try:
                raw = line.split("Step")[1].split(":")[0].strip()
                current = int(raw.rstrip("abcdefghij"))
            except (ValueError, IndexError):
                pass
        if "  ✗  " in line or "exit status 1" in line:
            has_err = True

    last = lines[-1][:90] if lines else ""
    return done, current, has_err, last


def print_checklist(state, done_steps, current_step, last_log_line):
    """Print the full checklist. Lines within 200ms batch into one Monitor notification."""
    phase    = state["phase"]
    elapsed  = elapsed_fmt(state["start_ts"])
    eta      = eta_fmt(state["start_ts"], state["model_size"], state["backend"])
    instance = state["instance"]
    label    = state.get("provider_label", "?")
    rate     = state.get("rate", 0)
    model    = state.get("model_id", "?")
    backend  = state.get("backend", "?")
    msize    = state.get("model_size", "?")
    action   = state.get("last_action", "")

    PHASE_ORDER = ["BUILDING", "DEPLOY_SCRIPTS", "SETUP", "GRADIO_LIVE", "DONE"]

    def mark(target_phase):
        ci = PHASE_ORDER.index(phase)       if phase       in PHASE_ORDER else 0
        ti = PHASE_ORDER.index(target_phase) if target_phase in PHASE_ORDER else 99
        if ci > ti:  return "✓"
        if ci == ti: return "→"
        return " "

    buf = [
        f"Cosmos Deploy Monitor — {instance}  (elapsed: {elapsed} | ETA: {eta})",
        f"  {label} · ${rate:.2f}/hr · {model} ({msize}, {backend})",
        "──────────────────────────────────────────────────────────────",
        f"  [{mark('BUILDING')}] Wait for SHELL READY",
        f"  [{mark('DEPLOY_SCRIPTS')}] Deploy scripts",
    ]

    if phase in ("SETUP", "GRADIO_LIVE", "DONE"):
        buf.append(f"  [✓] Setup launched")
        all_done = phase in ("GRADIO_LIVE", "DONE")
        for num, label_s in STEP_LABELS.items():
            if num in done_steps or all_done:
                buf.append(f"       ✓ Step {num}: {label_s}")
            elif num == current_step:
                buf.append(f"       → Step {num}: {label_s}  (running)")
            else:
                buf.append(f"       [ ] Step {num}: {label_s}")
    else:
        buf.append(f"  [{mark('SETUP')}] Setup (byo_video_setup.py)")

    buf.append(f"  [{mark('GRADIO_LIVE')}] Gradio live")
    buf.append("──────────────────────────────────────────────────────────────")

    if action:
        buf.append(f"  ↳ {action}")
    if last_log_line and phase == "SETUP":
        buf.append(f"  log: {last_log_line}")

    if state.get("gradio_url"):
        buf.append(f"  🎉 {state['gradio_url']}")

    print("\n".join(buf), flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance",       required=True)
    ap.add_argument("--model-id",       required=True)
    ap.add_argument("--model-size",     default="2B")
    ap.add_argument("--backend",        default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--rate",           type=float, default=3.70)
    ap.add_argument("--provider",       default="gpu-h100-sxm.1gpu-16vcpu-200gb")
    ap.add_argument("--provider-label", default="H100 SXM")
    args = ap.parse_args()

    # Find this provider's index in the priority list for rotation
    prov_idx = next(
        (i for i, p in enumerate(PROVIDER_PRIORITY) if p["type"] == args.provider), 0
    )

    state = {
        "phase":          "BUILDING",
        "start_ts":       time.time(),
        "instance":       args.instance,
        "model_id":       args.model_id,
        "model_size":     args.model_size,
        "backend":        args.backend,
        "rate":           args.rate,
        "provider":       args.provider,
        "provider_label": args.provider_label,
        "gradio_url":     None,
        "last_action":    "Waiting for SHELL READY...",
        "exit_code":      None,
        "message":        None,
    }
    write_state(state)

    scripts_deployed = False
    setup_launched   = False
    unhealthy_start  = None
    done_steps: set  = set()
    current_step     = None

    # ── Poll loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            brev_status, brev_build, brev_shell = brev_ls(args.instance)
        except subprocess.TimeoutExpired:
            time.sleep(15)
            continue

        # ── UNHEALTHY detection ───────────────────────────────────────────────
        if brev_build in ("UNHEALTHY", "FAILED"):
            if unhealthy_start is None:
                unhealthy_start = time.time()
                state["last_action"] = f"⚠️ UNHEALTHY — 3-min recovery window starting"

            wait_remaining = 180 - int(time.time() - unhealthy_start)
            if wait_remaining > 0:
                state["phase"]       = "BUILDING"
                state["last_action"] = f"⚠️ UNHEALTHY — checking again in 30s ({wait_remaining}s left in window)"
                print_checklist(state, done_steps, current_step, "")
                write_state(state)
                time.sleep(30)
                continue

            # Recovery window expired — rotate provider
            print(f"✗ {args.provider_label} UNHEALTHY (3-min window expired) — rotating provider", flush=True)
            brev_delete(args.instance)
            time.sleep(15)

            next_providers = PROVIDER_PRIORITY[prov_idx + 1:]
            if not next_providers:
                state["phase"]    = "ERROR"
                state["exit_code"] = 2
                state["message"]  = (
                    f"All {len(PROVIDER_PRIORITY)} providers exhausted after UNHEALTHY on "
                    f"{args.provider_label}. Please choose an action."
                )
                state["options"] = ["Try again with same provider", "Cancel deployment"]
                write_state(state)
                print_checklist(state, done_steps, current_step, "")
                sys.exit(2)

            next_p           = next_providers[0]
            prov_idx        += 1
            args.provider    = next_p["type"]
            args.provider_label = next_p["label"]
            args.rate        = next_p["rate"]
            state["provider"]       = next_p["type"]
            state["provider_label"] = next_p["label"]
            state["rate"]           = next_p["rate"]
            state["last_action"]    = f"✗ {args.provider_label} — rotating to {next_p['label']}"

            print(f"  → trying {next_p['label']} ({next_p['type']})", flush=True)
            _, rc = brev_create(args.instance, next_p)
            if rc != 0:
                # Create failed — try next provider on next iteration
                prov_idx -= 1  # will be re-incremented next UNHEALTHY cycle
            unhealthy_start  = None
            scripts_deployed = False
            setup_launched   = False
            done_steps       = set()
            current_step     = None
            state["phase"]   = "BUILDING"
            write_state(state)
            time.sleep(30)
            continue

        else:
            unhealthy_start = None  # reset on clean status

        # ── BUILDING: wait for SHELL READY ────────────────────────────────────
        if brev_shell != "READY":
            state["phase"]       = "BUILDING"
            state["last_action"] = f"Building... (BUILD={brev_build})"
            print_checklist(state, done_steps, current_step, "")
            write_state(state)
            time.sleep(30)
            continue

        # ── SHELL READY: deploy scripts ───────────────────────────────────────
        if not scripts_deployed:
            state["phase"]       = "DEPLOY_SCRIPTS"
            state["last_action"] = "SHELL READY — deploying scripts..."
            print_checklist(state, done_steps, current_step, "")

            for name in ("byo_video_setup", "gradio_cr2_byo"):
                path = os.path.join(SCRIPTS_DIR, f"{name}.py")
                if not os.path.exists(path):
                    state["phase"]    = "ERROR"
                    state["exit_code"] = 1
                    state["message"]  = f"Script not found locally: {path}"
                    write_state(state)
                    print(f"ERROR: {state['message']}", flush=True)
                    sys.exit(1)

                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()

                remote_deploy = (
                    f"python3 -c \""
                    f"import base64; "
                    f"open('/tmp/{name}.py','wb').write("
                    f"base64.b64decode('{b64}'))\""
                )
                _, rc = brev_exec(args.instance, remote_deploy, timeout=90)
                if rc != 0:
                    # one retry
                    time.sleep(5)
                    _, rc = brev_exec(args.instance, remote_deploy, timeout=90)
                    if rc != 0:
                        state["phase"]    = "ERROR"
                        state["exit_code"] = 1
                        state["message"]  = f"Failed to deploy {name}.py after 2 attempts"
                        write_state(state)
                        print(f"ERROR: {state['message']}", flush=True)
                        sys.exit(1)

            scripts_deployed     = True
            state["last_action"] = "Scripts deployed ✓"
            write_state(state)

        # ── Launch setup in background ────────────────────────────────────────
        if not setup_launched:
            state["phase"]       = "SETUP"
            state["last_action"] = "Launching setup in background..."
            print_checklist(state, done_steps, current_step, "")

            inner = (
                f"export INFERENCE_BACKEND={args.backend} "
                f"MODEL_ID={args.model_id} "
                f"BREV_RATE_PER_HOUR={args.rate} "
                f"PATH=~/.local/bin:~/.cargo/bin:$PATH && "
                f"python3 /tmp/byo_video_setup.py"
            )
            launch_cmd = f"nohup bash -c \"{inner}\" > {SETUP_LOG} 2>&1 &"
            brev_exec(args.instance, launch_cmd, timeout=30)
            setup_launched       = True
            state["last_action"] = "Setup launched — watching log"
            write_state(state)
            time.sleep(20)
            continue

        # ── SETUP running: tail log, check live flag ──────────────────────────
        if state["phase"] == "SETUP":
            log_raw, _ = brev_exec(
                args.instance, f"tail -60 {SETUP_LOG} 2>/dev/null", timeout=30
            )
            done_steps, current_step, has_err, last_line = parse_setup_log(log_raw)

            # Check live flag
            flag_out, flag_rc = brev_exec(
                args.instance, f"cat {LIVE_FLAG} 2>/dev/null", timeout=15
            )
            url = flag_out.strip()
            if flag_rc == 0 and url.startswith("http"):
                state["gradio_url"]  = url
                state["phase"]       = "GRADIO_LIVE"
                state["last_action"] = f"Gradio live ✓"
                done_steps           = set(STEP_LABELS.keys())
                print_checklist(state, done_steps, None, "")
                state["exit_code"]   = 0
                write_state(state)
                print(f"DONE: {url}", flush=True)
                sys.exit(0)

            # Check for setup failure
            if has_err:
                state["last_action"] = f"⚠️ Error in setup log: {last_line}"

                # Check if vLLM or critical step failed
                critical_failure = any(
                    kw in log_raw
                    for kw in ("sys.exit(1)", "CUDA out of memory", "No space left", "SIGKILL")
                )
                if critical_failure:
                    state["phase"]    = "ERROR"
                    state["exit_code"] = 1
                    state["message"]  = f"Setup failed: {last_line}"
                    write_state(state)
                    print_checklist(state, done_steps, current_step, last_line)
                    sys.exit(1)
            else:
                state["last_action"] = last_line or "setup running..."

            print_checklist(state, done_steps, current_step, last_line)
            write_state(state)
            time.sleep(30)
            continue

        time.sleep(30)


if __name__ == "__main__":
    main()
