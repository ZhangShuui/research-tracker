# Reproducing a paper on the HKUST SuperPod — agent guide

**Audience:** a coding agent (Claude Code / Codex / …) asked to *reproduce a paper* on the
HKUST SuperPod. Follow this guide. It encodes a fixed SOP, the two Python modules that do the
work, and the hard-won conventions that make a reproduction actually run (skip them and you WILL
re-hit every wall we already hit).

**Golden rules**
1. Do **not** ad-hoc `ssh` and hand-run cluster commands. Use `paper_tracker.reproduction` (built on
   `paper_tracker.superpod`). Those carry the connection multiplexing, login-shell wrapping, SLURM
   module loading, and job polling.
2. **Every compute step is human-gated.** Call it with `dry_run=True` first, show the user the
   generated `sbatch`, get approval, then run for real. Never burn GPU before the user OKs the step.
3. Produce a `report.md` whose last section is an **anomaly hunt → brainstorm seeds**. The point of a
   reproduction here is *reproduce → find load-bearing/under-justified mechanisms → seed ideas*, not
   just "did the number match."

---

## The 4-stage SOP

1. **init** — make a dedicated workdir under `/project/visworld01/repro-<slug>/` and `git clone` the
   official code (if any).
2. **build_container** — build/patch an enroot container from an NGC base image + a paper-specific
   build script. (Expect a short patch loop; see gotchas.)
3. **run_step** — write & submit experiment `sbatch`(s), **step by step**, each in the workdir.
4. **collect + summarize** — gather results and write `report.md`.

---

## The modules

### `paper_tracker.reproduction` (orchestration — use this)
```python
init(name, *, repo_url=None, project_root=None) -> Reproduction
    # Stage 1: <root>/repro-<slug>/ (+ logs/ outputs/), git clone repo_url if given.

build_container(repro, *, base, build_script, save_as=None,
                mounts=None, remap_root=True, writable=True,
                partition="preempt", gpus=1, cpus=8, mem="64G", time="01:00:00",
                wait_timeout=5400, dry_run=False) -> (state, log)
    # Stage 2. base = NGC URI "docker://nvcr.io#..." for a fresh build, OR a local .sqsh path to
    # PATCH an existing image. Saves to <save_as>.tmp then atomically mv's onto save_as
    # (default <workdir>/<slug>.sqsh) so a failed build never clobbers a good image.

run_step(repro, step_name, command, *, container=None, workdir=None,
         mounts=None, remap_root=True,
         partition="preempt", gpus=1, cpus=8, mem="96G", time="01:30:00",
         wait_timeout=5400, dry_run=False) -> (state, log)
    # Stage 3. If container (NGC URI or local .sqsh) is given, `command` runs inside it via
    # `srun --container-image`; otherwise on the host node. `command` is one shell string.

collect(repro, *, files=("outputs/metrics.json",)) -> dict   # fetch result files
summarize(repro, *, cfg, paper, collected, notes="", model="opus", timeout=600) -> str  # LLM report
write_report(repro, content, *, filename="report.md") -> str  # write into the workdir
```
`state` is the terminal SLURM state (`COMPLETED` / `FAILED` / …) or `"DRY_RUN"` (then the 2nd return
value is the generated sbatch text). Jobs are recorded in `repro.jobs`.

### `paper_tracker.superpod` (lower-level channel — only if you need it)
`run(cmd, *, login_shell=True, timeout)`, `slurm(cmd)`, `make_workdir(name)`, `write_remote(path,
content)`, `fetch(path)`, `submit(script_path)`, `job_state(id)`, `wait(id, *, poll, timeout)`,
`reachable()`, `master_alive()`, `close()`. Cluster commands need a **login shell** (so `module load
slurm` works) — `run`/`slurm` handle that. `write_remote` pipes content over stdin (no quoting hell).

---

## How to actually run it

`build_container` / `run_step` **block** until the SLURM job is terminal (minutes). So drive them
from a small Python script and run that script as a **background process**, then poll its output —
don't call them inline in a single turn. Pattern that works (mirror it):

```python
# /tmp/repro_driver.py   (run with:  uv run python /tmp/repro_driver.py   in the background)
from paper_tracker import reproduction
from paper_tracker.config import load
cfg = load()

r = reproduction.init("framepack", repo_url="https://github.com/lllyasviel/FramePack")

# --- STEP: build. FIRST dry_run, print the sbatch for human approval, then run. ---
state, log = reproduction.build_container(
    r, base="docker://nvcr.io#nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
    build_script=open("build.sh").read(), dry_run=True)
print(state); print(log)         # <-- show user, get approval, then re-run with dry_run=False
```

For polling long background jobs from a Claude Code/Codex turn: launch the driver with the Bash tool
`run_in_background`, then `Read` its output file; or have the driver itself `print` each `(state,
log)` and exit. Always surface the log tail + any `metrics.json`.

---

## Critical conventions & gotchas (do NOT skip)

These are distilled from a real reproduction. Each one cost a failed job to learn.

1. **Containers: pull from NGC (`nvcr.io`), not Docker Hub.** This cluster's enroot cannot parse
   Docker Hub's `nvidia/cuda` OCI manifest — it dies at `Fetching image manifest → Could not process
   JSON input → curl (23)` on every node. NGC's own manifests parse fine. The NGC API key lives in
   `~/.config/enroot/.credentials` (both `machine nvcr.io` AND `machine authn.nvidia.com` lines, same
   key) and **expires** — if you get `403 proxy_auth`, ask the user to regenerate it at ngc.nvidia.com.
2. **Unset the proxy for any download** on the SuperPod (model weights, pip, apt). The build/run
   sbatch templates already `unset http_proxy https_proxy …`; keep it. (Hard user rule.)
3. **Faithful build = cuda *devel* base → apt python → pip torch cu126 → requirements.** Do **not**
   append `pip install -U "huggingface_hub[cli]"` — it pulls hf_hub ≥1.0 which transformers <4.5x
   forbids; the import-verify then fails. If you need a specific hf_hub, pin `huggingface_hub<1.0`.
4. **cuda devel base lacks OpenGL/GLib** that OpenCV needs → `import cv2: libGL.so.1: cannot open`.
   apt-install `libgl1 libglib2.0-0 libsm6 libxext6 libxrender1` in the build script.
5. **Offline model loading.** Pre-download weights once with `huggingface_hub.snapshot_download`
   (proxy unset) into `<workdir>/hf-cache` (set `HF_HOME` to it). Then **in the inference script**
   resolve local dirs: `path = snapshot_download(repo_id, local_files_only=True)` and pass `path` to
   `from_pretrained` — because diffusers' sharded-checkpoint loader calls `model_info()` (a network
   call) even when cached, which raises under `HF_HUB_OFFLINE=1`. Passing a local dir skips it.
6. **torch/torchvision API drift.** FramePack's README install is *unpinned* (`pip install torch
   …/cu126`) → today it resolves to torch 2.12 / torchvision 0.23, which **removed
   `torchvision.io.write_video`**. Either pin to the paper's era (e.g. torch 2.6 / torchvision 0.21)
   for fidelity, or monkeypatch a PyAV-based `write_video` shim. State which you chose in the report.
7. **To reproduce a low-/constant-memory claim on a big GPU, FORCE the low-VRAM path.** Demos
   auto-select a high-VRAM mode when lots of memory is free (e.g. on an 80 GB H800) and will *not*
   exercise the offload mechanism. The **length-invariance** of peak memory is the GPU-independent
   claim and reproduces directly; the literal small number (e.g. "6 GB") needs artificial memory
   pressure and is a separate experiment.
8. **Measurement hygiene.** `torch.cuda.empty_cache()` + `reset_peak_memory_stats()` before each run;
   read `max_memory_allocated` / `max_memory_reserved` after. The **first** point in a sweep absorbs
   cold-start (CUDA/cuDNN autotune, first weight swap) — warm up or discount it, or total time looks
   non-monotonic.
9. **Partitions / QOS.** `cpu` partition: 8 cpu/node cap, **no GPU**, no interactive `srun` — use it
   only for host-side downloads. Build containers and run GPU work on `preempt` (or `normal`) with
   `--gpus`. Account is `visworld01`.
10. **`--container-save` writes the image even when the in-container command FAILS** (exit code is
    ignored). So a broken build still leaves a `.sqsh` you can *patch* (pass it as `base` to
    `build_container` with a fix script) instead of rebuilding from scratch.

---

## Deliverables (what "done" looks like)

In `<workdir>/`:
- `report.md` — sections: **Objective** (claims tested) · **Results** (a numbers table) · **Verdict**
  (which claims reproduced + evidence + caveats) · **Anomalies / brainstorm seeds** (load-bearing or
  under-justified mechanisms worth perturbing — REQUIRED).
- `outputs/metrics.json` + artifacts (videos/frames/logs).
- The build + experiment scripts (`build.sh`, `*.sbatch`, the instrumented driver) so the run is
  reproducible.

`reproduction.summarize(...)` will draft the report from collected results; review and tighten it.

---

## Worked example — FramePack (use as a template)

A complete reproduction lives at **`/project/visworld01/repro-framepack/`**: `build.sh`,
`build_stage2.sbatch`, `infer_headless.py` (a faithful, instrumented port of the repo's demo worker),
`report.md`, `metrics.json`, and `brainstorm_ideas.txt`. It confirmed FramePack's headline claim —
peak VRAM flat (~33.6 GB reserved) across 1 s→5 s (37→145 frames) on one H800 — and the report's
anomaly hunt produced 5 grounded research ideas. Copy its shape: instrument the repo's own inference
path (don't reimplement the model), sweep the variable the paper claims invariance over, and measure.

---

## Optional next loop — ideas from the findings

To turn the report's anomalies into research ideas, call
`paper_tracker.brainstorm_agent.generate_ideas(task_prompt, manifest=[card], data_dir=..., topic_id=...,
cfg=...)`. It's a tool-using agent that searches arXiv + the local library/KB for prior work and
returns ideas grounded against it (mechanism + a lightweight experiment each). Feed the anomalies as
`task_prompt` and a card for the paper as the single-item `manifest`.
