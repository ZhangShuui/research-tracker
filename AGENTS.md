# Agent guide — paper-tracker

Multi-topic research dashboard + brainstorm partner (FastAPI backend, port 8000 + Next.js frontend).

## Reproducing a paper on the HKUST SuperPod
If you are asked to **reproduce a paper** on the SuperPod, read and follow
**[docs/reproducing-papers.md](docs/reproducing-papers.md)**. It defines:
- the fixed **4-stage SOP** (workdir+clone → build container → step-wise experiment sbatch → report);
- the modules to use — `paper_tracker.reproduction` (orchestration) on top of
  `paper_tracker.superpod` (the SSH/SLURM channel). **Do not ad-hoc `ssh`**; use these primitives;
- the **human-in-the-loop rule**: call each compute step with `dry_run=True` first, show the user the
  generated `sbatch`, get approval, then run;
- the cluster **gotchas** that otherwise each cost a failed job (NGC-not-Docker-Hub, unset proxy,
  offline model loading, cv2 `libGL`, torch/torchvision API drift, forcing the low-VRAM path, …).

## Dev
- Backend tests: `uv run pytest -q`
- Backend server: `uv run uvicorn paper_tracker.server:app --reload --port 8000`
- Frontend: `cd frontend && npm run dev`
- Python 3.14 + `uv` (the project venv); minimum LLM model is sonnet, prefer opus for important tasks.
