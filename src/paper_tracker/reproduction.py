"""4-stage reproduction orchestration on the SuperPod.

Codifies the user-defined SOP for reproducing a paper, built on the
``superpod.py`` primitives (``make_workdir`` / ``write_remote`` / ``submit`` /
``wait`` / ``fetch``):

  1. ``init``            -> dedicated workdir + clone official code
  2. ``build_container`` -> build/patch an enroot container from a base image
  3. ``run_step``        -> write & submit an experiment sbatch, step by step
  4. ``collect`` + ``summarize`` -> gather results and write a report

Every compute step (``build_container`` / ``run_step``) takes ``dry_run`` so a
caller (a human-in-the-loop user or an agent) can PROPOSE the generated sbatch,
get approval, then run it. Each call submits one job and blocks until it reaches
a terminal SLURM state, returning ``(state, log)``.
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass, field

from paper_tracker import superpod
from paper_tracker.llm import call_cli


@dataclass
class Reproduction:
    name: str
    workdir: str
    repo_url: str | None = None
    # step_name -> {"job_id": str, "state": str}
    jobs: dict = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return superpod._slug(self.name)

    @property
    def logs_dir(self) -> str:
        return f"{self.workdir}/logs"

    @property
    def outputs_dir(self) -> str:
        return f"{self.workdir}/outputs"


# ----------------------------------------------------------------------------
# Stage 1 — workdir + clone
# ----------------------------------------------------------------------------
def init(name: str, *, repo_url: str | None = None,
         project_root: str | None = None) -> Reproduction:
    """Create ``<root>/repro-<slug>/`` (+ logs/ outputs/) and clone the repo if given."""
    workdir = superpod.make_workdir(name, project_root=project_root)
    r = superpod.run(
        f"mkdir -p {shlex.quote(workdir + '/logs')} {shlex.quote(workdir + '/outputs')}",
        login_shell=False)
    if r.returncode != 0:
        raise RuntimeError(f"init: mkdir failed: {r.stderr.strip()}")
    repro = Reproduction(name=name, workdir=workdir, repo_url=repo_url)
    if repo_url:
        c = superpod.run(
            f"cd {shlex.quote(workdir)} && git clone --depth 1 {shlex.quote(repo_url)}",
            timeout=900)
        if c.returncode != 0:
            raise RuntimeError(f"init: git clone failed: {c.stderr.strip()}")
    return repro


# ----------------------------------------------------------------------------
# sbatch helper
# ----------------------------------------------------------------------------
_PROXY_UNSET = ("unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY "
                "ALL_PROXY no_proxy NO_PROXY")


def _sbatch_header(job_name: str, logs_dir: str, *, account: str = "visworld01",
                   partition: str = "preempt", gpus: int = 1, cpus: int = 8,
                   mem: str = "64G", time: str = "01:00:00") -> str:
    return (
        "#!/bin/bash\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --account={account}\n"
        f"#SBATCH --partition={partition}\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --gpus={gpus}\n"
        f"#SBATCH --cpus-per-task={cpus}\n"
        f"#SBATCH --mem={mem}\n"
        f"#SBATCH --time={time}\n"
        f"#SBATCH --output={logs_dir}/{job_name}-%j.out\n"
        "set -eo pipefail\n"
        f"{_PROXY_UNSET}\n"
    )


def _submit_and_wait(repro: Reproduction, key: str, script_path: str, job_name: str,
                     *, wait_timeout: int) -> tuple[str, str]:
    job_id = superpod.submit(script_path, workdir=repro.workdir)
    state = superpod.wait(job_id, timeout=wait_timeout)
    repro.jobs[key] = {"job_id": job_id, "state": state}
    try:
        log = superpod.fetch(f"{repro.logs_dir}/{job_name}-{job_id}.out")
    except Exception as e:  # pragma: no cover - log is best-effort
        log = f"(log fetch failed: {e})"
    return state, log


# ----------------------------------------------------------------------------
# Stage 2 — build / patch container
# ----------------------------------------------------------------------------
def build_container(repro: Reproduction, *, base: str, build_script: str,
                    save_as: str | None = None, mounts: str | None = None,
                    remap_root: bool = True, writable: bool = True,
                    partition: str = "preempt", gpus: int = 1, cpus: int = 8,
                    mem: str = "64G", time: str = "01:00:00",
                    wait_timeout: int = 5400, dry_run: bool = False) -> tuple[str, str]:
    """Build or patch an enroot container.

    ``base`` is an NGC docker URI (``docker://nvcr.io#...``) for a fresh build, or
    a local ``.sqsh`` path to patch an existing image. The container is saved to a
    ``.tmp`` then atomically moved onto ``save_as`` (default ``<workdir>/<slug>.sqsh``)
    so a failed build never clobbers a good image.

    Returns ``(state, log)``, or ``("DRY_RUN", sbatch_text)`` when ``dry_run``.
    """
    wd = repro.workdir
    save_as = save_as or f"{wd}/{repro.slug}.sqsh"
    tmp = f"{save_as}.tmp"
    mounts = mounts or f"{wd}:{wd}"
    job_name = f"{repro.slug}-build"

    superpod.write_remote(f"{wd}/_build.sh", build_script)

    extra = []
    if remap_root:
        extra.append("--container-remap-root")
    if writable:
        extra.append("--container-writable")
    srun = (
        f"srun --container-image {shlex.quote(base)} --container-save {shlex.quote(tmp)} "
        f"--no-container-mount-home --container-mounts {shlex.quote(mounts)} "
        f"--container-workdir {shlex.quote(wd)} {' '.join(extra)} "
        f"bash -l {shlex.quote(wd + '/_build.sh')}"
    )
    body = f"{srun}\nmv -f {shlex.quote(tmp)} {shlex.quote(save_as)}\necho BUILD_OK\n"
    sbatch = _sbatch_header(job_name, repro.logs_dir, partition=partition, gpus=gpus,
                            cpus=cpus, mem=mem, time=time) + body

    if dry_run:
        return "DRY_RUN", sbatch
    script_path = f"{wd}/_build.sbatch"
    superpod.write_remote(script_path, sbatch)
    return _submit_and_wait(repro, "build", script_path, job_name, wait_timeout=wait_timeout)


# ----------------------------------------------------------------------------
# Stage 3 — run an experiment step
# ----------------------------------------------------------------------------
def run_step(repro: Reproduction, step_name: str, command: str, *,
             container: str | None = None, workdir: str | None = None,
             mounts: str | None = None, remap_root: bool = True,
             partition: str = "preempt", gpus: int = 1, cpus: int = 8,
             mem: str = "96G", time: str = "01:30:00",
             wait_timeout: int = 5400, dry_run: bool = False) -> tuple[str, str]:
    """Write & submit one experiment sbatch.

    If ``container`` (an NGC URI or local ``.sqsh``) is given, ``command`` runs
    inside it via ``srun --container-image``; otherwise it runs on the host node.
    ``command`` is a single shell string. Returns ``(state, log)`` or
    ``("DRY_RUN", sbatch_text)``.
    """
    wd = repro.workdir
    step_slug = superpod._slug(step_name)
    job_name = f"{repro.slug}-{step_slug}"
    cwd = workdir or wd

    if container:
        mounts = mounts or f"{wd}:{wd}"
        extra = "--container-remap-root " if remap_root else ""
        inner = (
            f"srun --container-image {shlex.quote(container)} --no-container-mount-home "
            f"--container-mounts {shlex.quote(mounts)} --container-workdir {shlex.quote(cwd)} "
            f"{extra}bash -lc {shlex.quote(command)}"
        )
    else:
        inner = f"cd {shlex.quote(cwd)} && {command}"
    sbatch = _sbatch_header(job_name, repro.logs_dir, partition=partition, gpus=gpus,
                            cpus=cpus, mem=mem, time=time) + inner + "\n"

    if dry_run:
        return "DRY_RUN", sbatch
    script_path = f"{wd}/_{step_slug}.sbatch"
    superpod.write_remote(script_path, sbatch)
    return _submit_and_wait(repro, step_name, script_path, job_name, wait_timeout=wait_timeout)


# ----------------------------------------------------------------------------
# Stage 4 — collect + summarize
# ----------------------------------------------------------------------------
def collect(repro: Reproduction, *, files=("outputs/metrics.json",)) -> dict:
    """Fetch named result files (relative to the workdir). Missing files are recorded as errors."""
    out: dict[str, str] = {}
    for rel in files:
        try:
            out[rel] = superpod.fetch(f"{repro.workdir}/{rel}")
        except Exception as e:
            out[rel] = f"(fetch failed: {e})"
    return out


_REPORT_PROMPT = """You are writing a concise reproduction report for the paper "{paper}".

Below are the collected experiment results and notes from running the reproduction on an
HKUST SuperPod H800 GPU. Write a focused Markdown report with these sections:
1. Objective — what claims were tested.
2. Results — a table of the key numbers.
3. Verdict — which claims reproduced, with the evidence; flag any caveats.
4. Anomalies / brainstorm seeds — load-bearing or under-justified mechanisms worth perturbing.

Be precise and do not invent numbers beyond what is given.

=== RESULTS / NOTES ===
{body}
"""


def summarize(repro: Reproduction, *, cfg: dict, paper: str, collected: dict,
              notes: str = "", model: str = "opus", timeout: int = 600) -> str | None:
    """LLM-summarize the collected results into a Markdown report (Stage 4)."""
    body = notes + "\n\n" + "\n\n".join(
        f"--- {name} ---\n{content}" for name, content in collected.items())
    prompt = _REPORT_PROMPT.format(paper=paper, body=body)
    return call_cli(prompt, cfg, model=model, timeout=timeout)


def write_report(repro: Reproduction, content: str, *, filename: str = "report.md") -> str:
    """Write a report file into the workdir; return its remote path."""
    path = f"{repro.workdir}/{filename}"
    superpod.write_remote(path, content)
    return path
