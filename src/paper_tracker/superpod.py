"""Smooth, multiplexed SSH channel to HKUST SuperPod (foundation for the experiment runner).

Uses an app-owned SSH ControlMaster socket so repeated commands (submit / poll
``sacct`` / fetch results) reuse ONE persistent connection — no repeated login.
SLURM commands are wrapped in a LOGIN shell (``bash -lc``) because the ``module``
function (and hence sbatch/squeue/sacct) is NOT available in a plain non-interactive
``ssh host 'cmd'``.

Verified live (2026-06-11): cold connect ~1.3s, warm reuse ~0.2s; ``bash -lc
"module load slurm && ..."`` exposes ``/cm/shared/apps/slurm/current/bin/{sbatch,
squeue,sacct}``. The app uses its own ControlPath (``~/.ssh/cm/pt-*``) so it never
interferes with the user's interactive ``spod``/tmux sessions.
"""

from __future__ import annotations

import functools
import re
import shlex
import subprocess
import time
from pathlib import Path

# Defaults — overridable via config.toml [superpod] (host / project_root).
HOST = "superpod"                       # ~/.ssh/config alias
PROJECT_ROOT = "/project/visworld01"    # default work-dir root


@functools.lru_cache(maxsize=1)
def _conf() -> dict:
    """Cached ``[superpod]`` config section (host / project_root). {} on any failure."""
    try:
        from paper_tracker.config import load
        return load().get("superpod", {}) or {}
    except Exception:
        return {}


def _host() -> str:
    return _conf().get("host") or HOST


def _project_root() -> str:
    return _conf().get("project_root") or PROJECT_ROOT


_CM_DIR = Path.home() / ".ssh" / "cm"
# App-owned control socket — distinct from spod's /tmp/spod-ssh-* so we don't
# disturb the user's interactive sessions.
_CONTROL_PATH = str(_CM_DIR / "pt-%r@%h:%p")

_SSH_OPTS = [
    "-o", "ControlMaster=auto",
    "-o", f"ControlPath={_CONTROL_PATH}",
    "-o", "ControlPersist=10m",          # keep the master warm 10 min after last use
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=4",
    "-o", "BatchMode=yes",               # never prompt; key-based (MFA is at the VPN layer)
    "-o", "ConnectTimeout=15",
]


def _remote(cmd: str, *, login_shell: bool) -> str:
    """Wrap a remote command in a login shell when the SLURM/module env is needed."""
    return ("bash -lc " + shlex.quote(cmd)) if login_shell else cmd


def run(remote_cmd: str, *, login_shell: bool = True, timeout: int = 60,
        input: str | None = None) -> subprocess.CompletedProcess:
    """Run a command on SuperPod over the multiplexed channel.

    ``login_shell=True`` wraps the command in ``bash -lc`` so ``module`` / SLURM
    tools resolve. The first call opens the master (~1.3s); later calls reuse it
    (~0.2s) — no repeated login. ``input`` is piped to the remote command's stdin
    (used by :func:`write_remote`).
    """
    _CM_DIR.mkdir(parents=True, exist_ok=True)
    argv = ["ssh", *_SSH_OPTS, _host(), _remote(remote_cmd, login_shell=login_shell)]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, input=input)


def slurm(cmd: str, *, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a SLURM command (loads the slurm module first, in a login shell)."""
    return run(f"module load slurm && {cmd}", login_shell=True, timeout=timeout)


def master_alive() -> bool:
    """True if the persistent control master is currently up."""
    p = subprocess.run(
        ["ssh", "-O", "check", "-o", f"ControlPath={_CONTROL_PATH}", _host()],
        capture_output=True, text=True,
    )
    return p.returncode == 0


def close() -> None:
    """Close the control master (e.g., on app shutdown)."""
    subprocess.run(
        ["ssh", "-O", "exit", "-o", f"ControlPath={_CONTROL_PATH}", _host()],
        capture_output=True, text=True,
    )


def reachable(timeout: int = 8) -> bool:
    """Quick reachability probe (opens or reuses the master). False if VPN/cluster is down."""
    try:
        return run("true", login_shell=False, timeout=timeout).returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# SOP primitives: workdir / files / sbatch submit -> poll -> fetch
# (the execution substrate every reproduce-SOP compute step is built on)
# ---------------------------------------------------------------------------

_JOBID_RE = re.compile(r"Submitted batch job (\d+)")
_TERMINAL_STATES = frozenset({
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED",
})


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9._-]+", "-", (name or "").strip().lower()).strip("-")
    return s or "work"


def make_workdir(name: str, *, project_root: str | None = None) -> str:
    """Create ``<project_root>/repro-<slug>/`` and return its absolute path.

    ``project_root`` defaults to config ``[superpod].project_root`` (then the
    module default). Pass it explicitly to override per reproduction.
    """
    root = project_root or _project_root()
    path = f"{root}/repro-{_slug(name)}"
    r = run(f"mkdir -p {shlex.quote(path)} && echo {shlex.quote(path)}", login_shell=False)
    if r.returncode != 0:
        raise RuntimeError(f"make_workdir failed: {r.stderr.strip()}")
    return r.stdout.strip() or path


def write_remote(path: str, content: str) -> None:
    """Write text to a remote file (e.g. an sbatch script) over the multiplexed channel."""
    r = run(f"cat > {shlex.quote(path)}", login_shell=False, input=content)
    if r.returncode != 0:
        raise RuntimeError(f"write_remote {path} failed: {r.stderr.strip()}")


def fetch(path: str, *, timeout: int = 60) -> str:
    """Return the contents of a remote file (result.json, slurm-*.out, ...)."""
    return run(f"cat {shlex.quote(path)}", login_shell=False, timeout=timeout).stdout


def submit(script_path: str, *, workdir: str | None = None, timeout: int = 60) -> str:
    """``sbatch`` a script that already exists remotely; return the SLURM job id."""
    cd = f"cd {shlex.quote(workdir)} && " if workdir else ""
    r = slurm(f"{cd}sbatch {shlex.quote(script_path)}", timeout=timeout)
    m = _JOBID_RE.search(r.stdout or "")
    if not m:
        raise RuntimeError(f"sbatch returned no job id: {(r.stdout or '')!r} / {(r.stderr or '')!r}")
    return m.group(1)


def job_state(job_id: str, *, timeout: int = 30) -> str:
    """Current SLURM state of a job via ``sacct`` ('' if not yet recorded)."""
    r = slurm(f"sacct -j {shlex.quote(job_id)} -n -X -o State%-30", timeout=timeout)
    lines = (r.stdout or "").strip().splitlines()
    return lines[0].split()[0] if lines and lines[0].split() else ""


def wait(job_id: str, *, poll: int = 15, timeout: int = 3600) -> str:
    """Poll ``sacct`` until the job reaches a terminal state (or ``timeout``).

    Returns the final state (COMPLETED / FAILED / TIMEOUT / ...). Raises on timeout.
    """
    waited, st = 0, ""
    while waited <= timeout:
        st = job_state(job_id)
        if st in _TERMINAL_STATES:
            return st
        time.sleep(poll)
        waited += poll
    raise TimeoutError(f"job {job_id} unfinished after {timeout}s (last state {st!r})")
