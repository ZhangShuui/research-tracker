"""Shared CLI-based LLM invocation utility.

Provides a single call_cli() used by summarizer, report, insights, and brainstorm modules.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time

log = logging.getLogger(__name__)


def _run_with_idle_timeout(
    cmd: list[str],
    prompt: str,
    *,
    idle_timeout: int,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run a CLI command with activity-based (idle) timeout.

    Instead of killing the process after a fixed wall-clock duration, we monitor
    stdout/stderr for ongoing output.  As long as the CLI keeps producing bytes
    (i.e. it's still streaming tokens), the timer resets.  We only kill the
    process when it goes silent for *idle_timeout* seconds.
    """
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    # Write prompt to stdin and close
    try:
        proc.stdin.write(prompt.encode())
        proc.stdin.close()
    except OSError:
        pass

    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    last_activity = time.monotonic()
    lock = threading.Lock()

    def _reader(fd: int, chunks: list[bytes]) -> None:
        nonlocal last_activity
        while True:
            try:
                data = os.read(fd, 8192)
            except OSError:
                break
            if not data:
                break
            with lock:
                chunks.append(data)
                last_activity = time.monotonic()

    t_out = threading.Thread(
        target=_reader, args=(proc.stdout.fileno(), stdout_chunks), daemon=True,
    )
    t_err = threading.Thread(
        target=_reader, args=(proc.stderr.fileno(), stderr_chunks), daemon=True,
    )
    t_out.start()
    t_err.start()

    timed_out = False
    while True:
        t_out.join(timeout=5.0)
        if not t_out.is_alive():
            break
        with lock:
            idle = time.monotonic() - last_activity
        if idle > idle_timeout:
            proc.kill()
            timed_out = True
            break

    t_out.join(timeout=5)
    t_err.join(timeout=5)
    proc.wait(timeout=10)

    if timed_out:
        raise subprocess.TimeoutExpired(cmd, idle_timeout)

    stdout = b"".join(stdout_chunks).decode(errors="replace")
    stderr = b"".join(stderr_chunks).decode(errors="replace")
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def call_cli(
    prompt: str,
    cfg: dict,
    *,
    model: str | None = None,
    timeout: int | None = None,
) -> str | None:
    """Call Claude CLI (or Codex fallback) with the given prompt.

    Args:
        prompt: The full prompt text.
        cfg: Pipeline config dict (needs cfg["summarizer"] for CLI paths).
        model: Override the Claude model (default from config).
        timeout: Idle timeout — max seconds of silence before giving up.

    Returns:
        Raw stdout string, or None on failure.
    """
    scfg = cfg.get("summarizer", {})

    # --- Claude CLI ---
    claude_path = scfg.get("claude_path", "claude")
    claude_model = model or scfg.get("claude_model", "opus")
    _timeout = timeout or max(scfg.get("claude_timeout", 120), 120)

    env = os.environ.copy()
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    env.pop("CLAUDECODE", None)

    try:
        cmd = [claude_path, "-p", "-"]
        if claude_model:
            cmd.extend(["--model", claude_model])
        result = _run_with_idle_timeout(
            cmd, prompt, idle_timeout=_timeout, env=env,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        log.warning("Claude CLI returned code %d, stderr: %s",
                    result.returncode, (result.stderr or "")[:500])
    except FileNotFoundError:
        log.warning("Claude CLI not found at '%s'", claude_path)
    except subprocess.TimeoutExpired:
        log.warning("Claude CLI idle timeout after %ds (no output)", _timeout)

    # --- Codex CLI fallback ---
    codex_path = scfg.get("codex_path", "codex")
    codex_timeout = timeout or max(scfg.get("codex_timeout", 120), 120)

    env2 = os.environ.copy()
    env2.pop("OPENAI_API_KEY", None)
    env2.pop("OPENAI_BASE_URL", None)

    try:
        result = _run_with_idle_timeout(
            [codex_path, "exec", prompt],
            "",  # codex exec takes prompt as arg, not stdin
            idle_timeout=codex_timeout,
            env=env2,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        log.warning("Codex CLI returned code %d", result.returncode)
    except FileNotFoundError:
        log.warning("Codex CLI not found at '%s'", codex_path)
    except subprocess.TimeoutExpired:
        log.warning("Codex CLI idle timeout after %ds (no output)", codex_timeout)

    return None


def call_codex(
    prompt: str,
    cfg: dict,
    *,
    timeout: int | None = None,
) -> str | None:
    """Call Codex CLI directly. No fallback — returns None on failure.

    Args:
        prompt: The full prompt text (passed as positional arg to ``codex exec``).
        cfg: Pipeline config dict (needs cfg["summarizer"] for CLI paths).
        timeout: Idle timeout in seconds.

    Returns:
        Raw stdout string, or None on failure.
    """
    scfg = cfg.get("summarizer", {})
    codex_path = scfg.get("codex_path", "codex")
    codex_timeout = timeout or max(scfg.get("codex_timeout", 120), 120)

    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env.pop("OPENAI_BASE_URL", None)

    try:
        result = _run_with_idle_timeout(
            [codex_path, "exec", prompt],
            "",  # codex exec takes prompt as arg, not stdin
            idle_timeout=codex_timeout,
            env=env,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        log.warning("Codex CLI returned code %d", result.returncode)
    except FileNotFoundError:
        log.warning("Codex CLI not found at '%s'", codex_path)
    except subprocess.TimeoutExpired:
        log.warning("Codex CLI idle timeout after %ds", codex_timeout)

    return None


def _strip_copilot_stats(text: str) -> str:
    """Remove trailing Copilot usage stats block from output.

    Copilot appends lines like:
        Total usage est:  1 Premium request
        API time spent:   12s
        ...
    Even with ``-s`` some versions still emit them.
    """
    # Find the first "Total usage est:" line and cut everything from there
    marker = re.search(r"\n\s*Total usage est:", text)
    if marker:
        return text[: marker.start()].strip()
    return text


def call_copilot(
    prompt: str,
    cfg: dict,
    *,
    timeout: int | None = None,
) -> str | None:
    """Call GitHub Copilot CLI (``copilot -p ... -s``). No fallback.

    Uses ``-s`` (silent) to suppress usage stats in output.

    Args:
        prompt: The full prompt text.
        cfg: Pipeline config dict (needs cfg["summarizer"] for CLI paths).
        timeout: Idle timeout in seconds.

    Returns:
        Raw stdout string, or None on failure.
    """
    scfg = cfg.get("summarizer", {})
    copilot_path = scfg.get("copilot_path", "copilot")
    copilot_timeout = timeout or max(scfg.get("copilot_timeout", 300), 120)
    copilot_model = scfg.get("copilot_model", "")

    cmd = [copilot_path, "-p", prompt, "-s"]
    if copilot_model:
        cmd.extend(["--model", copilot_model])

    try:
        result = _run_with_idle_timeout(
            cmd,
            "",  # copilot -p takes prompt as arg, not stdin
            idle_timeout=copilot_timeout,
        )
        if result.returncode == 0 and result.stdout.strip():
            return _strip_copilot_stats(result.stdout.strip())
        log.warning("Copilot CLI returned code %d, stderr: %s",
                    result.returncode, (result.stderr or "")[:500])
    except FileNotFoundError:
        log.warning("Copilot CLI not found at '%s'", copilot_path)
    except subprocess.TimeoutExpired:
        log.warning("Copilot CLI idle timeout after %ds", copilot_timeout)

    return None
