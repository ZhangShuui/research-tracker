"""Fetch CLI tool usage data from Claude Code, Codex, and GitHub Copilot."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

# Simple time-based cache to avoid hitting rate limits
_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 120  # seconds


def _cached(key: str) -> dict | None:
    if key in _cache:
        ts, data = _cache[key]
        if time.time() - ts < _CACHE_TTL:
            return data
    return None


def _set_cache(key: str, data: dict) -> None:
    _cache[key] = (time.time(), data)


# ---------------------------------------------------------------------------
# Claude Code — OAuth usage API
# ---------------------------------------------------------------------------

def get_claude_usage() -> dict:
    """Fetch Claude Code usage from Anthropic OAuth API.

    Reads access token from ~/.claude/.credentials.json.
    Returns 5-hour, 7-day, and per-model utilization percentages.
    """
    cached = _cached("claude")
    if cached:
        return cached

    result: dict = {"service": "claude", "status": "ok", "limits": [], "error": ""}

    creds_path = Path.home() / ".claude" / ".credentials.json"
    if not creds_path.exists():
        result["status"] = "unconfigured"
        result["error"] = "~/.claude/.credentials.json not found"
        return result

    try:
        creds = json.loads(creds_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        result["status"] = "error"
        result["error"] = f"Failed to read credentials: {e}"
        return result

    # Token can be at top level or nested
    token = creds.get("accessToken") or creds.get("claudeAiOauth", {}).get("accessToken")
    if not token:
        result["status"] = "unconfigured"
        result["error"] = "No access token in credentials file"
        return result

    try:
        resp = httpx.get(
            "https://api.anthropic.com/api/oauth/usage",
            headers={
                "Authorization": f"Bearer {token}",
                "anthropic-beta": "oauth-2025-04-20",
                "Content-Type": "application/json",
                "User-Agent": "paper-tracker/1.0",
            },
            timeout=10,
        )
        if resp.status_code == 429:
            result["status"] = "error"
            result["error"] = "Rate limited (429). Try again later."
            return result
        resp.raise_for_status()
    except httpx.HTTPError as e:
        result["status"] = "error"
        result["error"] = f"API request failed: {e}"
        return result

    data = resp.json()

    # Map known keys to display names
    key_names = {
        "five_hour": "5-Hour",
        "seven_day": "7-Day",
        "seven_day_opus": "7-Day Opus",
        "seven_day_sonnet": "7-Day Sonnet",
        "extra_usage": "Extra Usage",
    }

    for key, display_name in key_names.items():
        if key in data and isinstance(data[key], dict):
            util = data[key].get("utilization")
            if util is None:
                continue  # skip entries with no utilization data
            result["limits"].append({
                "name": display_name,
                "utilization": util,
                "resets_at": data[key].get("resets_at", ""),
            })

    # Include plan info if available
    plan = creds.get("subscriptionType", "")
    if plan:
        result["plan"] = plan

    _set_cache("claude", result)
    return result


# ---------------------------------------------------------------------------
# Codex CLI — ChatGPT backend WHAM usage API
# ---------------------------------------------------------------------------

def get_codex_usage() -> dict:
    """Fetch Codex CLI usage from ChatGPT backend API.

    Reads session token from ~/.codex/auth.json.
    Calls the same endpoint that /status uses internally:
      GET https://chatgpt.com/backend-api/wham/usage
    Returns 5-hour (primary) and weekly (secondary) window utilization.
    """
    cached = _cached("codex")
    if cached:
        return cached

    result: dict = {"service": "codex", "status": "ok", "limits": [], "error": ""}

    auth_path = Path.home() / ".codex" / "auth.json"
    if not auth_path.exists():
        result["status"] = "unconfigured"
        result["error"] = "~/.codex/auth.json not found. Install & login to Codex CLI."
        return result

    try:
        auth = json.loads(auth_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        result["status"] = "error"
        result["error"] = f"Failed to read auth.json: {e}"
        return result

    # Tokens can be at top level or nested under "tokens"
    tokens = auth.get("tokens", auth)
    token = tokens.get("access_token", "")
    account_id = tokens.get("account_id", "")
    if not token:
        result["status"] = "unconfigured"
        result["error"] = "No access_token in ~/.codex/auth.json"
        return result

    # Extract plan from id_token JWT (base64-decoded payload)
    plan = _extract_codex_plan(tokens)
    if plan:
        result["plan"] = plan

    headers: dict[str, str] = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "codex-cli",
    }
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id

    try:
        resp = httpx.get(
            "https://chatgpt.com/backend-api/wham/usage",
            headers=headers,
            timeout=10,
            follow_redirects=True,
        )
        if resp.status_code == 401:
            # Token may be expired, try to hint
            result["status"] = "error"
            result["error"] = "Session token expired. Re-login in Codex CLI."
            return result
        if resp.status_code == 429:
            result["status"] = "error"
            result["error"] = "Rate limited. Try again later."
            return result
        resp.raise_for_status()
    except httpx.HTTPError as e:
        # Fallback: try reading latest local session file for rate limits
        local = _codex_usage_from_local_sessions()
        if local:
            result["limits"] = local
            _set_cache("codex", result)
            return result
        result["status"] = "error"
        result["error"] = f"API request failed: {e}"
        return result

    try:
        data = resp.json()
    except Exception:
        # Fallback to local session files
        local = _codex_usage_from_local_sessions()
        if local:
            result["limits"] = local
            _set_cache("codex", result)
            return result
        result["status"] = "error"
        result["error"] = "Failed to parse API response"
        return result

    rate_limit = data.get("rate_limit") or {}

    # Primary window (5-hour)
    primary = rate_limit.get("primary_window") or {}
    if primary:
        used_pct = primary.get("used_percent", 0)
        reset_at = primary.get("reset_at")
        window_secs = primary.get("limit_window_seconds", 0)
        window_label = _window_label(window_secs)
        entry: dict = {"name": window_label, "utilization": used_pct}
        if reset_at:
            from datetime import datetime, timezone
            entry["resets_at"] = datetime.fromtimestamp(reset_at, tz=timezone.utc).isoformat()
        result["limits"].append(entry)

    # Secondary window (weekly)
    secondary = rate_limit.get("secondary_window") or {}
    if secondary:
        used_pct = secondary.get("used_percent", 0)
        reset_at = secondary.get("reset_at")
        window_secs = secondary.get("limit_window_seconds", 0)
        window_label = _window_label(window_secs)
        entry = {"name": window_label, "utilization": used_pct}
        if reset_at:
            from datetime import datetime, timezone
            entry["resets_at"] = datetime.fromtimestamp(reset_at, tz=timezone.utc).isoformat()
        result["limits"].append(entry)

    # Additional rate limits (e.g. codex_other)
    for extra in (data.get("additional_rate_limits") or []):
        name = extra.get("limit_name", "Other")
        rl = extra.get("rate_limit") or {}
        for wkey in ("primary_window", "secondary_window"):
            wdata = rl.get(wkey)
            if wdata and wdata.get("used_percent", 0) > 0:
                result["limits"].append({
                    "name": f"{name} ({_window_label(wdata.get('limit_window_seconds', 0))})",
                    "utilization": wdata.get("used_percent", 0),
                })

    # Credits info
    credits = data.get("credits") or {}
    if credits.get("has_credits") and credits.get("balance"):
        try:
            balance = float(credits["balance"])
            result["limits"].append({
                "name": "Credits Balance",
                "utilization": -1,
                "value": balance,
                "unit": "USD",
            })
        except (ValueError, TypeError):
            pass

    if not result["limits"]:
        # Fallback to local session data
        local = _codex_usage_from_local_sessions()
        if local:
            result["limits"] = local

    _set_cache("codex", result)
    return result


def _window_label(secs: int) -> str:
    """Convert window seconds to a human label."""
    if secs <= 0:
        return "Window"
    hours = secs / 3600
    if hours <= 6:
        return "5-Hour"
    days = hours / 24
    if days <= 2:
        return "Daily"
    return "Weekly"


def _extract_codex_plan(auth: dict) -> str:
    """Extract plan type from Codex auth JWT id_token."""
    id_token = auth.get("id_token", "")
    if not id_token:
        return ""
    try:
        import base64
        parts = id_token.split(".")
        if len(parts) < 2:
            return ""
        # Add padding
        payload_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return payload.get("chatgpt_plan_type", "")
    except Exception:
        return ""


def _codex_usage_from_local_sessions() -> list[dict]:
    """Fallback: read rate limits from the latest local Codex session file."""
    from datetime import datetime, timezone

    sessions_dir = Path.home() / ".codex" / "sessions"
    if not sessions_dir.exists():
        return []

    # Find the most recent rollout file
    rollout_files = sorted(sessions_dir.rglob("rollout-*.jsonl"), reverse=True)
    if not rollout_files:
        return []

    limits = []
    # Read the last file, scan from end for rate_limits
    try:
        lines = rollout_files[0].read_text().strip().split("\n")
        for line in reversed(lines):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = entry.get("payload", {})
            if payload.get("type") != "token_count":
                continue
            rl = payload.get("rate_limits", {})
            if not rl:
                continue
            for key, label in [("primary", "5-Hour"), ("secondary", "Weekly")]:
                w = rl.get(key, {})
                if w:
                    item: dict = {"name": label, "utilization": w.get("used_percent", 0)}
                    reset_ts = w.get("resets_at")
                    if reset_ts:
                        item["resets_at"] = datetime.fromtimestamp(reset_ts, tz=timezone.utc).isoformat()
                    limits.append(item)
            break  # found the latest rate_limits entry
    except OSError:
        pass

    return limits


# ---------------------------------------------------------------------------
# GitHub Copilot — Internal user API (same as VS Code status bar)
# ---------------------------------------------------------------------------

def get_copilot_usage() -> dict:
    """Fetch GitHub Copilot premium request usage.

    Uses the same internal endpoint that VS Code calls for the status bar:
      GET https://api.github.com/copilot_internal/user
    Works with standard gh CLI auth, no extra scopes needed.
    """
    cached = _cached("copilot")
    if cached:
        return cached

    result: dict = {"service": "copilot", "status": "ok", "limits": [], "error": ""}

    # Call the internal API via gh CLI
    try:
        out = subprocess.run(
            ["gh", "api", "/copilot_internal/user", "--jq", "."],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            stderr = out.stderr.strip()
            if "auth" in stderr.lower() or "login" in stderr.lower():
                result["status"] = "unconfigured"
                result["error"] = "gh CLI not authenticated. Run 'gh auth login'."
            else:
                result["status"] = "error"
                result["error"] = f"API failed: {stderr}"
            return result
    except FileNotFoundError:
        result["status"] = "unconfigured"
        result["error"] = "gh CLI not found. Install GitHub CLI."
        return result
    except subprocess.TimeoutExpired:
        result["status"] = "error"
        result["error"] = "API request timed out"
        return result

    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        result["status"] = "error"
        result["error"] = "Failed to parse API response"
        return result

    # Plan info
    plan = data.get("copilot_plan", "") or data.get("access_type_sku", "")
    if plan:
        result["plan"] = plan

    # Quota reset date
    reset_date = data.get("quota_reset_date", "")

    # Parse quota snapshots
    snapshots = data.get("quota_snapshots", {})
    for key, label in [
        ("premium_interactions", "Premium Requests"),
        ("chat", "Chat"),
        ("completions", "Completions"),
    ]:
        snap = snapshots.get(key, {})
        if not snap:
            continue

        unlimited = snap.get("unlimited", False)
        if unlimited:
            continue  # skip unlimited quotas

        entitlement = snap.get("entitlement", 0)
        remaining = snap.get("remaining", 0)
        pct_remaining = snap.get("percent_remaining", 0)

        if entitlement <= 0:
            continue

        used = entitlement - remaining
        used_pct = round(100 - pct_remaining, 1)

        entry: dict = {
            "name": label,
            "utilization": used_pct,
            "value": int(remaining),
            "unit": f"/ {int(entitlement)} remaining",
        }
        if reset_date:
            entry["resets_at"] = reset_date
        result["limits"].append(entry)

    _set_cache("copilot", result)
    return result


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def get_all_usage() -> list[dict]:
    """Fetch usage from all configured services."""
    results = []
    for fetcher in [get_claude_usage, get_codex_usage, get_copilot_usage]:
        try:
            results.append(fetcher())
        except Exception as e:
            log.error("Usage fetch failed for %s: %s", fetcher.__name__, e)
            service = fetcher.__name__.replace("get_", "").replace("_usage", "")
            results.append({
                "service": service,
                "status": "error",
                "error": str(e),
                "limits": [],
            })
    return results
