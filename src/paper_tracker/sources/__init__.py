"""Paper-tracker source modules."""

from __future__ import annotations

import logging
import random
import time

import httpx

log = logging.getLogger(__name__)

# Polite APIs (arXiv, OpenAlex, ...) ask for a descriptive User-Agent so they can
# identify and contact heavy clients instead of silently blocking them. A bare
# ``python-httpx/x.y`` UA is more likely to get rate-limited.
DEFAULT_USER_AGENT = (
    "paper-tracker/0.1 (research paper tracking bot; "
    "+https://github.com/paper-tracker)"
)

# Statuses we treat as "back off and retry" rather than hard failures.
# arXiv historically threw 503 when throttling; since ~Feb 2026 it returns 429.
_RETRYABLE_STATUS = frozenset({429, 503})


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    """Parse a ``Retry-After`` header in delta-seconds form, if present."""
    val = resp.headers.get("Retry-After")
    if not val:
        return None
    try:
        return max(0.0, float(val))
    except ValueError:
        # HTTP-date form is uncommon here; let the caller fall back to backoff.
        return None


def httpx_get_with_retry(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 30,
    retries: int = 5,
    backoff: float = 3.0,
    **kwargs,
) -> httpx.Response:
    """httpx.get with automatic retry on transient errors and rate limits.

    - Always sends a descriptive ``User-Agent`` (overridable via *headers*).
    - Retries transient network errors (SSL, connect, timeout) with linear backoff.
    - Retries HTTP 429/503 with exponential backoff + jitter, honoring the
      ``Retry-After`` header when the server provides one. The minimum wait is
      3s to respect arXiv's "1 request / 3 seconds" guidance.

    Raises the last exception if all retries are exhausted.
    """
    merged_headers = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        merged_headers.update(headers)

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = httpx.get(
                url, params=params, headers=merged_headers,
                timeout=timeout, follow_redirects=True, **kwargs,
            )
            resp.raise_for_status()
            return resp
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout,
                httpx.PoolTimeout, httpx.ConnectTimeout) as e:
            last_exc = e
            if attempt < retries:
                wait = backoff * attempt
                log.warning(
                    "Transient error on %s (attempt %d/%d), retrying in %.1fs: %s",
                    url, attempt, retries, wait, e,
                )
                time.sleep(wait)
            else:
                raise
        except httpx.HTTPStatusError as e:
            # Retry rate-limit / service-unavailable; never retry other 4xx.
            if e.response.status_code in _RETRYABLE_STATUS and attempt < retries:
                wait = _retry_after_seconds(e.response)
                if wait is None:
                    # Exponential backoff (3, 6, 12, 24, ...) + jitter, min 3s.
                    wait = max(3.0, backoff * (2 ** (attempt - 1))) + random.uniform(0, 1.0)
                log.warning(
                    "Rate limited (HTTP %d) on %s (attempt %d/%d), retrying in %.1fs",
                    e.response.status_code, url, attempt, retries, wait,
                )
                time.sleep(wait)
                last_exc = e
            else:
                raise
    raise last_exc  # type: ignore[misc]
