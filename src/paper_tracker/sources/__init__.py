"""Paper-tracker source modules."""

from __future__ import annotations

import logging
import time

import httpx

log = logging.getLogger(__name__)


def httpx_get_with_retry(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 30,
    retries: int = 3,
    backoff: float = 2.0,
    **kwargs,
) -> httpx.Response:
    """httpx.get with automatic retry on transient errors (SSL, connect, timeout).

    Raises the last exception if all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = httpx.get(
                url, params=params, headers=headers,
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
            # Don't retry on 4xx client errors (except 429 rate limit)
            if e.response.status_code == 429 and attempt < retries:
                wait = backoff * attempt
                log.warning("Rate limited on %s, retrying in %.1fs", url, wait)
                time.sleep(wait)
                last_exc = e
            else:
                raise
    raise last_exc  # type: ignore[misc]
