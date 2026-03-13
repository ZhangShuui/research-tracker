"""Send email notification via SMTP."""

from __future__ import annotations

import logging
import smtplib
from email.mime.text import MIMEText
from pathlib import Path

log = logging.getLogger(__name__)


def notify(report_path: Path | None, papers: list[dict], repos: list[dict], cfg: dict) -> bool:
    """Send an email with the daily summary. Returns True on success."""
    email_cfg = cfg.get("notify", {}).get("email", {})
    if not email_cfg.get("enabled", False):
        log.info("Email notifications disabled")
        return True  # treat disabled as success

    host = email_cfg["smtp_host"]
    port = email_cfg.get("smtp_port", 587)
    use_tls = email_cfg.get("use_tls", True)
    username = email_cfg["username"]
    password = email_cfg["password"]
    from_addr = email_cfg["from_addr"]
    to_addr = email_cfg["to_addr"]

    subject = f"Paper Tracker: {len(papers)} papers, {len(repos)} repos"

    if report_path and report_path.exists():
        body = report_path.read_text(encoding="utf-8")
    else:
        body = f"Found {len(papers)} new papers and {len(repos)} new repos. No report file generated."

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr

    try:
        if use_tls:
            server = smtplib.SMTP(host, port, timeout=30)
            server.starttls()
        else:
            server = smtplib.SMTP(host, port, timeout=30)
        server.login(username, password)
        server.sendmail(from_addr, [to_addr], msg.as_string())
        server.quit()
        log.info("Email sent to %s", to_addr)
        return True
    except Exception as e:
        log.error("Email notification failed: %s", e)
        return False
