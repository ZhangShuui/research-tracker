"""Windows Toast notification via native WinRT XML through PowerShell."""

from __future__ import annotations

import logging
import subprocess

log = logging.getLogger(__name__)

# Native WinRT XML template — no BurntToast dependency required
_PS_TEMPLATE = r"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType = WindowsRuntime] | Out-Null

$xml = @"
<toast>
  <visual>
    <binding template="ToastGeneric">
      <text>{title}</text>
      <text>{body}</text>
    </binding>
  </visual>
</toast>
"@

$doc = New-Object Windows.Data.Xml.Dom.XmlDocument
$doc.LoadXml($xml)
$toast = [Windows.UI.Notifications.ToastNotification]::new($doc)
$notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("PaperTracker")
$notifier.Show($toast)
"""


def notify(title: str, body: str, cfg: dict) -> bool:
    """Show a Windows toast notification. Returns True on success."""
    toast_cfg = cfg.get("notify", {}).get("toast", {})
    if not toast_cfg.get("enabled", True):
        log.info("Toast notifications disabled")
        return True  # treat as success (not an error)

    ps_path = toast_cfg.get("powershell_path", "powershell.exe")

    # Escape for XML embedding
    safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe_body = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    script = _PS_TEMPLATE.format(title=safe_title, body=safe_body)

    try:
        result = subprocess.run(
            [ps_path, "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            log.info("Toast notification sent")
            return True
        log.error("Toast failed (rc=%d): %s", result.returncode, result.stderr.strip())
    except FileNotFoundError:
        log.error("PowerShell not found at '%s'", ps_path)
    except subprocess.TimeoutExpired:
        log.error("Toast notification timed out")
    return False
