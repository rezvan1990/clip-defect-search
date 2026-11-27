"""Pluggable defect alerts.

A defect always lands in the SQLite alert table and in alerts.jsonl.
Optional channels, all configured by environment variables:

- DEFECT_WEBHOOK_URL: POST the event JSON
- SLACK_WEBHOOK_URL: POST a Slack incoming-webhook body
- ALERT_EMAIL_TO plus ALERT_SMTP_HOST: send one plain-text email
  (ALERT_EMAIL_FROM, ALERT_SMTP_PORT, ALERT_SMTP_USER, ALERT_SMTP_PASSWORD,
  ALERT_SMTP_STARTTLS)

If a channel errors, the others still run and the error is stored on the alert.
The log write is the fallback and does not depend on those variables.
"""

from __future__ import annotations

import json
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from urllib import error, request

from defectclip.inspection_log import InspectionLog, utc_now


def decision_for(score: float, threshold: float | None) -> str:
    if threshold is None:
        return "unscored"
    if score >= threshold:
        return "defect"
    return "pass"


class AlertRouter:
    def __init__(self, log: InspectionLog, jsonl_path: Path | str, env: dict | None = None, timeout: float = 5.0):
        self.log = log
        self.jsonl_path = Path(jsonl_path)
        self.env = env if env is not None else dict(os.environ)
        self.timeout = timeout

    def send(self, event: dict, inspection_id: int | None = None) -> dict:
        payload = dict(event)
        payload.setdefault("created_at", utc_now())
        channels = ["log"]
        errors: list[str] = []
        self._append_jsonl(payload)
        webhook = self.env.get("DEFECT_WEBHOOK_URL", "").strip()
        if webhook:
            self._try("webhook", channels, errors, lambda: self._post_json(webhook, payload))
        slack = self.env.get("SLACK_WEBHOOK_URL", "").strip()
        if slack:
            text = (
                f"QC defect: {payload.get('category')} "
                f"score {payload.get('score')} "
                f"(threshold {payload.get('threshold')}) "
                f"image {payload.get('image_name')}"
            )
            self._try("slack", channels, errors, lambda: self._post_json(slack, {"text": text}))
        if self.env.get("ALERT_EMAIL_TO", "").strip() and self.env.get("ALERT_SMTP_HOST", "").strip():
            self._try("email", channels, errors, lambda: self._send_email(payload))
        alert_id = self.log.add_alert(
            inspection_id=inspection_id,
            category=str(payload.get("category", "")),
            score=payload.get("score"),
            threshold=payload.get("threshold"),
            image_name=str(payload.get("image_name", "")),
            channels=channels,
            errors=errors,
            payload=payload,
        )
        return {"alerted": True, "alert_id": alert_id, "channels": channels, "errors": errors}

    def _try(self, name: str, channels: list[str], errors: list[str], fn) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — channel failures must not drop the log
            errors.append(f"{name}: {exc}")
        else:
            channels.append(name)

    def _append_jsonl(self, payload: dict) -> None:
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _post_json(self, url: str, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                if response.status >= 300:
                    raise RuntimeError(f"HTTP {response.status} from {url}")
        except error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code} from {url}") from exc

    def _send_email(self, payload: dict) -> None:
        host = self.env["ALERT_SMTP_HOST"].strip()
        port = int(self.env.get("ALERT_SMTP_PORT", "587") or "587")
        sender = self.env.get("ALERT_EMAIL_FROM", "").strip() or "qc@localhost"
        recipient = self.env["ALERT_EMAIL_TO"].strip()
        message = EmailMessage()
        message["Subject"] = f"QC defect alert: {payload.get('category')}"
        message["From"] = sender
        message["To"] = recipient
        message.set_content(json.dumps(payload, indent=2))
        with smtplib.SMTP(host, port, timeout=self.timeout) as smtp:
            if self.env.get("ALERT_SMTP_STARTTLS", "").strip() in {"1", "true", "True"}:
                smtp.starttls()
            user = self.env.get("ALERT_SMTP_USER", "").strip()
            password = self.env.get("ALERT_SMTP_PASSWORD", "")
            if user:
                smtp.login(user, password)
            smtp.send_message(message)
