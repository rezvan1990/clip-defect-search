import json
import tempfile
import unittest
from email.message import EmailMessage
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch

from defectclip.alerts import AlertRouter, decision_for
from defectclip.inspection_log import InspectionLog


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.server.seen.append((self.path, body))
        status = 500 if self.path == "/fail" else 200
        self.send_response(status)
        self.end_headers()

    def log_message(self, fmt, *args):
        return


class LogAndAlertTest(unittest.TestCase):
    def test_decision_boundaries(self):
        self.assertEqual(decision_for(0.2, None), "unscored")
        self.assertEqual(decision_for(0.2, 0.2), "defect")
        self.assertEqual(decision_for(0.2, 0.21), "pass")

    def test_summary_ignores_unscored_in_the_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = InspectionLog(f"{tmp}/qc.db")
            log.add_inspection(
                category="bottle",
                score=0.4,
                decision="defect",
                threshold=0.1,
                image_name="a.png",
                source="api",
                alerted=False,
            )
            log.add_inspection(
                category="bottle",
                score=0.0,
                decision="pass",
                threshold=0.1,
                image_name="b.png",
                source="api",
                alerted=False,
            )
            log.add_inspection(
                category="bottle",
                score=0.05,
                decision="unscored",
                threshold=None,
                image_name="c.png",
                source="api",
                alerted=False,
            )
            summary = log.summary()
            self.assertEqual(summary["n"], 3)
            self.assertEqual(summary["n_unscored"], 1)
            bottle = summary["by_category"][0]
            self.assertEqual(bottle["n_scored"], 2)
            self.assertAlmostEqual(bottle["defect_rate"], 0.5)

    def test_log_fallback_and_webhook_and_failed_slack(self):
        server = HTTPServer(("127.0.0.1", 0), _Handler)
        server.seen = []
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        try:
            with tempfile.TemporaryDirectory() as tmp:
                log = InspectionLog(f"{tmp}/qc.db")
                router = AlertRouter(
                    log,
                    f"{tmp}/alerts.jsonl",
                    env={
                        "DEFECT_WEBHOOK_URL": f"http://{host}:{port}/hook",
                        "SLACK_WEBHOOK_URL": f"http://{host}:{port}/fail",
                    },
                )
                result = router.send(
                    {
                        "event": "defect",
                        "category": "tile",
                        "score": 0.4,
                        "threshold": 0.1,
                        "image_name": "t.png",
                    }
                )
                self.assertTrue(result["alerted"])
                self.assertIn("log", result["channels"])
                self.assertIn("webhook", result["channels"])
                self.assertNotIn("slack", result["channels"])
                self.assertTrue(any(item.startswith("slack:") for item in result["errors"]))
                lines = open(f"{tmp}/alerts.jsonl", encoding="utf-8").read().strip().splitlines()
                self.assertEqual(json.loads(lines[0])["category"], "tile")
                stored = log.recent_alerts(5)
                self.assertEqual(stored[0]["channels"], result["channels"])
                self.assertEqual(log.summary()["n_alerts"], 1)
                paths = [item[0] for item in server.seen]
                self.assertIn("/hook", paths)
                self.assertIn("/fail", paths)
        finally:
            server.shutdown()

    def test_email_uses_smtp(self):
        sent = {}

        class DummySMTP:
            def __init__(self, host, port, timeout=None):
                sent["address"] = (host, port, timeout)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def starttls(self):
                sent["tls"] = True

            def login(self, user, password):
                sent["login"] = (user, password)

            def send_message(self, message: EmailMessage):
                sent["subject"] = message["Subject"]
                sent["to"] = message["To"]

        with tempfile.TemporaryDirectory() as tmp:
            log = InspectionLog(f"{tmp}/qc.db")
            router = AlertRouter(
                log,
                f"{tmp}/alerts.jsonl",
                env={
                    "ALERT_EMAIL_TO": "line@example.com",
                    "ALERT_EMAIL_FROM": "qc@example.com",
                    "ALERT_SMTP_HOST": "smtp.example.com",
                    "ALERT_SMTP_PORT": "2525",
                    "ALERT_SMTP_STARTTLS": "1",
                    "ALERT_SMTP_USER": "qc",
                    "ALERT_SMTP_PASSWORD": "secret",
                },
            )
            with patch("defectclip.alerts.smtplib.SMTP", DummySMTP):
                result = router.send(
                    {
                        "event": "defect",
                        "category": "zipper",
                        "score": 1.0,
                        "threshold": 0.0,
                        "image_name": "z.png",
                    }
                )
            self.assertIn("email", result["channels"])
            self.assertEqual(sent["address"][0], "smtp.example.com")
            self.assertEqual(sent["address"][1], 2525)
            self.assertTrue(sent["tls"])
            self.assertEqual(sent["login"], ("qc", "secret"))
            self.assertIn("zipper", sent["subject"])
            self.assertEqual(sent["to"], "line@example.com")
            payload = json.loads(Path(tmp, "alerts.jsonl").read_text(encoding="utf-8"))
            self.assertNotIn("secret", json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
