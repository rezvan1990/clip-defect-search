"""SQLite log of API inspections and alerts."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class InspectionLog:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    category TEXT NOT NULL,
                    score REAL,
                    decision TEXT NOT NULL,
                    threshold REAL,
                    image_name TEXT,
                    source TEXT NOT NULL,
                    alerted INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    inspection_id INTEGER,
                    category TEXT NOT NULL,
                    score REAL,
                    threshold REAL,
                    image_name TEXT,
                    channels TEXT NOT NULL,
                    errors TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )

    def add_inspection(
        self,
        *,
        category: str,
        score: float,
        decision: str,
        threshold: float | None,
        image_name: str,
        source: str,
        alerted: bool,
        created_at: str | None = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO inspections (
                    created_at, category, score, decision, threshold, image_name, source, alerted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at or utc_now(),
                    category,
                    score,
                    decision,
                    threshold,
                    image_name,
                    source,
                    int(alerted),
                ),
            )
            return int(cur.lastrowid)

    def add_alert(
        self,
        *,
        inspection_id: int | None,
        category: str,
        score: float,
        threshold: float | None,
        image_name: str,
        channels: list[str],
        errors: list[str],
        payload: dict,
        created_at: str | None = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO alerts (
                    created_at, inspection_id, category, score, threshold,
                    image_name, channels, errors, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at or utc_now(),
                    inspection_id,
                    category,
                    score,
                    threshold,
                    image_name,
                    json.dumps(channels),
                    json.dumps(errors),
                    json.dumps(payload),
                ),
            )
            if inspection_id is not None:
                conn.execute("UPDATE inspections SET alerted = 1 WHERE id = ?", (inspection_id,))
            return int(cur.lastrowid)

    def recent_inspections(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM inspections ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_alerts(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["channels"] = json.loads(item["channels"])
            item["errors"] = json.loads(item["errors"])
            out.append(item)
        return out

    def summary(self) -> dict:
        with self._connect() as conn:
            totals = conn.execute(
                """
                SELECT
                    COUNT(*) AS n,
                    SUM(decision = 'defect') AS n_defect,
                    SUM(decision = 'pass') AS n_pass,
                    SUM(decision = 'unscored') AS n_unscored
                FROM inspections
                """
            ).fetchone()
            n_alerts = conn.execute("SELECT COUNT(*) AS n FROM alerts").fetchone()["n"]
            grouped = conn.execute(
                """
                SELECT
                    category,
                    COUNT(*) AS n,
                    SUM(decision = 'defect') AS n_defect,
                    SUM(decision = 'pass') AS n_pass,
                    SUM(decision != 'unscored') AS n_scored
                FROM inspections
                GROUP BY category
                ORDER BY category
                """
            ).fetchall()
        by_category = []
        for row in grouped:
            scored = int(row["n_scored"] or 0)
            defects = int(row["n_defect"] or 0)
            by_category.append(
                {
                    "category": row["category"],
                    "n": int(row["n"]),
                    "n_defect": defects,
                    "n_pass": int(row["n_pass"] or 0),
                    "n_scored": scored,
                    "defect_rate": (defects / scored) if scored else None,
                }
            )
        return {
            "n": int(totals["n"] or 0),
            "n_defect": int(totals["n_defect"] or 0),
            "n_pass": int(totals["n_pass"] or 0),
            "n_unscored": int(totals["n_unscored"] or 0),
            "n_alerts": int(n_alerts or 0),
            "by_category": by_category,
        }
