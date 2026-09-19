"""Lightweight JSON / SQLite persistence for experiment results."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from engine.experiments.result import ExperimentResult


class ExperimentDatabase:
    """Persist experiment summaries. JSON files are primary; SQLite optional."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save(self, result: ExperimentResult) -> None:
        payload = json.dumps(result.to_dict())
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments
                (experiment_id, name, created_at, decision, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    result.experiment_id,
                    result.name,
                    result.created_at,
                    result.decision.value,
                    payload,
                ),
            )
            conn.commit()

    def get(self, experiment_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT payload FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def list_ids(self) -> list[str]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT experiment_id FROM experiments ORDER BY created_at DESC"
            ).fetchall()
        return [r[0] for r in rows]
