from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any
from app.config import get_settings


def _json(payload: Any) -> str:
    def fallback(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, set):
            return sorted(value)
        return str(value)

    return json.dumps(payload, default=fallback)


def _open_connection(path: Path, probe: bool = True) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    if probe:
        conn.execute("CREATE TABLE IF NOT EXISTS _db_write_probe (id INTEGER PRIMARY KEY)")
        conn.execute("DELETE FROM _db_write_probe WHERE id < 0")
    return conn


def _connect() -> sqlite3.Connection:
    settings = get_settings()
    path = Path(settings.database_path)
    try:
        return _open_connection(path)
    except sqlite3.OperationalError:
        fallback = Path(tempfile.gettempdir()) / "ai_cost_optimizer.sqlite3"
        return _open_connection(fallback, probe=False)


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                request_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                payload TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                cache_key TEXT PRIMARY KEY,
                response TEXT NOT NULL,
                saved_tokens INTEGER NOT NULL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                hits INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                payload TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS robustness (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                payload TEXT NOT NULL
            )
        """)


def save_trace(trace: dict[str, Any]) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO traces (request_id, timestamp, payload) VALUES (?, ?, ?)",
            (trace["request_id"], trace["timestamp"], _json(trace)),
        )


def recent_traces(limit: int = 50) -> list[dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT payload FROM traces ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
    return [json.loads(row["payload"]) for row in rows]


def cache_get(key: str) -> dict[str, Any] | None:
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT response, hits FROM cache WHERE cache_key = ?", (key,)).fetchone()
        if not row:
            return None
        conn.execute("UPDATE cache SET hits = hits + 1 WHERE cache_key = ?", (key,))
    data = json.loads(row["response"])
    data["cache_hit"] = True
    return data


def cache_set(key: str, response: dict[str, Any], saved_tokens: int) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cache (cache_key, response, saved_tokens, hits) VALUES (?, ?, ?, COALESCE((SELECT hits FROM cache WHERE cache_key = ?), 0))",
            (key, _json(response), saved_tokens, key),
        )


def save_evaluation(payload: dict[str, Any]) -> None:
    init_db()
    with _connect() as conn:
        conn.execute("INSERT INTO evaluations (payload) VALUES (?)", (_json(payload),))


def list_evaluations() -> list[dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT payload FROM evaluations ORDER BY id DESC LIMIT 50").fetchall()
    return [json.loads(row["payload"]) for row in rows]


def save_robustness(payload: dict[str, Any]) -> None:
    init_db()
    with _connect() as conn:
        conn.execute("INSERT INTO robustness (payload) VALUES (?)", (_json(payload),))


def list_robustness() -> list[dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT payload FROM robustness ORDER BY id DESC LIMIT 50").fetchall()
    return [json.loads(row["payload"]) for row in rows]


def cache_stats() -> dict[str, Any]:
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) entries, COALESCE(SUM(hits), 0) hits, COALESCE(SUM(saved_tokens), 0) saved FROM cache").fetchone()
    entries = int(row["entries"])
    hits = int(row["hits"])
    return {
        "entries": entries,
        "hits": hits,
        "hit_rate": round((hits / max(1, hits + entries)) * 100, 2),
        "saved_estimated_tokens": int(row["saved"]),
    }


