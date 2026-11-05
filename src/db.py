import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/app.db")


@contextmanager
def get_conn(db_path=None):
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path=None):
    with get_conn(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT NOT NULL,
                text        TEXT NOT NULL,
                label       TEXT NOT NULL,
                confidence  REAL,
                model       TEXT,
                source      TEXT DEFAULT 'manual'
            )
            """
        )


def save_prediction(text, label, confidence=None, model="baseline",
                    source="manual", db_path=None):
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO predictions (created_at, text, label, confidence, model, source) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), text, label,
             confidence, model, source),
        )


def get_predictions(limit=500, model=None, db_path=None):
    query = "SELECT * FROM predictions"
    params = []
    if model:
        query += " WHERE model = ?"
        params.append(model)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with get_conn(db_path) as conn:
        return [dict(row) for row in conn.execute(query, params).fetchall()]