import sqlite3

from src.db import (
    clear_predictions,
    delete_predictions,
    get_predictions,
    init_db,
    save_prediction,
)


def test_save_and_read(tmp_path):
    db = tmp_path / "test.db"
    init_db(db)
    save_prediction("I love this phone!", "positive", 0.97, "baseline", db_path=db)
    rows = get_predictions(db_path=db)
    assert len(rows) == 1
    assert rows[0]["label"] == "positive"


def test_save_and_read_posted_at(tmp_path):
    db = tmp_path / "dated.db"
    init_db(db)
    posted_at = "2024-01-02T03:04:05Z"
    save_prediction(
        "A useful comment",
        "neutral",
        0.75,
        "bert",
        source="youtube",
        posted_at=posted_at,
        db_path=db,
    )

    rows = get_predictions(db_path=db)
    assert rows[0]["posted_at"] == posted_at


def test_init_db_migrates_existing_table_without_posted_at(tmp_path):
    db = tmp_path / "legacy.db"
    with sqlite3.connect(db) as connection:
        connection.execute(
            "CREATE TABLE predictions ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, "
            "text TEXT NOT NULL, label TEXT NOT NULL, confidence REAL, "
            "model TEXT, source TEXT DEFAULT 'manual')"
        )
        connection.execute(
            "INSERT INTO predictions (created_at, text, label) "
            "VALUES ('2024-01-01T00:00:00Z', 'Old row', 'positive')"
        )

    init_db(db)
    save_prediction("New row", "negative", db_path=db)
    rows = get_predictions(db_path=db)

    assert len(rows) == 2
    assert rows[0]["text"] == "New row"
    assert rows[0]["posted_at"] is None
    assert rows[1]["text"] == "Old row"
    assert rows[1]["posted_at"] is None


def test_delete_selected_prediction_ids(tmp_path):
    db = tmp_path / "delete_selected.db"
    init_db(db)
    for text in ["first", "second", "third"]:
        save_prediction(text, "neutral", db_path=db)
    rows = get_predictions(limit=None, db_path=db)

    deleted = delete_predictions([rows[0]["id"], rows[2]["id"]], db_path=db)

    assert deleted == 2
    assert [row["text"] for row in get_predictions(limit=None, db_path=db)] == ["second"]


def test_clear_all_predictions(tmp_path):
    db = tmp_path / "clear_all.db"
    init_db(db)
    save_prediction("first", "positive", db_path=db)
    save_prediction("second", "negative", db_path=db)

    assert clear_predictions(db_path=db) == 2
    assert get_predictions(limit=None, db_path=db) == []


def test_clear_predictions_by_source(tmp_path):
    db = tmp_path / "clear_source.db"
    init_db(db)
    save_prediction("one", "positive", source="upload", db_path=db)
    save_prediction("two", "negative", source="youtube", db_path=db)
    save_prediction("three", "neutral", source="upload", db_path=db)

    assert clear_predictions(source="upload", db_path=db) == 2
    rows = get_predictions(limit=None, db_path=db)
    assert [row["text"] for row in rows] == ["two"]


def test_deleting_nothing_returns_zero(tmp_path):
    db = tmp_path / "delete_nothing.db"
    init_db(db)
    save_prediction("keep me", "neutral", db_path=db)

    assert delete_predictions([], db_path=db) == 0
    assert delete_predictions([999], db_path=db) == 0
    assert clear_predictions(source="missing", db_path=db) == 0
