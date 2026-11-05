from src.db import init_db, save_prediction, get_predictions


def test_save_and_read(tmp_path):
    db = tmp_path / "test.db"
    init_db(db)
    save_prediction("I love this phone!", "positive", 0.97, "baseline", db_path=db)
    rows = get_predictions(db_path=db)
    assert len(rows) == 1
    assert rows[0]["label"] == "positive"