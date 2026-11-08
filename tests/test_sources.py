import pandas as pd

from src.sources.csv_source import CSVSource


def test_csv_source_reads_non_empty_text(tmp_path):
    csv_path = tmp_path / "posts.csv"
    pd.DataFrame({"message": ["first post", None, "  ", "second post"]}).to_csv(
        csv_path, index=False
    )

    posts = CSVSource(csv_path, text_col="message").fetch(limit=2)

    assert len(posts) == 2
    assert [post.text for post in posts] == ["first post", "second post"]
    assert all(post.source == "csv" for post in posts)
