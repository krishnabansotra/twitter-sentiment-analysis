"""CSV-backed post source."""
from pathlib import Path

import pandas as pd

from src.sources.base import Post, Source


class CSVSource(Source):
    """Read posts from a CSV file."""

    name = "csv"

    def __init__(self, path, text_col="text"):
        self.path = Path(path)
        self.text_col = text_col

    def fetch(self, limit=100) -> list[Post]:
        """Return up to limit non-empty posts from the CSV."""
        if limit < 0:
            raise ValueError("limit must be non-negative")

        frame = pd.read_csv(self.path)
        if self.text_col not in frame.columns:
            raise ValueError(
                f"CSV file {self.path} has no {self.text_col!r} column"
            )
        texts = frame[self.text_col].dropna().astype(str)
        texts = texts[texts.str.strip().ne("")].head(limit)
        return [Post(text=text, source=self.name) for text in texts]
