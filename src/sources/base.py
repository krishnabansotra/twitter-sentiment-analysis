"""Shared post and source types."""
from dataclasses import dataclass


@dataclass
class Post:
    """A text post and its originating source."""

    text: str
    source: str


class Source:
    """Base interface for post sources."""

    name = "base"

    def fetch(self, limit=100) -> list[Post]:
        """Fetch posts from the source."""
        raise NotImplementedError
