"""YouTube comments source using the YouTube Data API."""

import os
import re
from urllib.parse import parse_qs, urlparse

import requests
from dotenv import load_dotenv

from src.sources.base import Post, Source

load_dotenv()


class YouTubeSource(Source):
    """Fetch top-level comments from a YouTube video."""

    name = "youtube"
    API_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
    VIDEO_API_URL = "https://www.googleapis.com/youtube/v3/videos"
    VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")

    def __init__(self, video):
        self.video_id = self.extract_video_id(video)
        self.api_key = os.environ.get("YOUTUBE_API_KEY")
        if not self.api_key:
            raise ValueError("Missing YOUTUBE_API_KEY environment variable.")

    @classmethod
    def extract_video_id(cls, video):
        """Extract and validate a video ID from a URL or plain ID."""
        value = video.strip()
        if "://" not in value and "www." not in value and "/" not in value:
            video_id = value
        else:
            parsed = urlparse(value if "://" in value else f"https://{value}")
            host = parsed.netloc.lower().split(":")[0]
            if host in {"youtu.be", "www.youtu.be"}:
                video_id = parsed.path.strip("/").split("/")[0]
            elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
                if parsed.path == "/watch":
                    video_id = parse_qs(parsed.query).get("v", [""])[0]
                else:
                    match = re.match(r"^/(?:shorts|embed)/([^/?]+)", parsed.path)
                    video_id = match.group(1) if match else ""
            else:
                video_id = ""

        if not cls.VIDEO_ID_PATTERN.fullmatch(video_id):
            raise ValueError("Invalid YouTube video URL or ID.")
        return video_id

    def fetch_video_details(self) -> dict:
        """Fetch the video title and total comment count."""
        payload = self._get_payload(
            self.VIDEO_API_URL,
            {"part": "snippet,statistics", "id": self.video_id, "key": self.api_key},
        )
        items = payload.get("items", [])
        if not items:
            raise RuntimeError("The YouTube video was not found; check the video ID.")

        item = items[0]
        return {
            "title": item.get("snippet", {}).get("title", "Untitled video"),
            "comment_count": int(item.get("statistics", {}).get("commentCount", 0)),
        }

    def fetch(self, limit=100, order="relevance") -> list[Post]:
        """Fetch up to limit top-level comments."""
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if order not in {"newest", "relevance", "time"}:
            raise ValueError("order must be 'newest' or 'relevance'.")

        posts = []
        page_token = None

        while len(posts) < limit:
            params = {
                "part": "snippet",
                "videoId": self.video_id,
                "maxResults": min(100, limit - len(posts)),
                "order": "time" if order in {"newest", "time"} else "relevance",
                "textFormat": "plainText",
                "key": self.api_key,
            }
            if page_token:
                params["pageToken"] = page_token

            payload = self._get_payload(self.API_URL, params)

            for item in payload.get("items", []):
                comment = item.get("snippet", {}).get("topLevelComment", {})
                details = comment.get("snippet", {})
                text = details.get("textDisplay", "")
                if not text:
                    continue
                posts.append(
                    Post(
                        text=text,
                        source=self.name,
                        meta={
                            "comment_id": comment.get("id", item.get("id", "")),
                            "video_id": details.get("videoId", self.video_id),
                            "published_at": details.get("publishedAt", ""),
                        },
                    )
                )
                if len(posts) == limit:
                    break

            page_token = payload.get("nextPageToken")
            if not page_token:
                break

        return posts

    def _get_payload(self, url, params):
        try:
            response = requests.get(url, params=params, timeout=15)
        except requests.RequestException:
            raise RuntimeError("YouTube network error. Check your connection and try again.") from None

        try:
            payload = response.json()
        except ValueError:
            raise RuntimeError("The YouTube API returned an invalid response.") from None

        self._raise_for_api_error(response.status_code, payload)
        return payload

    @staticmethod
    def _raise_for_api_error(status_code, payload):
        """Translate YouTube API errors into readable messages."""
        if status_code < 400:
            return

        error = payload.get("error", {})
        errors = error.get("errors", [])
        reasons = {item.get("reason") for item in errors}
        if "commentsDisabled" in reasons:
            detail = "Comments are disabled for this video."
        elif reasons & {"videoNotFound", "notFound"}:
            detail = "The YouTube video was not found; check the video ID."
        elif "quotaExceeded" in reasons:
            detail = "YouTube API quota exceeded."
        else:
            detail = f"YouTube API error ({status_code})."
        raise RuntimeError(detail)
