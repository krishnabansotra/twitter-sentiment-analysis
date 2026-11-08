import pytest
import requests

from src.sources.youtube_source import YouTubeSource


@pytest.mark.parametrize(
    "video, expected",
    [
        ("abcdefghijk", "abcdefghijk"),
        ("https://www.youtube.com/watch?v=abcdefghijk", "abcdefghijk"),
        ("https://youtu.be/abcdefghijk?t=10", "abcdefghijk"),
        ("https://www.youtube.com/shorts/abcdefghijk", "abcdefghijk"),
    ],
)
def test_extract_video_id(video, expected):
    assert YouTubeSource.extract_video_id(video) == expected


def test_extract_video_id_rejects_invalid_input():
    with pytest.raises(ValueError, match="Invalid YouTube video URL or ID"):
        YouTubeSource.extract_video_id("https://example.com/not-a-video")


def test_fetch_paginates_and_keeps_only_comment_metadata(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "test-key")
    source = YouTubeSource("abcdefghijk")
    responses = [
        {
            "items": [
                {
                    "id": "thread1",
                    "snippet": {
                        "topLevelComment": {
                            "id": "comment1",
                            "snippet": {
                                "textDisplay": "First comment",
                                "videoId": "abcdefghijk",
                                "publishedAt": "2026-01-01T00:00:00Z",
                                "authorDisplayName": "Do not retain",
                            },
                        },
                    },
                }
            ],
            "nextPageToken": "next",
        },
        {
            "items": [
                {
                    "snippet": {
                        "topLevelComment": {
                            "id": "comment2",
                            "snippet": {
                                "textDisplay": "Second comment",
                                "videoId": "abcdefghijk",
                                "publishedAt": "2026-01-02T00:00:00Z",
                            },
                        },
                    },
                }
            ],
        },
    ]
    calls = []

    class Response:
        status_code = 200

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    def fake_get(url, params, timeout):
        calls.append(params)
        return Response(responses.pop(0))

    monkeypatch.setattr("src.sources.youtube_source.requests.get", fake_get)

    posts = source.fetch(limit=2)

    assert [post.text for post in posts] == ["First comment", "Second comment"]
    assert posts[0].meta == {
        "comment_id": "comment1",
        "video_id": "abcdefghijk",
        "published_at": "2026-01-01T00:00:00Z",
    }
    assert len(calls) == 2
    assert calls[0]["order"] == "relevance"
    assert calls[0]["textFormat"] == "plainText"
    assert calls[1]["pageToken"] == "next"


def test_fetch_maps_newest_to_youtube_time_order(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "test-key")
    source = YouTubeSource("abcdefghijk")
    calls = []

    class Response:
        status_code = 200

        def json(self):
            return {"items": []}

    def fake_get(url, params, timeout):
        calls.append(params)
        return Response()

    monkeypatch.setattr("src.sources.youtube_source.requests.get", fake_get)

    assert source.fetch(limit=1, order="newest") == []
    assert calls[0]["order"] == "time"


def test_fetch_video_details_returns_title_and_comment_count(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "test-key")
    source = YouTubeSource("abcdefghijk")
    calls = []

    class Response:
        status_code = 200

        def json(self):
            return {
                "items": [
                    {
                        "snippet": {"title": "A sample video"},
                        "statistics": {"commentCount": "245"},
                    }
                ]
            }

    def fake_get(url, params, timeout):
        calls.append((url, params))
        return Response()

    monkeypatch.setattr("src.sources.youtube_source.requests.get", fake_get)

    assert source.fetch_video_details() == {
        "title": "A sample video",
        "comment_count": 245,
    }
    assert calls[0][0] == YouTubeSource.VIDEO_API_URL
    assert calls[0][1]["part"] == "snippet,statistics"
    assert calls[0][1]["id"] == "abcdefghijk"


def test_fetch_video_details_reports_missing_video(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "test-key")
    source = YouTubeSource("abcdefghijk")

    class Response:
        status_code = 200

        def json(self):
            return {"items": []}

    monkeypatch.setattr(
        "src.sources.youtube_source.requests.get",
        lambda *args, **kwargs: Response(),
    )

    with pytest.raises(RuntimeError, match="video was not found"):
        source.fetch_video_details()


def test_fetch_reports_quota_exceeded(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "test-key")
    source = YouTubeSource("abcdefghijk")

    class Response:
        status_code = 403

        def json(self):
            return {"error": {"errors": [{"reason": "quotaExceeded"}]}}

    monkeypatch.setattr(
        "src.sources.youtube_source.requests.get",
        lambda *args, **kwargs: Response(),
    )

    with pytest.raises(RuntimeError, match="quota exceeded"):
        source.fetch_video_details()


def test_fetch_network_error_does_not_include_request_details(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "test-key")
    source = YouTubeSource("abcdefghijk")

    def failed_request(*args, **kwargs):
        raise requests.RequestException("request URL contained test-key")

    monkeypatch.setattr("src.sources.youtube_source.requests.get", failed_request)

    with pytest.raises(RuntimeError, match="network error") as error:
        source.fetch_video_details()
    assert "test-key" not in str(error.value)


def test_fetch_reports_comments_disabled(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "test-key")
    source = YouTubeSource("abcdefghijk")

    class Response:
        status_code = 403

        def json(self):
            return {
                "error": {
                    "message": "Comments are disabled",
                    "errors": [{"reason": "commentsDisabled"}],
                },
            }

    monkeypatch.setattr(
        "src.sources.youtube_source.requests.get",
        lambda *args, **kwargs: Response(),
    )

    with pytest.raises(RuntimeError, match="Comments are disabled"):
        source.fetch()
