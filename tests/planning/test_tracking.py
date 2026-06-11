"""Tests for TokenUsageTracker and TrackedModel."""
from __future__ import annotations

import threading
from types import SimpleNamespace

from runweave.planning.tracking import TokenUsageTracker, TrackedModel


class _FakeModel:
    model_id = "fake-model"

    def __init__(self, usage: SimpleNamespace | None) -> None:
        self.usage = usage
        self.calls: list[tuple] = []
        self.extra_attr = "extra-value"

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return SimpleNamespace(content="reply", token_usage=self.usage)


def test_tracker_accumulates_across_calls() -> None:
    tracker = TokenUsageTracker()
    tracker.add(100, 50)
    tracker.add(200, 75)
    assert tracker.input_tokens == 300
    assert tracker.output_tokens == 125
    assert tracker.call_count == 2
    assert tracker.as_dict() == {"input_tokens": 300, "output_tokens": 125}


def test_tracker_handles_none_token_usage() -> None:
    """When wrapped model returns token_usage=None, tracker stays at zero."""
    tracker = TokenUsageTracker()
    fake = _FakeModel(usage=None)
    tracked = TrackedModel(fake, tracker)
    tracked("hi")
    assert tracker.input_tokens == 0
    assert tracker.output_tokens == 0
    assert tracker.call_count == 0  # add() was never called


def test_tracked_model_forwards_attribute_access() -> None:
    tracker = TokenUsageTracker()
    fake = _FakeModel(usage=SimpleNamespace(input_tokens=10, output_tokens=5))
    tracked = TrackedModel(fake, tracker)
    # model_id is an explicit property
    assert tracked.model_id == "fake-model"
    # Any other attribute should forward via __getattr__
    assert tracked.extra_attr == "extra-value"


def test_tracked_model_preserves_call_args() -> None:
    tracker = TokenUsageTracker()
    fake = _FakeModel(usage=SimpleNamespace(input_tokens=10, output_tokens=5))
    tracked = TrackedModel(fake, tracker)
    tracked("some-messages", temperature=0.7, max_tokens=100)
    assert fake.calls == [("some-messages", {"temperature": 0.7, "max_tokens": 100})]
    assert tracker.input_tokens == 10
    assert tracker.output_tokens == 5


def test_tracker_thread_safe() -> None:
    """Concurrent add() calls from multiple threads accumulate exactly."""
    tracker = TokenUsageTracker()

    def worker() -> None:
        for _ in range(100):
            tracker.add(1, 2)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert tracker.input_tokens == 10 * 100 * 1
    assert tracker.output_tokens == 10 * 100 * 2
    assert tracker.call_count == 10 * 100
