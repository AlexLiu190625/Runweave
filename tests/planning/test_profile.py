"""Tests for ModelProfile dataclass."""
from __future__ import annotations

from types import SimpleNamespace

from runweave.planning.profile import ModelProfile


def _fake_model(model_id: str = "fake-model"):
    return SimpleNamespace(model_id=model_id)


def _make_profile(**overrides) -> ModelProfile:
    defaults = dict(
        model=_fake_model(),
        context_window=200_000,
        supports_tools=True,
        supports_structured_output=True,
        coding_score=0.85,
        long_context_score=0.7,
        latency="medium",
        cost_tier="medium",
    )
    defaults.update(overrides)
    return ModelProfile(**defaults)


def test_profile_field_access() -> None:
    p = _make_profile()
    assert p.context_window == 200_000
    assert p.supports_tools is True
    assert p.coding_score == 0.85


def test_profile_model_id_property() -> None:
    p = _make_profile(model=_fake_model("claude-sonnet-4-6"))
    assert p.model_id == "claude-sonnet-4-6"


def test_profile_carries_model_instance() -> None:
    fake = _fake_model("gpt-5.3")
    p = _make_profile(model=fake)
    # The Router consumer must be able to reach the underlying model directly.
    assert p.model is fake
