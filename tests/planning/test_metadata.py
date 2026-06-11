"""Tests for StepMetadata dataclass."""
from __future__ import annotations

import pytest
from dataclasses import FrozenInstanceError

from runweave.planning.metadata import StepMetadata


def _make_metadata(**overrides) -> StepMetadata:
    defaults = dict(
        kind="edit",
        needs_long_context=False,
        needs_structured_output=False,
        needs_tools=True,
        difficulty="medium",
        style="precise",
    )
    defaults.update(overrides)
    return StepMetadata(**defaults)


def test_metadata_field_access() -> None:
    m = _make_metadata()
    assert m.kind == "edit"
    assert m.needs_long_context is False
    assert m.difficulty == "medium"


def test_metadata_is_frozen() -> None:
    m = _make_metadata()
    with pytest.raises(FrozenInstanceError):
        m.kind = "read"  # type: ignore[misc]


def test_metadata_is_hashable() -> None:
    """Frozen dataclasses with hashable fields are themselves hashable."""
    m1 = _make_metadata()
    m2 = _make_metadata()
    assert hash(m1) == hash(m2)
    assert m1 == m2


def test_metadata_distinguishes_by_fields() -> None:
    m1 = _make_metadata(difficulty="low")
    m2 = _make_metadata(difficulty="high")
    assert m1 != m2
    assert hash(m1) != hash(m2)
