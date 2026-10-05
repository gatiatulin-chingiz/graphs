"""Тесты parquet-кэша для losses/cards."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fin_effects import loaders


def test_load_losses_uses_parquet_cache(tmp_path: Path, monkeypatch) -> None:
    """При наличии parquet SQL не вызывается."""
    cache = tmp_path / "losses.parquet"
    expected = pd.DataFrame({"LossNumber": ["A", "B"], "ValueRub": [1.0, 2.0]})
    expected.to_parquet(cache, index=False)

    def _fail_sql(*_args, **_kwargs):
        raise AssertionError("SQL не должен вызываться при наличии кэша")

    monkeypatch.setattr(loaders, "load_losses_from_db", _fail_sql)
    frame = loaders.load_losses(cache_path=cache, refresh=False)
    assert list(frame["LossNumber"]) == ["A", "B"]


def test_load_losses_refresh_rewrites_cache(tmp_path: Path, monkeypatch) -> None:
    """refresh=True читает SQL и перезаписывает parquet."""
    cache = tmp_path / "losses.parquet"
    pd.DataFrame({"LossNumber": ["OLD"]}).to_parquet(cache, index=False)
    fresh = pd.DataFrame({"LossNumber": ["NEW"], "ValueRub": [10.0]})

    monkeypatch.setattr(loaders, "load_losses_from_db", lambda db_config=None: fresh.copy())
    frame = loaders.load_losses(cache_path=cache, refresh=True)
    assert list(frame["LossNumber"]) == ["NEW"]
    assert list(pd.read_parquet(cache)["LossNumber"]) == ["NEW"]


def test_load_cards_writes_cache_on_miss(tmp_path: Path, monkeypatch) -> None:
    """Без файла — SQL и запись кэша."""
    cache = tmp_path / "cards.parquet"
    fresh = pd.DataFrame({"Номер Убытка": ["L1"], "Дата создания карточки": ["2024-01-01"]})
    monkeypatch.setattr(loaders, "load_cards_from_db", lambda db_config=None: fresh.copy())
    frame = loaders.load_cards(cache_path=cache, refresh=False)
    assert cache.exists()
    assert len(frame) == 1
