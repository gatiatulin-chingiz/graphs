"""Тесты нормализации текста и ключей сущностей."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_effects.textnorm import (
    build_entity_key_row,
    build_entity_keys,
    format_keys,
    format_person_key,
    group_key_purity,
    keys_match,
    normalize_series,
    normalize_text,
    normalize_year,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Иванов\n Иван   Иванович ", "иванов иван иванович"),
        ("ПЕТРОВ\u00a0ПЁТР", "петров пётр"),
        ("", ""),
        (None, ""),
        (float("nan"), ""),
        ("nan", ""),
        ("None", ""),
        (pd.NaT, ""),
        ("иванов, иван", "иванов иван"),
    ],
)
def test_normalize_text(raw, expected) -> None:
    """normalize_text приводит значения к каноничному виду."""
    assert normalize_text(raw) == expected


def test_normalize_series_matches_scalar_version() -> None:
    """Векторизованная нормализация совпадает с построчной."""
    values = [" ИВАНОВ ", "петров\u00a0пётр", None, np.nan, "NaN", "Сидоров-Младший"]
    series = pd.Series(values, dtype="object")
    assert normalize_series(series).tolist() == [normalize_text(value) for value in values]


def test_normalize_series_empty() -> None:
    """Пустая серия обрабатывается без ошибок."""
    assert normalize_series(pd.Series([], dtype="object")).empty


def test_format_person_key_includes_date() -> None:
    """Ключ участника содержит ФИО и дату рождения."""
    assert format_person_key("Иванов Иван", "1980-01-01") == "иванов иван 1980-01-01"
    assert format_person_key("Иванов Иван", None) == "иванов иван"
    assert format_person_key("", "1980-01-01") == ""


def test_format_keys_vectorized_matches_scalar() -> None:
    """Векторизованный формат ключей совпадает с поэлементным."""
    names = pd.Series(["Иванов Иван", "Петров Пётр"])
    dates = pd.Series([pd.Timestamp("1980-01-01"), None])
    expected = [format_person_key("Иванов Иван", "1980-01-01"), format_person_key("Петров Пётр", None)]
    assert format_keys(names, dates).tolist() == expected


def test_build_entity_key_row_drops_empty_and_sorts() -> None:
    """Ключ группы: без пустых значений, отсортирован."""
    key = build_entity_key_row(["Петров", None, "иванов", np.nan, ""])
    assert key == ("иванов", "петров")


def test_build_entity_keys_is_stable_across_rows() -> None:
    """Разные убытки одной группы дают одинаковый ключ."""
    frame = pd.DataFrame(
        {
            "_a": ["Иванов", "Петров"],
            "_b": ["Петров", "Иванов"],
            "_c": [None, None],
        }
    )
    keys = build_entity_keys(frame, ["_a", "_b", "_c"])
    assert keys.nunique() == 1
    assert keys.iloc[0] == ("иванов", "петров")


def test_build_entity_keys_without_columns() -> None:
    """Если колонок нет — ключи пустые."""
    frame = pd.DataFrame({"_x": [1, 2]})
    keys = build_entity_keys(frame, ["_missing"])
    assert keys.tolist() == [(), ()]


def test_keys_match_order_insensitive() -> None:
    """Сравнение ключей не зависит от порядка."""
    assert keys_match(("a", "b"), ("b", "a"))
    assert not keys_match(("a",), ("a", "b"))
    assert not keys_match((), ())


def test_normalize_year() -> None:
    """Год извлекается из разных форматов даты."""
    assert normalize_year("2023-05-01") == 2023
    assert normalize_year(pd.Timestamp("1999-12-31")) == 1999
    assert np.isnan(normalize_year("не дата"))
    assert np.isnan(normalize_year(None))


def test_group_key_purity_detects_random_keys() -> None:
    """Метрика чистоты ловит ключи, смешивающие группы."""
    frame = pd.DataFrame(
        {
            "key": [("a",), ("a",), ("b",), ("b",)],
            "group": ["g1", "g1", "g2", "g2"],
        }
    )
    assert group_key_purity(frame, "key", "group") == pytest.approx(1.0)

    mixed = pd.DataFrame(
        {
            "key": [("x",), ("x",), ("y",), ("y",)],
            "group": ["g1", "g2", "g1", "g2"],
        }
    )
    assert group_key_purity(mixed, "key", "group") == pytest.approx(0.5)
