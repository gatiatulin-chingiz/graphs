"""Тесты построения групп: обнаружение, жизнь, активные годы, классы релевантности."""

from __future__ import annotations

import pandas as pd
import pytest

from fin_effects.config import (
    RELEVANCE_ALIVE,
    RELEVANCE_DIED_AFTER,
    RELEVANCE_DIED_BEFORE,
)
from fin_effects.groups import (
    build_group_summary,
    classify_group_relevance,
    filter_groups_by_size,
    filter_observation_period,
    loss_detection_dates,
    year_matrix,
)
from fin_effects.preprocess import collapse_losses_to_facts, normalize_losses
from fin_effects.preprocess import attach_groups_to_losses

from .conftest import TEST_PARAMS


@pytest.fixture(scope="module")
def facts_with_groups(synthetic):
    """Пары «убыток-группа» на синтетических данных."""
    normalized = normalize_losses(synthetic.losses)
    facts, _ = collapse_losses_to_facts(normalized)
    facts, _ = filter_observation_period(facts, TEST_PARAMS)
    _, paired = attach_groups_to_losses(facts, synthetic.beneficiaries)
    return paired


def test_loss_detection_dates_takes_minimum(synthetic) -> None:
    """Дата обнаружения — минимальная дата карточки по убытку."""
    dates = loss_detection_dates(synthetic.cards)
    assert dates["card_date"].notna().all()
    assert dates["LossNumber"].is_unique
    assert dates["card_date"].dt.year.between(2020, 2027).all()


def test_loss_detection_dates_normalizes_loss_number() -> None:
    """Номер убытка нормализуется: иначе карточки не джойнятся с фактами."""
    cards = pd.DataFrame(
        {
            "Номер Убытка": ["  L0001  ", "l0001"],
            "Дата создания карточки": ["2024-05-01", "2024-03-01"],
        }
    )
    dates = loss_detection_dates(cards)
    assert dates["LossNumber"].tolist() == ["l0001"]
    assert dates["card_date"].iloc[0] == pd.Timestamp("2024-03-01")


def test_loss_detection_dates_requires_columns() -> None:
    """Понятная ошибка при отсутствии колонок карточек."""
    with pytest.raises(KeyError, match="Номер Убытка"):
        loss_detection_dates(pd.DataFrame({"Дата создания карточки": []}))


def test_build_group_summary_matches_ground_truth(synthetic, facts_with_groups) -> None:
    """Гранулярность сводки — одна строка на группу с карточкой УПС."""
    summary, _ = build_group_summary(facts_with_groups, synthetic.cards, TEST_PARAMS)
    assert summary["group_id"].is_unique
    assert set(summary["group_id"]).issubset(set(facts_with_groups["group_id"]))
    assert (summary["losses_total"] >= 1).all()
    assert summary["detection_date"].notna().all()


def test_build_group_summary_detection_matches_truth(synthetic, facts_with_groups) -> None:
    """Год обнаружения из карточек совпадает с эталоном генератора."""
    summary, _ = build_group_summary(facts_with_groups, synthetic.cards, TEST_PARAMS)
    expected = synthetic.groups.set_index("group_id")["observed_detection_year"]
    actual = summary.set_index("group_id")["detection_year"].astype(int)
    common = actual.index.intersection(expected.index)
    assert (actual.loc[common] == expected.loc[common].astype(int)).all()


def test_build_group_summary_fallback_without_cards(synthetic, facts_with_groups) -> None:
    """Без карточек группы не попадают в расчёт эффекта."""
    summary, stats = build_group_summary(facts_with_groups, None, TEST_PARAMS)
    assert summary.empty
    assert stats.groups_without_card == facts_with_groups["group_id"].nunique()


def test_build_group_summary_empty_input() -> None:
    """Пустой вход даёт пустую сводку."""
    summary, stats = build_group_summary(pd.DataFrame(), None, TEST_PARAMS)
    assert summary.empty
    assert stats.groups_without_card == 0


def test_classify_group_relevance_rules() -> None:
    """Статус берётся из стадии карточки УПС."""
    summary = pd.DataFrame(
        {
            "group_id": ["open", "closed", "unknown"],
            "card_stage": [RELEVANCE_ALIVE, RELEVANCE_DIED_AFTER, None],
            "detection_year": [2022, 2023, 2024],
        }
    )
    classified = classify_group_relevance(summary)
    result = classified.set_index("group_id")["group_relevance"]
    assert result["open"] == RELEVANCE_ALIVE
    assert result["closed"] == RELEVANCE_DIED_AFTER
    assert "unknown" not in result.index


def test_classify_group_relevance_from_raw_stage() -> None:
    """Сырые стадии «В работе» / «Закрыта» нормализуются."""
    summary = pd.DataFrame(
        {"group_id": ["g"], "card_stage": ["В работе"], "detection_year": [2022]}
    )
    assert classify_group_relevance(summary).loc[0, "group_relevance"] == RELEVANCE_ALIVE
    closed = pd.DataFrame(
        {"group_id": ["g"], "card_stage": ["Закрыта"], "detection_year": [2022]}
    )
    assert classify_group_relevance(closed).loc[0, "group_relevance"] == RELEVANCE_DIED_AFTER


def test_classify_group_relevance_matches_reference(synthetic, facts_with_groups) -> None:
    """Классы на синтетике совпадают с эталоном генератора на 100%."""
    summary, _ = build_group_summary(facts_with_groups, synthetic.cards, TEST_PARAMS)
    summary, _ = filter_groups_by_size(summary, TEST_PARAMS)
    classified = classify_group_relevance(summary)
    merged = classified.merge(
        synthetic.groups.loc[:, ["group_id", "expected_relevance"]], on="group_id", how="left"
    )
    merged = merged.loc[merged["expected_relevance"].notna()]
    assert not merged.empty
    assert (merged["group_relevance"] == merged["expected_relevance"]).all()


def test_filter_groups_by_size() -> None:
    """Порог по числу убытков отсекает «группы» из одного убытка."""
    summary = pd.DataFrame(
        {"group_id": ["a", "b", "c"], "losses_total": [1, 3, 5]}
    )
    filtered, dropped = filter_groups_by_size(summary, TEST_PARAMS)
    assert dropped == 1
    assert set(filtered["group_id"]) == {"b", "c"}


def test_filter_groups_by_size_empty() -> None:
    """Пустая сводка не падает."""
    filtered, dropped = filter_groups_by_size(pd.DataFrame(), TEST_PARAMS)
    assert filtered.empty
    assert dropped == 0


def test_filter_observation_period() -> None:
    """Убытки вне периода наблюдения отбрасываются."""
    facts = pd.DataFrame({"loss_year": [2018, 2020, 2025, 2030], "net_loss": [1.0] * 4})
    filtered, dropped = filter_observation_period(facts, TEST_PARAMS)
    assert dropped == 2
    assert filtered["loss_year"].tolist() == [2020, 2025]


def test_year_matrix() -> None:
    """Матрица «группа × год» кодирует активные годы единицами."""
    summary = pd.DataFrame(
        {
            "group_id": ["g1", "g2"],
            "group_relevance": [RELEVANCE_ALIVE, RELEVANCE_DIED_AFTER],
            "detection_year": [2022, 2023],
            "active_years": [(2021, 2022), (2023,)],
        }
    )
    matrix = year_matrix(summary)
    assert set(matrix.columns) >= {2021, 2022, 2023}
    row = matrix.set_index("group_id").loc["g1"]
    assert row[2021] == 1
    assert row[2022] == 1
    assert row[2023] == 0


def test_year_matrix_empty() -> None:
    """Пустая сводка даёт пустую матрицу."""
    assert year_matrix(pd.DataFrame()).empty
