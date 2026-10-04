"""Тесты предобработки убытков: дедупликация, денежные метрики, сопоставление с группами."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_effects.config import BENEFICIARY_KEY_COLUMNS
from fin_effects.preprocess import (
    add_group_number_column,
    attach_groups_to_losses,
    collapse_losses_to_facts,
    money_columns_present,
    normalize_losses,
    safe_divide,
)
from fin_effects.textnorm import group_key_purity, normalize_series


def test_collapse_sums_payments_without_double_counting(small_losses) -> None:
    """Дубли строк по платежам сворачиваются в один убыток без двойного учёта."""
    normalized = normalize_losses(small_losses)
    facts, stats = collapse_losses_to_facts(normalized)

    assert len(facts) == 1
    assert stats.duplicated_loss_numbers == 1
    assert facts["gross_loss"].iloc[0] == pytest.approx(150_000.0)
    assert facts["recoveries"].iloc[0] == pytest.approx(10_000.0)
    assert facts["net_loss"].iloc[0] == pytest.approx(140_000.0)


def test_collapse_equals_row_sum_for_unique_losses(synthetic) -> None:
    """Для убытков без дублей свёртка не меняет сумму выплат."""
    normalized = normalize_losses(synthetic.losses)
    facts, _ = collapse_losses_to_facts(normalized)

    counts = normalized["LossNumber"].value_counts()
    only_unique = normalized.loc[normalized["LossNumber"].isin(counts[counts == 1].index)]
    expected = float(
        only_unique.groupby("LossNumber")
        .apply(
            lambda group: group["ValueRub"].sum() + group["Сумма_выплат_по_претензиям"].sum(),
            include_groups=False,
        )
        .sum()
    )
    actual = float(
        facts.loc[facts["LossNumber"].isin(only_unique["LossNumber"]), "gross_loss"].sum()
    )
    assert actual == pytest.approx(expected)


def test_collapse_never_inflates_total_versus_rows(synthetic) -> None:
    """Свёрнутая сумма не превышает построчную (защита от завышения)."""
    normalized = normalize_losses(synthetic.losses)
    facts, _ = collapse_losses_to_facts(normalized)

    row_level = (
        normalized["ValueRub"].fillna(0) + normalized["Сумма_выплат_по_претензиям"].fillna(0)
    ).sum()
    assert facts["gross_loss"].sum() == pytest.approx(row_level, rel=1e-9)


def test_recoveries_are_subtracted(synthetic) -> None:
    """Возвраты уменьшают базу, а не увеличивают её."""
    normalized = normalize_losses(synthetic.losses)
    facts, _ = collapse_losses_to_facts(normalized)
    assert (facts["recoveries"] >= 0).all()
    assert facts["net_loss"].sum() == pytest.approx(
        facts["gross_loss"].sum() - facts["recoveries"].sum()
    )
    assert facts["net_loss"].sum() <= facts["gross_loss"].sum()


def test_normalize_losses_builds_person_keys(small_losses) -> None:
    """Нормализация собирает ключи «ФИО + ДР» и приводит даты к datetime."""
    normalized = normalize_losses(small_losses)
    assert pd.api.types.is_datetime64_any_dtype(normalized["LossDateTime"])
    assert normalized["_Виновник.Полн наименование"].iloc[0] == "иванов иван 1980-01-01"
    assert normalized["_Потерпевший.Полн наименование"].iloc[0] == "петров пётр 1990-05-05"


def test_normalize_losses_requires_key_columns() -> None:
    """Без обязательных колонок поднимается понятная ошибка."""
    with pytest.raises(KeyError, match="LossNumber"):
        normalize_losses(pd.DataFrame({"LossDateTime": []}))
    with pytest.raises(KeyError, match="LossDateTime"):
        normalize_losses(pd.DataFrame({"LossNumber": []}))


def test_normalize_losses_fills_missing_money_with_zero() -> None:
    """Отсутствующие денежные колонки считаются нулями."""
    frame = pd.DataFrame({"LossNumber": ["L1"], "LossDateTime": [pd.Timestamp("2023-01-01")]})
    normalized = normalize_losses(frame)
    for column in ("ValueRub", "Сумма_выплат_по_претензиям"):
        assert column in normalized.columns
        assert normalized[column].iloc[0] == 0.0


def test_group_key_is_person_based(synthetic) -> None:
    """Ключ группы не содержит VIN и не вырождается в «убыток = группа»."""
    normalized = normalize_losses(synthetic.losses)
    facts, _ = collapse_losses_to_facts(normalized)
    keys = facts["group_key"]
    assert keys.map(len).max() <= len(BENEFICIARY_KEY_COLUMNS)
    assert keys.nunique() < len(facts)


def test_attach_groups_matches_expected_groups(synthetic) -> None:
    """Каждому убытку группы соответствует её group_id."""
    normalized = normalize_losses(synthetic.losses)
    facts, _ = collapse_losses_to_facts(normalized)
    _, facts_with_groups = attach_groups_to_losses(facts, synthetic.beneficiaries)

    assert not facts_with_groups.empty
    assert facts_with_groups["group_id"].nunique() > 1
    assert (facts_with_groups["matched_members"] >= 1).all()

    purity = group_key_purity(
        facts_with_groups[["LossNumber", "group_key", "group_id"]].drop_duplicates(),
        "group_key",
        "group_id",
    )
    assert purity == pytest.approx(1.0)


def test_attach_groups_returns_all_facts_when_no_beneficiaries(synthetic) -> None:
    """Без выгрузки сервиса убытки не привязываются ни к одной группе."""
    normalized = normalize_losses(synthetic.losses)
    facts, _ = collapse_losses_to_facts(normalized)
    _, facts_with_groups = attach_groups_to_losses(facts, pd.DataFrame())
    assert facts_with_groups.empty


def test_attach_groups_respects_min_members(synthetic) -> None:
    """Порог min_members_matched фильтрует пары убыток-группа."""
    normalized = normalize_losses(synthetic.losses)
    facts, _ = collapse_losses_to_facts(normalized)
    _, loose = attach_groups_to_losses(facts, synthetic.beneficiaries, min_members_matched=1)
    _, strict = attach_groups_to_losses(facts, synthetic.beneficiaries, min_members_matched=2)
    assert len(strict) <= len(loose)


def test_add_group_number_column(synthetic) -> None:
    """Номер группы из выгрузки сервиса переносится в пары убыток-группа."""
    normalized = normalize_losses(synthetic.losses)
    facts, _ = collapse_losses_to_facts(normalized)
    _, facts_with_groups = attach_groups_to_losses(facts, synthetic.beneficiaries)
    enriched = add_group_number_column(facts_with_groups, synthetic.beneficiaries)
    assert "group_number" in enriched.columns
    assert enriched["group_number"].notna().all()

    truth = synthetic.groups.set_index("group_id")["group_number"]
    assert enriched["group_number"].equals(enriched["group_id"].map(truth).astype(enriched["group_number"].dtype))


def test_add_group_number_column_without_source_column(synthetic) -> None:
    """Если номера групп нет в выгрузке — колонка не добавляется."""
    normalized = normalize_losses(synthetic.losses)
    facts, _ = collapse_losses_to_facts(normalized)
    _, facts_with_groups = attach_groups_to_losses(facts, synthetic.beneficiaries)
    result = add_group_number_column(facts_with_groups, synthetic.beneficiaries.drop(columns=["group_number"]))
    assert "group_number" not in result.columns


def test_money_columns_present() -> None:
    """Проверка наличия денежных колонок."""
    frame = pd.DataFrame({"ValueRub": [1.0], "Прочее": [2.0]})
    assert money_columns_present(frame) == ["ValueRub"]


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [(10.0, 2.0, 5.0), (10.0, 0.0, np.nan), (10.0, float("nan"), np.nan)],
)
def test_safe_divide(numerator, denominator, expected) -> None:
    """Деление с защитой от нуля."""
    result = safe_divide(numerator, denominator)
    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result == pytest.approx(expected)


def test_normalize_series_used_on_object_columns(small_losses) -> None:
    """Нормализация выгрузки не падает на числовых колонках."""
    normalized = normalize_losses(small_losses)
    assert normalize_series(normalized["LossNumber"]).tolist() == ["l1", "l1"]
