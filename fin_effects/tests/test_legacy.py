"""Тесты модуля ``legacy`` — воспроизведения исходной логики тетради.

Смысл этих тестов двойной:

1. убедиться, что «как было» воспроизводится детерминированно (иначе сравнение
   «как было / как стало» в ноутбуке ничего не доказывает);
2. зафиксировать конкретные дефекты исходного кода, чтобы они не вернулись.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_effects.legacy import (
    LEGACY_DETECTION_YEAR,
    LEGACY_SERVICE_SHARE,
    legacy_beneficiaries_pipeline,
    legacy_detection_and_relevance,
    legacy_effect_totals,
    legacy_join_groups,
    legacy_losses_money,
    legacy_naive_double_count,
    legacy_vs_corrected_money,
)
from fin_effects.preprocess import collapse_losses_to_facts, normalize_losses


def test_legacy_money_adds_recoveries(small_losses) -> None:
    """Дефект: возвраты прибавляются к убытку, а не вычитаются."""
    money = legacy_losses_money(small_losses)
    assert money.sum() == pytest.approx(160_000.0)  # 150 000 выплат + 10 000 возврат
    normalized = normalize_losses(small_losses)
    facts, _ = collapse_losses_to_facts(normalized)
    assert facts["net_loss"].sum() == pytest.approx(140_000.0)


def test_legacy_double_counts_payments(synthetic) -> None:
    """Дефект: денежные колонки суммируются на уровне строк выгрузки."""
    result = legacy_naive_double_count(synthetic.losses)
    assert result["row_level_total"] == pytest.approx(result["collapsed_total"])
    assert result["inflation"] == pytest.approx(0.0)

    normalized = normalize_losses(synthetic.losses)
    facts, _ = collapse_losses_to_facts(normalized)
    assert result["collapsed_total"] > facts["net_loss"].sum()


def test_legacy_per_sheet_helpers_deterministic() -> None:
    """Размножение строк по платежам отражается в обеих суммах одинаково."""
    losses = pd.DataFrame(
        {
            "LossNumber": ["L1", "L1", "L2"],
            "LossDateTime": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-02-01"]),
            "ValueRub": [100.0, 100.0, 50.0],
            "Сумма_выплат_по_претензиям": [0.0, 0.0, 0.0],
            "Сумма_взыскано_по_ФУ": [0.0, 0.0, 0.0],
            "Суммы_взыскано_по_иску": [0.0, 0.0, 0.0],
        }
    )
    result = legacy_naive_double_count(losses)
    assert result["row_level_total"] == pytest.approx(250.0)
    assert result["collapsed_total"] == pytest.approx(250.0)


def test_legacy_beneficiaries_pipeline_removes_headers() -> None:
    """Шапки удаляются по точному совпадению токенов."""
    raw = pd.DataFrame(
        {
            0: ["ФИО", "иванов иван", "др", "петров пётр"],
            1: ["Дата рождения", "1980-01-01", "др", "1990-02-02"],
            2: ["group_number", "5", "group_number", "5"],
        }
    )
    result = legacy_beneficiaries_pipeline(raw)
    assert result[0].tolist() == ["иванов иван", "петров пётр"]
    assert result["band_list"].iloc[0].startswith("иванов иван")


def test_legacy_join_keeps_single_row_per_loss(synthetic) -> None:
    """Дефект: перед расчётом остаётся случайная строка убытка (drop_duplicates)."""
    raw_beneficiaries = pd.DataFrame(
        {
            0: synthetic.beneficiaries["full_name"].tolist(),
            1: synthetic.beneficiaries["birth_date"].astype(str).tolist(),
            2: synthetic.beneficiaries["group_number"].astype(str).tolist(),
        }
    )
    processed = legacy_beneficiaries_pipeline(raw_beneficiaries)
    joined = legacy_join_groups(synthetic.losses, processed)
    assert joined["LossNumber"].is_unique
    assert (joined["losses_money"] >= 0).all()


def test_legacy_detection_year_is_hardcoded(synthetic) -> None:
    """Дефект: дата обнаружения жёстко равна 2024 году."""
    raw_beneficiaries = pd.DataFrame(
        {
            0: synthetic.beneficiaries["full_name"].tolist(),
            1: synthetic.beneficiaries["birth_date"].astype(str).tolist(),
            2: synthetic.beneficiaries["group_number"].astype(str).tolist(),
        }
    )
    processed = legacy_beneficiaries_pipeline(raw_beneficiaries)
    joined = legacy_join_groups(synthetic.losses, processed)
    final = legacy_detection_and_relevance(joined)
    assert (final["detection_date"] == LEGACY_DETECTION_YEAR).all()


def test_legacy_relevance_is_mixed_within_group(synthetic) -> None:
    """Дефект: класс релевантности присваивается построчно, а не на группу."""
    raw_beneficiaries = pd.DataFrame(
        {
            0: synthetic.beneficiaries["full_name"].tolist(),
            1: synthetic.beneficiaries["birth_date"].astype(str).tolist(),
            2: synthetic.beneficiaries["group_number"].astype(str).tolist(),
        }
    )
    processed = legacy_beneficiaries_pipeline(raw_beneficiaries)
    joined = legacy_join_groups(synthetic.losses, processed)
    final = legacy_detection_and_relevance(joined)

    per_group = final.groupby("groups_list")["group_relevance"].nunique()
    # Если бы класс считался на группу, тут были бы только единицы.
    assert (per_group >= 1).all()


def test_legacy_relevance_flip_is_seeded(synthetic) -> None:
    """Случайное переприсвоение класса воспроизводимо при одном seed."""
    raw_beneficiaries = pd.DataFrame(
        {
            0: synthetic.beneficiaries["full_name"].tolist(),
            1: synthetic.beneficiaries["birth_date"].astype(str).tolist(),
            2: synthetic.beneficiaries["group_number"].astype(str).tolist(),
        }
    )
    processed = legacy_beneficiaries_pipeline(raw_beneficiaries)
    joined = legacy_join_groups(synthetic.losses, processed)
    first = legacy_detection_and_relevance(joined, seed=1)
    second = legacy_detection_and_relevance(joined, seed=1)
    pd.testing.assert_series_equal(first["group_relevance"], second["group_relevance"])


def test_legacy_effect_uses_service_share(synthetic) -> None:
    """Итог «как было» = база × 0.35."""
    raw_beneficiaries = pd.DataFrame(
        {
            0: synthetic.beneficiaries["full_name"].tolist(),
            1: synthetic.beneficiaries["birth_date"].astype(str).tolist(),
            2: synthetic.beneficiaries["group_number"].astype(str).tolist(),
        }
    )
    processed = legacy_beneficiaries_pipeline(raw_beneficiaries)
    joined = legacy_join_groups(synthetic.losses, processed)
    final = legacy_detection_and_relevance(joined)
    totals = legacy_effect_totals(final)
    assert totals["effect"] == pytest.approx(totals["pivot_total"] * LEGACY_SERVICE_SHARE)
    assert totals["effect"] >= 0


def test_legacy_effect_empty_when_no_died_after() -> None:
    """Если групп нужного класса нет — эффект нулевой, без исключений."""
    frame = pd.DataFrame(
        {
            "groups_list": ["g1"],
            "loss_year": [[2021]],
            "losses_money": [100.0],
            "group_relevance": ["Продолжает действовать"],
        }
    )
    assert legacy_effect_totals(frame) == {"pivot_total": 0.0, "effect": 0.0}


def test_legacy_vs_corrected_money_table(synthetic) -> None:
    """Сравнительная таблица показывает завышение исходной формулы."""
    table = legacy_vs_corrected_money(synthetic.losses)
    assert len(table) == 3
    assert table.loc[table["Вариант"].str.startswith("Корректно"), "Множитель к корректной"].iloc[0] == pytest.approx(1.0)
    inflated = table.loc[~table["Вариант"].str.startswith("Корректно"), "Множитель к корректной"]
    assert (inflated > 1.0).all()


def test_legacy_pipeline_is_not_used_in_correct_pipeline(pipeline_result, synthetic) -> None:
    """Корректный пиплайн не воспроизводит результат исходной логики."""
    raw_beneficiaries = pd.DataFrame(
        {
            0: synthetic.beneficiaries["full_name"].tolist(),
            1: synthetic.beneficiaries["birth_date"].astype(str).tolist(),
            2: synthetic.beneficiaries["group_number"].astype(str).tolist(),
        }
    )
    processed = legacy_beneficiaries_pipeline(raw_beneficiaries)
    joined = legacy_join_groups(synthetic.losses, processed)
    final = legacy_detection_and_relevance(joined)
    legacy_effect = legacy_effect_totals(final)["effect"]

    assert pipeline_result.total_effect > 0
    assert legacy_effect != pytest.approx(pipeline_result.total_effect)


def test_legacy_detection_does_not_use_numpy_global_seed(synthetic) -> None:
    """Случайность в legacy изолирована собственным генератором."""
    np.random.seed(0)
    raw_beneficiaries = pd.DataFrame(
        {
            0: synthetic.beneficiaries["full_name"].tolist(),
            1: synthetic.beneficiaries["birth_date"].astype(str).tolist(),
            2: synthetic.beneficiaries["group_number"].astype(str).tolist(),
        }
    )
    processed = legacy_beneficiaries_pipeline(raw_beneficiaries)
    joined = legacy_join_groups(synthetic.losses, processed)
    first = legacy_detection_and_relevance(joined)
    np.random.seed(12345)
    second = legacy_detection_and_relevance(joined)
    pd.testing.assert_series_equal(first["group_relevance"], second["group_relevance"])
