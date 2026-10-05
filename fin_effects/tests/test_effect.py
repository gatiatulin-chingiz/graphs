"""Тесты ядра методики: база эффекта, классы релевантности, горизонт прогноза."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_effects.config import (
    RELEVANCE_ALIVE,
    RELEVANCE_DIED_AFTER,
    RELEVANCE_DIED_BEFORE,
    EffectParams,
)
from fin_effects.effect import (
    compute_group_baselines,
    compute_group_effect,
    effect_by_cohort,
    effect_long,
    effect_matrix,
    reconcile,
    totals_by_relevance,
)
from fin_effects.groups import filter_observation_period

#: Группа «действует»: 3 убытка × 100 000 ₽ = 300 000 ₽ за один активный год.
ALIVE_GROUP = {
    "group_id": "alive",
    "group_relevance": RELEVANCE_ALIVE,
    "detection_year": 2023,
    "active_years_count": 1,
    "losses_total": 3,
    "net_loss_total": 300_000.0,
    "active_years": (2023,),
    "net_loss_active": 300_000.0,
}

#: Группа «распалась после обнаружения»: 2 убытка × 50 000 ₽.
DIED_AFTER_GROUP = {
    "group_id": "died_after",
    "group_relevance": RELEVANCE_DIED_AFTER,
    "detection_year": 2022,
    "active_years_count": 1,
    "losses_total": 2,
    "net_loss_total": 100_000.0,
    "active_years": (2022,),
    "net_loss_active": 100_000.0,
}

#: Группа «распалась самостоятельно»: эффект должен быть нулевым.
DIED_BEFORE_GROUP = {
    "group_id": "died_before",
    "group_relevance": RELEVANCE_DIED_BEFORE,
    "detection_year": 2024,
    "active_years_count": 1,
    "losses_total": 4,
    "net_loss_total": 400_000.0,
    "active_years": (2021,),
    "net_loss_active": 400_000.0,
}


@pytest.fixture
def summary_fixture() -> pd.DataFrame:
    """Сводка по трём группам с разными классами релевантности."""
    return pd.DataFrame([ALIVE_GROUP, DIED_AFTER_GROUP, DIED_BEFORE_GROUP])


@pytest.fixture
def params_fixture() -> EffectParams:
    """Параметры с горизонтом 2 года и известной долей сервиса."""
    return EffectParams(
        service_share=0.5,
        min_losses_total=1,
        min_losses_per_year=1,
        first_year=2020,
        last_year=2025,
        service_start_year=2020,
        project_years=2,
    )


def test_baseline_is_average_per_active_year(summary_fixture, params_fixture) -> None:
    """База = сумма убытков активных лет / число активных лет."""
    baselines, missing = compute_group_baselines(summary_fixture, None, params_fixture)
    values = baselines.set_index("group_id")["mean_loss_per_active_year"]
    assert missing == 0
    assert values["alive"] == pytest.approx(300_000.0)
    assert values["died_after"] == pytest.approx(100_000.0)
    assert values["died_before"] == pytest.approx(400_000.0)


def test_baseline_average_over_two_active_years(params_fixture) -> None:
    """База усредняется по числу активных лет, а не берётся как сумма."""
    summary = pd.DataFrame(
        [
            {
                **ALIVE_GROUP,
                "active_years_count": 2,
                "active_years": (2022, 2023),
                "net_loss_active": 600_000.0,
            }
        ]
    )
    baselines, _ = compute_group_baselines(summary, None, params_fixture)
    assert baselines["mean_loss_per_active_year"].iloc[0] == pytest.approx(300_000.0)


def test_baseline_without_active_years_is_nan(params_fixture) -> None:
    """Группа без активных лет не получает базу и попадает в статистику."""
    summary = pd.DataFrame(
        [{**ALIVE_GROUP, "active_years_count": 0, "net_loss_active": 0.0}]
    )
    _, missing = compute_group_baselines(summary, None, params_fixture)
    assert missing == 1


def test_compute_group_effect_values(summary_fixture, params_fixture) -> None:
    """Эффект: доля сервиса от базы, умноженная на горизонт для «живых» групп."""
    effect, stats = compute_group_effect(None, summary_fixture, params_fixture)
    values = effect.set_index("group_id")

    assert values.loc["alive", "effect_first_year"] == pytest.approx(150_000.0)
    assert values.loc["alive", "effect_total"] == pytest.approx(300_000.0)
    assert values.loc["died_after", "effect_first_year"] == pytest.approx(50_000.0)
    assert values.loc["died_after", "effect_total"] == pytest.approx(50_000.0)
    assert values.loc["died_before", "effect_total"] == pytest.approx(0.0)

    assert stats.groups_total == 3
    assert stats.groups_in_progress == 1
    assert stats.groups_closed == 1
    assert stats.groups_no_baseline == 0


def test_compute_group_effect_requires_relevance(summary_fixture, params_fixture) -> None:
    """Без класса релевантности расчёт не запускается."""
    with pytest.raises(KeyError, match="group_relevance"):
        compute_group_effect(None, summary_fixture.drop(columns=["group_relevance"]), params_fixture)


def test_compute_group_effect_empty(params_fixture) -> None:
    """Пустая сводка даёт пустой результат."""
    effect, stats = compute_group_effect(None, pd.DataFrame(), params_fixture)
    assert effect.empty
    assert stats.groups_total == 0


def test_effect_matrix_sums_not_averages(summary_fixture, params_fixture) -> None:
    """В матрице эффект суммируется по группам (а не усредняется)."""
    effect, _ = compute_group_effect(None, summary_fixture, params_fixture)
    matrix = effect_matrix(effect, params_fixture)

    assert matrix.loc[2022, RELEVANCE_DIED_AFTER] == pytest.approx(50_000.0)
    assert matrix.loc[2023, RELEVANCE_ALIVE] == pytest.approx(150_000.0)
    assert matrix.loc[2024, RELEVANCE_ALIVE] == pytest.approx(150_000.0)
    assert matrix["Итого"].sum() == pytest.approx(effect["effect_total"].sum())


def test_effect_matrix_respects_last_year() -> None:
    """Горизонт прогноза обрезается верхней границей периода."""
    params = EffectParams(
        service_share=1.0,
        min_losses_total=1,
        min_losses_per_year=1,
        first_year=2020,
        last_year=2024,
        service_start_year=2020,
        project_years=5,
    )
    summary = pd.DataFrame([{**ALIVE_GROUP, "detection_year": 2023}])
    effect, _ = compute_group_effect(None, summary, params)
    matrix = effect_matrix(effect, params)
    assert matrix.index.max() == 2024
    assert matrix["Итого"].sum() == pytest.approx(600_000.0)


def test_effect_long_one_row_per_group_year(summary_fixture, params_fixture) -> None:
    """Длинная таблица: по строке на каждый год эффекта группы."""
    effect, _ = compute_group_effect(None, summary_fixture, params_fixture)
    long_frame = effect_long(effect, params_fixture)

    assert len(long_frame) == 4  # alive: 2 года, died_after: 1, died_before: 1
    alive = long_frame.loc[long_frame["group_id"] == "alive"]
    assert sorted(alive["year"]) == [2023, 2024]
    assert alive["effect"].sum() == pytest.approx(300_000.0)


def test_effect_by_cohort(summary_fixture, params_fixture) -> None:
    """Когорты группируются по году обнаружения."""
    effect, _ = compute_group_effect(None, summary_fixture, params_fixture)
    by_class, cohort = effect_by_cohort(effect, params_fixture)

    assert set(by_class["detection_year"]) == {2022, 2023, 2024}
    row_2023 = cohort.set_index("detection_year").loc[2023]
    assert row_2023["groups"] == 1
    assert row_2023["groups_in_progress"] == 1
    assert row_2023["effect_projected"] == pytest.approx(300_000.0)


def test_effect_by_cohort_aggregates_same_year() -> None:
    """Несколько групп одного года обнаружения не размножают строки когорты."""
    params = EffectParams(
        service_share=0.5,
        min_losses_total=1,
        min_losses_per_year=1,
        first_year=2020,
        last_year=2025,
        service_start_year=2020,
        project_years=2,
    )
    summary = pd.DataFrame(
        [
            {**ALIVE_GROUP, "group_id": "alive_a", "net_loss_total": 300_000.0, "net_loss_active": 300_000.0},
            {**ALIVE_GROUP, "group_id": "alive_b", "net_loss_total": 200_000.0, "net_loss_active": 200_000.0},
        ]
    )
    effect, _ = compute_group_effect(None, summary, params)
    _, cohort = effect_by_cohort(effect, params)

    assert len(cohort) == 1
    assert cohort.iloc[0]["groups"] == 2
    assert cohort.iloc[0]["effect_projected"] == pytest.approx(effect["effect_total"].sum())
    assert cohort.iloc[0]["effect_year_of_detection"] == pytest.approx(
        effect["effect_first_year"].sum()
    )


def test_totals_by_relevance(summary_fixture, params_fixture) -> None:
    """Итоги по классам: число групп и доли эффекта."""
    effect, _ = compute_group_effect(None, summary_fixture, params_fixture)
    totals = totals_by_relevance(effect).set_index("group_relevance")

    assert totals.loc[RELEVANCE_ALIVE, "groups"] == 1
    assert totals.loc[RELEVANCE_ALIVE, "effect_total"] == pytest.approx(300_000.0)
    assert totals.loc[RELEVANCE_DIED_BEFORE, "effect_total"] == pytest.approx(0.0)
    assert totals["share_of_effect"].sum() == pytest.approx(1.0)


def test_reconcile_detects_consistency(summary_fixture, params_fixture) -> None:
    """Сверка матрицы и итога по группам сходится."""
    effect, _ = compute_group_effect(None, summary_fixture, params_fixture)
    matrix = effect_matrix(effect, params_fixture)
    check = reconcile(matrix, effect)
    assert check["delta"] == pytest.approx(0.0)


def test_before_detection_window(synthetic) -> None:
    """Окно before_detection считает базу только по годам до обнаружения."""
    from fin_effects.groups import build_group_summary, classify_group_relevance
    from fin_effects.preprocess import (
        attach_groups_to_losses,
        collapse_losses_to_facts,
        normalize_losses,
    )

    params = EffectParams(
        service_share=1.0,
        min_losses_total=1,
        min_losses_per_year=1,
        first_year=2020,
        last_year=2027,
        service_start_year=2020,
        baseline_window="before_detection",
    )
    normalized = normalize_losses(synthetic.losses)
    facts, _ = collapse_losses_to_facts(normalized)
    _, pairs = attach_groups_to_losses(facts, synthetic.beneficiaries)
    summary, _ = build_group_summary(pairs, synthetic.cards, params)
    summary = classify_group_relevance(summary)

    baselines, missing = compute_group_baselines(summary, pairs, params)
    assert missing >= 1
    assert baselines["mean_loss_per_active_year"].sum() >= 0

    full_params = EffectParams(
        service_share=1.0,
        min_losses_total=1,
        min_losses_per_year=1,
        first_year=2020,
        last_year=2027,
        service_start_year=2020,
        baseline_window="full",
    )
    full, _ = compute_group_baselines(summary, pairs, full_params)
    assert baselines["mean_loss_per_active_year"].sum() <= full["mean_loss_per_active_year"].sum()


def test_params_validation() -> None:
    """Некорректные параметры методики отклоняются."""
    with pytest.raises(ValueError, match="service_share"):
        EffectParams(service_share=1.5)
    with pytest.raises(ValueError, match="baseline_window"):
        EffectParams(baseline_window="unknown")
    with pytest.raises(ValueError, match="last_year"):
        EffectParams(first_year=2025, last_year=2020)
    with pytest.raises(ValueError, match="project_years"):
        EffectParams(project_years=0)


def test_baseline_full_versus_before_detection_synthetic(synthetic) -> None:
    """На «живых» группах база до обнаружения не больше полной базы."""
    from fin_effects.effect import compute_group_effect
    from fin_effects.groups import (
        build_group_summary,
        classify_group_relevance,
        filter_groups_by_size,
        filter_observation_period,
    )
    from fin_effects.preprocess import (
        attach_groups_to_losses,
        collapse_losses_to_facts,
        normalize_losses,
    )

    normalized = normalize_losses(synthetic.losses)
    facts, _ = collapse_losses_to_facts(normalized)
    facts, _ = filter_observation_period(facts, EffectParams(first_year=2020, last_year=2027, service_start_year=2020))
    _, pairs = attach_groups_to_losses(facts, synthetic.beneficiaries)
    summary, _ = build_group_summary(
        pairs, synthetic.cards, EffectParams(last_year=2027, service_start_year=2020)
    )
    summary, _ = filter_groups_by_size(summary, EffectParams(last_year=2027, service_start_year=2020))
    summary = classify_group_relevance(summary)

    full_params = EffectParams(
        service_share=1.0, last_year=2027, project_years=1, service_start_year=2020
    )
    early_params = EffectParams(
        service_share=1.0,
        last_year=2027,
        project_years=1,
        service_start_year=2020,
        baseline_window="before_detection",
    )
    full_effect, _ = compute_group_effect(pairs, summary, full_params)
    early_effect, _ = compute_group_effect(pairs, summary, early_params)

    full_total = float(full_effect["effect_total"].sum())
    early_total = float(early_effect["effect_total"].sum())
    assert early_total <= full_total
    assert early_total >= 0
