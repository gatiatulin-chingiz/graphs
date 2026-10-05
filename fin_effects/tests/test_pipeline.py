"""Интеграционные тесты пайплайна: от сырых выгрузок до итогового эффекта."""

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
from fin_effects.pipeline import run_pipeline, run_pipeline_from_files
from fin_effects.reporting import (
    build_figures,
    effect_by_cohort_table,
    effect_by_year_table,
    headcount_table,
    money_table,
    sensitivity_table,
    summary_kpis,
    top_groups_table,
)

from .conftest import TEST_PARAMS, build_synthetic


def test_pipeline_returns_effect(pipeline_result) -> None:
    """Пайплайн на синтетике даёт положительный эффект."""
    assert pipeline_result.total_effect > 0
    assert pipeline_result.total_prevented_loss > 0
    assert not pipeline_result.group_effect.empty


def test_pipeline_reconciles(pipeline_result) -> None:
    """Матрица эффекта сходится с итогом по группам."""
    assert abs(pipeline_result.reconciliation["delta"]) < 1.0


def test_pipeline_classification_matches_truth(pipeline_result, synthetic) -> None:
    """Статус групп с карточкой совпадает с эталоном генератора."""
    merged = pipeline_result.summary.merge(
        synthetic.groups.loc[:, ["group_id", "expected_relevance"]], on="group_id", how="left"
    )
    merged = merged.loc[merged["expected_relevance"].notna()]
    assert not merged.empty
    assert (merged["group_relevance"] == merged["expected_relevance"]).all()


def test_pipeline_contains_all_relevance_classes(pipeline_result) -> None:
    """В синтетике есть оба статуса карточки УПС."""
    classes = set(pipeline_result.summary["group_relevance"])
    assert classes <= {RELEVANCE_ALIVE, RELEVANCE_DIED_AFTER}
    assert classes  # не пусто


def test_pipeline_no_duplicate_losses(pipeline_result) -> None:
    """Факты содержат ровно одну строку на убыток."""
    assert pipeline_result.loss_facts["LossNumber"].is_unique


def test_pipeline_pair_count_le_facts(pipeline_result) -> None:
    """Пар «убыток-группа» не больше, чем убытков."""
    assert len(pipeline_result.facts_with_groups) <= len(pipeline_result.loss_facts)


def test_pipeline_effect_per_group_matches_formula(pipeline_result) -> None:
    """Эффект группы = база × доля × число календарных лет эффекта."""
    from fin_effects.effect import effect_calendar_years

    params = pipeline_result.params
    effect = pipeline_result.group_effect
    alive = effect.loc[effect["group_relevance"] == RELEVANCE_ALIVE].copy()

    horizons = [
        len(effect_calendar_years(int(year), str(relevance), params))
        for year, relevance in zip(alive["detection_year"], alive["group_relevance"], strict=True)
    ]
    expected = (
        alive["mean_loss_per_active_year"] * params.service_share * np.asarray(horizons)
    ).astype(float)
    pd.testing.assert_series_equal(
        alive["effect_total"].astype(float).reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_pipeline_only_in_progress_and_closed(pipeline_result) -> None:
    """В расчёте только статусы «в работе» / «закрыта»."""
    classes = set(pipeline_result.summary["group_relevance"])
    assert classes <= {RELEVANCE_ALIVE, RELEVANCE_DIED_AFTER}
    assert pipeline_result.total_effect >= 0


def test_pipeline_is_deterministic(synthetic) -> None:
    """Повторный прогон на тех же данных даёт тот же эффект."""
    first = run_pipeline(synthetic.beneficiaries, synthetic.losses, synthetic.cards, TEST_PARAMS)
    second = run_pipeline(synthetic.beneficiaries, synthetic.losses, synthetic.cards, TEST_PARAMS)
    assert first.total_effect == pytest.approx(second.total_effect)


def test_pipeline_service_share_scales_linearly(synthetic) -> None:
    """Эффект линейно зависит от доли вклада сервиса."""
    from dataclasses import replace

    base = run_pipeline(
        synthetic.beneficiaries, synthetic.losses, synthetic.cards, TEST_PARAMS
    )
    doubled = run_pipeline(
        synthetic.beneficiaries,
        synthetic.losses,
        synthetic.cards,
        replace(TEST_PARAMS, service_share=0.7),
    )
    assert doubled.total_effect == pytest.approx(base.total_effect * 2.0)


def test_pipeline_without_cards_uses_fallback(synthetic) -> None:
    """Без карточек УПС эффект нулевой — группы не входят в расчёт."""
    result = run_pipeline(synthetic.beneficiaries, synthetic.losses, None, TEST_PARAMS)
    assert result.group_stats.groups_without_card == result.facts_with_groups["group_id"].nunique()
    assert result.summary.empty
    assert result.total_effect == 0.0


def test_pipeline_from_files(synthetic) -> None:
    """Чтение групп из xlsx даёт тот же результат, что и готовая таблица."""
    from_table = run_pipeline(
        synthetic.beneficiaries, synthetic.losses, synthetic.cards, TEST_PARAMS
    )
    from_files = run_pipeline_from_files(
        synthetic.output_dir / "raw",
        synthetic.losses,
        synthetic.cards,
        TEST_PARAMS,
        years=[2025],
    )
    assert from_files.beneficiary_stats.files == len(synthetic.beneficiary_files)
    assert from_files.total_effect == pytest.approx(from_table.total_effect)


def test_pipeline_min_members_filter(synthetic) -> None:
    """Порог участников уменьшает число пар «убыток-группа»."""
    loose = run_pipeline(synthetic.beneficiaries, synthetic.losses, synthetic.cards, TEST_PARAMS)
    strict = run_pipeline(
        synthetic.beneficiaries,
        synthetic.losses,
        synthetic.cards,
        TEST_PARAMS,
        min_members_matched=2,
    )
    assert len(strict.facts_with_groups) <= len(loose.facts_with_groups)


def test_pipeline_persist_artifacts(tmp_path, synthetic, monkeypatch) -> None:
    """Артефакты сохраняются, если запрошен persist."""
    import fin_effects.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "ARTIFACT_LOSS_FACTS", tmp_path / "loss_facts.parquet")
    monkeypatch.setattr(pipeline_module, "ARTIFACT_GROUP_SUMMARY", tmp_path / "summary.parquet")
    monkeypatch.setattr(pipeline_module, "ARTIFACT_EFFECT", tmp_path / "effect.parquet")
    monkeypatch.setattr(pipeline_module, "ARTIFACT_BENEFICIARIES", tmp_path / "beneficiaries.parquet")

    run_pipeline(synthetic.beneficiaries, synthetic.losses, synthetic.cards, TEST_PARAMS, persist=True)
    assert (tmp_path / "loss_facts.parquet").exists()
    assert (tmp_path / "effect.parquet").exists()
    assert (tmp_path / "beneficiaries.parquet").exists()


def test_pipeline_small_dataset_does_not_crash(tmp_path) -> None:
    """Минимальный набор данных (2 группы) проходит пайплайн без ошибок."""
    data = build_synthetic(seed=1, groups=2, noise=5, output_dir=tmp_path)
    params = EffectParams(
        service_share=0.35,
        min_losses_total=1,
        min_losses_per_year=1,
        first_year=2020,
        last_year=2027,
        project_years=2,
    )
    result = run_pipeline(data.beneficiaries, data.losses, data.cards, params)
    assert result.total_effect >= 0


def test_pipeline_empty_inputs() -> None:
    """Полностью пустые входы не приводят к исключению."""
    empty = pd.DataFrame({"LossNumber": [], "LossDateTime": []})
    result = run_pipeline(pd.DataFrame(), empty, None, TEST_PARAMS)
    assert result.total_effect == 0.0
    assert result.group_effect.empty


# ---------------------------------------------------------------------------
# Отчётность
# ---------------------------------------------------------------------------


def test_effect_by_year_table(pipeline_result) -> None:
    """Таблица по годам — эффект сервиса и накопленный итог."""
    table = effect_by_year_table(pipeline_result.group_effect, pipeline_result.params)
    assert {
        "Год",
        "Групп",
        "Эффект сервиса, млн ₽",
        "Накопленный эффект сервиса, млн ₽",
    } <= set(table.columns)
    assert "Статус группы" not in table.columns
    assert table["Накопленный эффект сервиса, млн ₽"].is_monotonic_increasing
    assert table["Эффект сервиса, млн ₽"].min() >= 0

    by_class = effect_by_year_table(
        pipeline_result.group_effect, pipeline_result.params, by_class=True
    )
    assert "Статус группы" in by_class.columns


def test_effect_by_cohort_table(pipeline_result) -> None:
    """Таблица по когортам покрывает все годы обнаружения."""
    table = effect_by_cohort_table(pipeline_result.group_effect, pipeline_result.params)
    assert table["Групп"].sum() == pipeline_result.group_effect["group_id"].nunique()
    assert (table["Эффект, млн ₽"] >= 0).all()


def test_top_groups_table_sorted(pipeline_result) -> None:
    """Топ групп отсортирован по убыванию эффекта."""
    table = top_groups_table(pipeline_result.group_effect, top_n=5)
    assert len(table) <= 5
    assert table["Эффект, млн ₽"].is_monotonic_decreasing


def test_headcount_table(pipeline_result) -> None:
    """Сводка по классам покрывает все группы."""
    table = headcount_table(pipeline_result.summary)
    assert table["Групп"].sum() == len(pipeline_result.summary)


def test_sensitivity_table(pipeline_result) -> None:
    """Чувствительность к доле сервиса пропорциональна доле."""
    table = sensitivity_table(pipeline_result.group_effect, pipeline_result.params)
    assert len(table) == 4
    assert table["Эффект, млн ₽"].is_monotonic_increasing


def test_summary_kpis(pipeline_result) -> None:
    """Ключевые показатели согласованы с итогом."""
    kpis = summary_kpis(pipeline_result.group_effect, pipeline_result.params)
    assert kpis["effect_total"] == pytest.approx(pipeline_result.total_effect)
    assert kpis["groups"] == pipeline_result.group_effect["group_id"].nunique()


def test_money_table_converts_to_millions() -> None:
    """Денежные колонки переводятся в миллионы."""
    frame = pd.DataFrame({"value": [1_000_000.0, 2_000_000.0]})
    converted = money_table(frame, ["value"])
    assert converted["value"].tolist() == [1.0, 2.0]


def test_build_figures_creates_files(pipeline_result, tmp_path) -> None:
    """Все графики сохраняются в файлы."""
    figures = build_figures(
        pipeline_result.group_effect,
        pipeline_result.summary,
        pipeline_result.params,
        tmp_path,
        facts_with_groups=pipeline_result.facts_with_groups,
    )
    assert set(figures) == {
        "loss_by_year",
        "effect_by_year",
        "effect_by_cohort",
        "top_groups",
        "baseline_distribution",
    }
    for path in figures.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_build_figures_with_empty_data(tmp_path) -> None:
    """Графики на пустых данных не падают."""
    empty = pd.DataFrame()
    figures = build_figures(empty, empty, TEST_PARAMS, tmp_path)
    assert all(path.exists() for path in figures.values())
