"""Оркестрация пайплайна: от сырых выгрузок до итогового финансового эффекта.

Один вызов :func:`run_pipeline` заменяет всю последовательность ячеек тетради.
Промежуточные результаты складываются в :class:`PipelineResult`, поэтому
ноутбуку не нужно «перечитывать pickle», чтобы получить доступ к таблице.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import (
    ARTIFACT_BENEFICIARIES,
    ARTIFACT_EFFECT,
    ARTIFACT_GROUP_SUMMARY,
    ARTIFACT_LOSS_FACTS,
    DEFAULT_PARAMS,
    EffectParams,
)
from .effect import (
    EffectStats,
    compute_group_effect,
    effect_by_cohort,
    effect_matrix,
    reconcile,
    totals_by_relevance,
)
from .excel_beneficiaries import (
    BeneficiaryLoadStats,
    load_beneficiaries_from_dir,
)
from .groups import (
    GroupBuildStats,
    build_group_summary,
    classify_group_relevance,
    filter_groups_by_size,
    filter_observation_period,
    year_matrix,
)
from .io_utils import cache_frame
from .preprocess import (
    LossPreprocessStats,
    add_group_number_column,
    attach_groups_to_losses,
    collapse_losses_to_facts,
    normalize_losses,
)

LOGGER = logging.getLogger(__name__)

#: Колонка с номером группы в выгрузке сервиса.
GROUP_NUMBER_COLUMN = "group_number"


@dataclass
class PipelineResult:
    """Результаты полного прогона пайплайна."""

    params: EffectParams
    beneficiaries: pd.DataFrame
    loss_facts: pd.DataFrame
    facts_with_groups: pd.DataFrame
    summary: pd.DataFrame
    group_effect: pd.DataFrame
    effect_matrix: pd.DataFrame
    effect_by_class: pd.DataFrame
    effect_cohort: pd.DataFrame
    relevance_totals: pd.DataFrame
    year_matrix: pd.DataFrame
    reconciliation: dict[str, float]
    beneficiary_stats: BeneficiaryLoadStats = field(default_factory=BeneficiaryLoadStats)
    loss_stats: LossPreprocessStats = field(default_factory=LossPreprocessStats)
    group_stats: GroupBuildStats = field(default_factory=GroupBuildStats)
    effect_stats: EffectStats = field(default_factory=EffectStats)

    @property
    def total_effect(self) -> float:
        """Итоговый финансовый эффект, ₽.

        Returns:
            Сумма эффекта по всем группам за горизонт прогноза.
        """
        return float(self.group_effect["effect_total"].sum()) if not self.group_effect.empty else 0.0

    @property
    def total_prevented_loss(self) -> float:
        """Предотвращённый убыток до применения доли сервиса, ₽.

        Returns:
            Сумма средних годовых убытков групп, дающих эффект
            (классы «действует» и «распалась после обнаружения»).
        """
        from .config import RELEVANT_CLASSES

        if self.group_effect.empty:
            return 0.0
        relevant = self.group_effect.loc[
            self.group_effect["group_relevance"].isin(RELEVANT_CLASSES),
            "mean_loss_per_active_year",
        ]
        return float(relevant.sum())

    def as_text(self) -> str:
        """Человекочитаемая сводка прогона.

        Returns:
            Многострочная строка.
        """
        groups = (
            self.beneficiaries["group_id"].nunique()
            if "group_id" in self.beneficiaries.columns
            else 0
        )
        classes = (
            self.summary["group_relevance"].value_counts().to_dict()
            if not self.summary.empty and "group_relevance" in self.summary.columns
            else {}
        )
        # «руб.» вместо «₽»: на Windows консоль cp1251 падает на символе рубля.
        unique_matched = (
            self.facts_with_groups["LossNumber"].nunique()
            if not self.facts_with_groups.empty
            else 0
        )
        return "\n".join(
            [
                f"Групп в выгрузке сервиса: {groups}",
                f"Убытков (фактов): {len(self.loss_facts)}",
                f"Убытков привязано к группам (уник.): {unique_matched}",
                f"Пар убыток-группа: {len(self.facts_with_groups)}",
                f"Групп после фильтров: {len(self.summary)}",
                f"Классы: {classes}",
                f"Итоговый эффект: {self.total_effect:,.0f} руб.".replace(",", " "),
            ]
        )


def run_pipeline(
    beneficiaries: pd.DataFrame,
    losses: pd.DataFrame,
    cards: pd.DataFrame | None = None,
    params: EffectParams = DEFAULT_PARAMS,
    *,
    persist: bool = False,
    min_members_matched: int = 1,
    match_mode: str = "auto",
) -> PipelineResult:
    """Прогоняет полный расчёт финансового эффекта.

    Args:
        beneficiaries: таблица участников групп из выгрузки сервиса.
        losses: выгрузка убытков (может содержать дубли строк по платежам).
        cards: таблица карточек УПС; ``None`` — дата обнаружения по первому убытку.
        params: параметры методики.
        persist: сохранять ли промежуточные артефакты в ``data/interim``.
        min_members_matched: сколько участников группы должно быть в убытке.
        match_mode: см. :func:`~fin_effects.preprocess.attach_groups_to_losses`.

    Returns:
        :class:`PipelineResult`.

    Examples:
        >>> from fin_effects.synth import SynthConfig, generate_synthetic_data
        >>> data = generate_synthetic_data(SynthConfig(groups=3, noise_losses=10))
        >>> result = run_pipeline(data.beneficiaries, data.losses, data.cards)
        >>> result.total_effect >= 0
        True
    """
    LOGGER.info("Старт пайплайна: %s", params)

    normalized_losses = normalize_losses(losses)
    loss_facts, loss_stats = collapse_losses_to_facts(normalized_losses)
    loss_facts, dropped_by_year = filter_observation_period(loss_facts, params)

    membership, facts_with_groups = attach_groups_to_losses(
        loss_facts,
        beneficiaries,
        min_members_matched=min_members_matched,
        match_mode=match_mode,
    )
    del membership
    facts_with_groups = add_group_number_column(facts_with_groups, beneficiaries, GROUP_NUMBER_COLUMN)

    summary, group_stats = build_group_summary(
        facts_with_groups,
        cards,
        params,
        loss_facts=loss_facts,
        beneficiaries=beneficiaries,
    )
    summary, dropped_short = filter_groups_by_size(summary, params)
    summary = classify_group_relevance(summary)
    group_stats = GroupBuildStats(
        groups_unmatched=group_stats.groups_unmatched,
        groups_without_card=group_stats.groups_without_card,
        groups_dropped_short=dropped_short,
        losses_dropped_by_year=dropped_by_year,
    )

    effect, effect_stats = compute_group_effect(facts_with_groups, summary, params)
    matrix = effect_matrix(effect, params)
    by_class, cohort = effect_by_cohort(effect, params)
    relevance_totals = totals_by_relevance(effect)
    years = year_matrix(summary) if not summary.empty else pd.DataFrame()
    check = reconcile(matrix, effect)

    if abs(check["delta"]) > 1.0:
        LOGGER.warning(
            "Расхождение матрицы и итога по группам: %.2f ₽ — проверьте горизонт прогноза",
            check["delta"],
        )

    result = PipelineResult(
        params=params,
        beneficiaries=beneficiaries,
        loss_facts=loss_facts,
        facts_with_groups=facts_with_groups,
        summary=summary,
        group_effect=effect,
        effect_matrix=matrix,
        effect_by_class=by_class,
        effect_cohort=cohort,
        relevance_totals=relevance_totals,
        year_matrix=years,
        reconciliation=check,
        beneficiary_stats=BeneficiaryLoadStats(),
        loss_stats=loss_stats,
        group_stats=group_stats,
        effect_stats=effect_stats,
    )

    if persist:
        persist_artifacts(result)

    LOGGER.info("Пайплайн завершён:\n%s", result.as_text())
    return result


def persist_artifacts(result: PipelineResult) -> dict[str, Path]:
    """Сохраняет промежуточные и итоговые артефакты.

    Args:
        result: результат прогона.

    Returns:
        Словарь «имя артефакта -> путь».
    """
    paths = {
        "beneficiaries": cache_frame(result.beneficiaries, ARTIFACT_BENEFICIARIES),
        "loss_facts": cache_frame(result.loss_facts, ARTIFACT_LOSS_FACTS),
        "group_summary": cache_frame(result.summary, ARTIFACT_GROUP_SUMMARY),
        "group_effect": cache_frame(result.group_effect, ARTIFACT_EFFECT),
    }
    LOGGER.info("Артефакты сохранены: %s", ", ".join(str(path) for path in paths.values()))
    return paths


def run_pipeline_from_files(
    beneficiaries_dir: str | Path,
    losses: pd.DataFrame,
    cards: pd.DataFrame | None = None,
    params: EffectParams = DEFAULT_PARAMS,
    *,
    years: Iterable[str | int] = (),
    engine: str = "openpyxl",
    workers: int = 1,
    persist: bool = False,
) -> PipelineResult:
    """Читает выгрузки сервиса из каталога и запускает пайплайн.

    Args:
        beneficiaries_dir: каталог с xlsx-выгрузками сервиса.
        losses: выгрузка убытков.
        cards: таблица карточек УПС.
        params: параметры методики.
        years: подкаталоги-годы в каталоге выгрузок.
        engine: движок чтения Excel.
        workers: число процессов чтения Excel.
        persist: сохранять ли артефакты.

    Returns:
        :class:`PipelineResult` (со статистикой чтения в ``beneficiary_stats``).
    """
    beneficiaries, stats = load_beneficiaries_from_dir(
        beneficiaries_dir, years, engine=engine, workers=workers, group_by_file=True
    )
    result = run_pipeline(beneficiaries, losses, cards, params, persist=persist)
    result.beneficiary_stats = stats
    return result
