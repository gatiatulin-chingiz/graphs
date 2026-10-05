"""Ядро методики: средний годовой убыток группы и финансовый эффект сервиса.

Методика в формализованном виде
-------------------------------

1. **База эффекта.** Для каждой группы считается средний убыток за *активный год*
   ``baseline = net_loss_total / active_years_count``, где активным считается год,
   в котором у группы было не меньше ``params.min_losses_per_year`` убытков.
   Смысл: если бы сервис не обнаружил группу, такой убыток повторялся бы каждый
   следующий год.

2. **Класс релевантности** (см. :func:`~fin_effects.groups.classify_group_relevance`):

   ========================================  ===================================
   Класс                                     Эффект
   ========================================  ===================================
   ``Продолжает действовать``                 предотвращается ``baseline`` в год
                                              обнаружения и каждый следующий год
   ``Распалась после обнаружения``            предотвращается ``baseline`` в год
                                              обнаружения (убытки этого года уже
                                              не повторятся)
   ``Распалась самостоятельно``               0 — убытки прекратились до
                                              обнаружения, сервис ни при чём
   ========================================  ===================================

3. **Доля сервиса.** ``params.service_share`` (в исходной тетради — ``0.35``) —
   доля вклада (заслуги) графового сервиса в предотвращённый убыток. Итоговый
   эффект года ``Y`` для группы ``g``::

       effect(g, Y) = baseline(g) * service_share

4. **Суммарный эффект** за период — сумма по группам и годам прогноза
   (``params.project_years`` лет начиная с года обнаружения).

Все предположения вынесены в :class:`~fin_effects.config.EffectParams` и
проговариваются в отчёте — в исходной тетради они были «магическими» числами.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_PARAMS,
    RELEVANCE_CLOSED,
    RELEVANCE_IN_PROGRESS,
    RELEVANCE_LABELS,
    RELEVANT_CLASSES,
    EffectParams,
)
from .preprocess import GROUP_KEY, LOSS_KEY, LOSS_YEAR, NET_LOSS

LOGGER = logging.getLogger(__name__)

#: Колонки таблицы эффекта по группам.
GROUP_EFFECT_COLUMNS = (
    "group_id",
    "group_relevance",
    "detection_year",
    "active_years_count",
    "losses_total",
    "net_loss_total",
    "mean_loss_per_active_year",
    "service_share",
    "effect_first_year",
    "effect_total",
)


@dataclass(frozen=True)
class EffectStats:
    """Статистика расчёта эффекта."""

    groups_total: int = 0
    groups_in_progress: int = 0
    groups_closed: int = 0
    groups_no_baseline: int = 0

    def as_text(self) -> str:
        """Человекочитаемая сводка."""
        return (
            f"групп: {self.groups_total} "
            f"(в работе {self.groups_in_progress}, закрыта {self.groups_closed}, "
            f"без базы {self.groups_no_baseline})"
        )


def active_year_losses(
    summary: pd.DataFrame,
    params: EffectParams = DEFAULT_PARAMS,
) -> pd.DataFrame:
    """Разворачивает активные годы группы в «длинную» таблицу «группа × год × убыток».

    Args:
        summary: сводка по группам (см. :func:`~fin_effects.groups.build_group_summary`).
        params: параметры методики.

    Returns:
        Таблица ``group_id``, ``loss_year``, ``baseline``, ``baseline_years_count``,
        ``detection_year``, ``group_relevance``.

    Note:
        Заменяет вложенный цикл тетради::

            for i, group_id in enumerate(min_3_losses['groups_list']):
                j = losses[(losses['groups_list'] == str(group_id)) &
                           (losses['loss_year'].isin(min_3_losses['loss_year'][i]))]
                losses_ = pd.concat([losses_, j], axis=0)

        который для каждой группы повторно фильтровал всю таблицу убытков.
    """
    if summary.empty:
        return pd.DataFrame(
            columns=[
                "group_id",
                LOSS_YEAR,
                "baseline",
                "baseline_years_count",
                "detection_year",
                "group_relevance",
            ]
        )

    loss_column = "net_loss_active" if "net_loss_active" in summary.columns else "net_loss_total"
    working = summary.loc[:, [
        "group_id",
        "active_years",
        "active_years_count",
        loss_column,
        "detection_year",
        "group_relevance",
    ]].copy()
    working = working.rename(columns={"group_id": GROUP_KEY, loss_column: "net_loss_active"})
    working["baseline"] = working["net_loss_active"] / working["active_years_count"].replace(0, np.nan)

    exploded = working.loc[:, [GROUP_KEY, "active_years", "baseline", "active_years_count",
                               "detection_year", "group_relevance"]].explode("active_years")
    exploded = exploded.rename(columns={"active_years": LOSS_YEAR})
    exploded[LOSS_YEAR] = pd.to_numeric(exploded[LOSS_YEAR], errors="coerce").astype("Int64")
    exploded = exploded.dropna(subset=[LOSS_YEAR]).reset_index(drop=True)
    exploded[LOSS_YEAR] = exploded[LOSS_YEAR].astype(int)
    return exploded


def compute_group_baselines(
    summary: pd.DataFrame,
    facts_with_groups: pd.DataFrame | None = None,
    params: EffectParams = DEFAULT_PARAMS,
) -> tuple[pd.DataFrame, int]:
    """Считает средний годовой убыток (``baseline``) по каждой группе.

    Args:
        summary: сводка по группам.
        facts_with_groups: пары «убыток-группа» — нужны для варианта
            ``baseline_window='before_detection'``.
        params: параметры методики.

    Returns:
        Кортеж «таблица ``group_id`` + ``mean_loss_per_active_year``,
        сколько групп осталось без базы».

    Note:
        В тетради база считалась как ``losses_money.sum() / year_count``, где
        ``year_count`` — число лет группы, попавших в выборку; при этом сама
        выборка формировалась циклом и содержала все года подряд, включая годы
        без убытков. Разница заметна на группах с «рваной» активностью.
    """
    if summary.empty:
        return pd.DataFrame(columns=["group_id", "mean_loss_per_active_year"]), 0

    working = summary.copy()
    if params.baseline_window == "before_detection" and facts_with_groups is not None:
        working = _baseline_before_detection(working, facts_with_groups, params)

    if "baseline_sum" in working.columns:
        numerator = working["baseline_sum"]
        denominator = working["baseline_years_count"]
    elif "net_loss_active" in working.columns:
        numerator = working["net_loss_active"]
        denominator = working["active_years_count"]
    else:
        LOGGER.warning(
            "В сводке нет активных лет — база считается по всем убыткам группы"
        )
        numerator = working["net_loss_total"]
        denominator = working["losses_total"]

    working["mean_loss_per_active_year"] = np.where(
        denominator > 0, numerator / denominator.replace(0, np.nan), np.nan
    )
    no_baseline = int(working["mean_loss_per_active_year"].isna().sum())
    if no_baseline:
        LOGGER.warning("Групп без базы для расчёта эффекта: %d", no_baseline)

    result = working.loc[:, ["group_id", "mean_loss_per_active_year"]]
    return result, no_baseline


def _baseline_before_detection(
    summary: pd.DataFrame,
    facts_with_groups: pd.DataFrame,
    params: EffectParams,
) -> pd.DataFrame:
    """Пересчитывает базу только по годам строго до года обнаружения.

    Args:
        summary: сводка по группам.
        facts_with_groups: пары «убыток-группа».
        params: параметры методики.

    Returns:
        Сводка с колонками ``baseline_sum`` и ``baseline_years_count``.
    """
    detection = summary.set_index("group_id")["detection_year"]
    working = facts_with_groups.copy()
    working["_detection_year"] = working[GROUP_KEY].map(detection)
    working = working.loc[
        pd.to_numeric(working[LOSS_YEAR], errors="coerce")
        < pd.to_numeric(working["_detection_year"], errors="coerce")
    ]
    per_loss = working.groupby([GROUP_KEY, LOSS_KEY], as_index=False).agg(
        loss_year=(LOSS_YEAR, "min"), net_loss=(NET_LOSS, "sum")
    )
    per_year = per_loss.groupby([GROUP_KEY, LOSS_YEAR], as_index=False).agg(
        losses=(LOSS_KEY, "nunique"), net_loss=("net_loss", "sum")
    )
    per_year = per_year.loc[per_year["losses"] >= params.min_losses_per_year]
    aggregated = per_year.groupby(GROUP_KEY, as_index=False).agg(
        baseline_sum=("net_loss", "sum"), baseline_years_count=(LOSS_YEAR, "nunique")
    )
    return summary.merge(aggregated, left_on="group_id", right_on=GROUP_KEY, how="left")


def compute_group_effect(
    facts_with_groups: pd.DataFrame,
    summary: pd.DataFrame,
    params: EffectParams = DEFAULT_PARAMS,
) -> tuple[pd.DataFrame, EffectStats]:
    """Считает финансовый эффект по каждой группе.

    Args:
        facts_with_groups: пары «убыток-группа» (для ``before_detection``).
        summary: сводка по группам (с классом релевантности).
        params: параметры методики.

    Returns:
        Кортеж «таблица эффекта по группам, статистика».

    Raises:
        KeyError: если в сводке нет колонки ``group_relevance``.
    """
    if summary.empty:
        return pd.DataFrame(columns=GROUP_EFFECT_COLUMNS), EffectStats()
    if "group_relevance" not in summary.columns:
        raise KeyError(
            "Сводка не содержит group_relevance — вызовите classify_group_relevance()"
        )

    baselines, no_baseline = compute_group_baselines(summary, facts_with_groups, params)
    frame = summary.loc[:, [
        "group_id",
        "group_relevance",
        "detection_year",
        "active_years",
        "losses_total",
        "net_loss_total",
    ]].merge(baselines, on="group_id", how="left")
    frame = frame.rename(columns={"active_years": "active_years_list"})

    frame["service_share"] = params.service_share
    frame["effect_first_year"] = np.where(
        frame["group_relevance"].isin(RELEVANT_CLASSES),
        frame["mean_loss_per_active_year"] * params.service_share,
        0.0,
    )
    # Горизонт усекается верхней границей периода наблюдения — иначе
    # effect_total расходится с матрицей/effect_long при detection близко к last_year.
    detection = pd.to_numeric(frame["detection_year"], errors="coerce")
    max_years = (params.last_year - detection + 1).clip(lower=0)
    in_progress_years = np.minimum(params.project_years, max_years)
    closed_years = np.minimum(1, max_years)
    frame["effect_total"] = np.where(
        frame["group_relevance"] == RELEVANCE_IN_PROGRESS,
        frame["effect_first_year"] * in_progress_years,
        np.where(
            frame["group_relevance"] == RELEVANCE_CLOSED,
            frame["effect_first_year"] * closed_years,
            0.0,
        ),
    )
    frame["effect_total"] = frame["effect_total"].fillna(0.0)

    stats = EffectStats(
        groups_total=len(frame),
        groups_in_progress=int((frame["group_relevance"] == RELEVANCE_IN_PROGRESS).sum()),
        groups_closed=int((frame["group_relevance"] == RELEVANCE_CLOSED).sum()),
        groups_no_baseline=no_baseline,
    )
    LOGGER.info("Расчёт эффекта по группам: %s", stats.as_text())
    return frame.loc[:, [column for column in GROUP_EFFECT_COLUMNS if column in frame.columns]], stats


def effect_matrix(
    group_effect: pd.DataFrame,
    params: EffectParams = DEFAULT_PARAMS,
    *,
    value_column: str = "effect_first_year",
) -> pd.DataFrame:
    """Строит матрицу «год × класс релевантности» с эффектом.

    Args:
        group_effect: таблица эффекта по группам.
        params: параметры методики.
        value_column: колонка с годовым эффектом группы.

    Returns:
        Таблица: индекс — год, колонки — классы релевантности, значения — эффект.

    Note:
        В тетради эффект за год брался из ``pivot_table(..., aggfunc='mean')`` по
        колонке ``mean_loss_money_by_year``, то есть усреднялся между группами, а
        затем суммировался (``total[2023].sum()``). Среднее вместо суммы занижает
        эффект в число раз, равное числу групп.
    """
    if group_effect.empty:
        return pd.DataFrame(columns=list(RELEVANCE_LABELS))

    rows: list[dict[str, object]] = []
    for record in group_effect.itertuples(index=False):
        detection_year = int(record.detection_year)
        is_in_progress = record.group_relevance == RELEVANCE_IN_PROGRESS
        years = range(
            detection_year,
            detection_year + (params.project_years if is_in_progress else 1),
        )
        for year in years:
            if year > params.last_year:
                continue
            rows.append(
                {
                    "year": int(year),
                    "group_id": record.group_id,
                    "group_relevance": record.group_relevance,
                    "effect": float(getattr(record, value_column) or 0.0),
                }
            )

    long_frame = pd.DataFrame(rows)
    matrix = (
        long_frame.pivot_table(
            index="year", columns="group_relevance", values="effect", aggfunc="sum"
        )
        .reindex(columns=list(RELEVANCE_LABELS))
        .fillna(0.0)
    )
    matrix["Итого"] = matrix.sum(axis=1)
    return matrix


def effect_long(
    group_effect: pd.DataFrame,
    params: EffectParams = DEFAULT_PARAMS,
) -> pd.DataFrame:
    """Разворачивает эффект в длинную таблицу «группа × год».

    Args:
        group_effect: таблица эффекта по группам.
        params: параметры методики.

    Returns:
        Таблица ``group_id``, ``year``, ``group_relevance``, ``detection_year``,
        ``effect``.
    """
    if group_effect.empty:
        return pd.DataFrame(
            columns=["group_id", "year", "group_relevance", "detection_year", "effect"]
        )

    rows: list[dict[str, object]] = []
    for record in group_effect.itertuples(index=False):
        detection_year = int(record.detection_year)
        years_count = (
            params.project_years if record.group_relevance == RELEVANCE_IN_PROGRESS else 1
        )
        for offset in range(years_count):
            year = detection_year + offset
            if year > params.last_year:
                continue
            rows.append(
                {
                    "group_id": record.group_id,
                    "year": int(year),
                    "group_relevance": record.group_relevance,
                    "detection_year": detection_year,
                    "effect": float(record.effect_first_year or 0.0),
                }
            )
    return pd.DataFrame(rows)


def effect_by_cohort(
    group_effect: pd.DataFrame,
    params: EffectParams = DEFAULT_PARAMS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Сводка эффекта по когортам года обнаружения.

    Args:
        group_effect: таблица эффекта по группам.
        params: параметры методики.

    Returns:
        Кортеж:

        * таблица «когорта × класс релевантности»: число групп и сумма эффекта;
        * итоговая таблица по когортам: число групп, убытки групп, эффект за
          первый год и накопленный эффект.

    Note:
        В тетради итог выводился как ``total[2023].sum()`` и
        ``total[2023].sum() + total[2024].sum()`` — привязка к конкретным годам
        вместо когорт обнаружения. Здесь когорта = год обнаружения, что делает
        расчёт воспроизводимым для любого года.
    """
    if group_effect.empty:
        return pd.DataFrame(columns=["detection_year"]), pd.DataFrame(columns=["detection_year"])

    long_effect = effect_long(group_effect, params)
    # Важно: сначала агрегируем по году обнаружения, иначе merge размножит
    # строки когорты (по одной на каждую группу внутри года).
    projected = (
        long_effect.groupby("detection_year", as_index=False)["effect"]
        .sum()
        .rename(columns={"effect": "effect_projected"})
        if not long_effect.empty
        else pd.DataFrame(columns=["detection_year", "effect_projected"])
    )

    by_class = (
        group_effect.groupby(["detection_year", "group_relevance"], as_index=False)
        .agg(groups=("group_id", "nunique"), effect_total=("effect_total", "sum"))
        .sort_values(["detection_year", "group_relevance"])
    )

    class_counts = (
        group_effect.assign(_one=1)
        .pivot_table(
            index="detection_year",
            columns="group_relevance",
            values="_one",
            aggfunc="sum",
        )
        .reindex(columns=list(RELEVANCE_LABELS))
        .fillna(0)
        .astype(int)
        .rename(
            columns={
                RELEVANCE_IN_PROGRESS: "groups_in_progress",
                RELEVANCE_CLOSED: "groups_closed",
            }
        )
        .reset_index()
    )

    relevant = group_effect.loc[
        group_effect["group_relevance"].isin(RELEVANT_CLASSES)
    ]
    year_of_detection = (
        relevant.groupby("detection_year", as_index=False)
        .agg(effect_year_of_detection=("effect_first_year", "sum"))
        if not relevant.empty
        else pd.DataFrame(columns=["detection_year", "effect_year_of_detection"])
    )

    cohort = (
        group_effect.groupby("detection_year", as_index=False)
        .agg(
            groups=("group_id", "nunique"),
            net_loss_total=("net_loss_total", "sum"),
            mean_loss_per_active_year=("mean_loss_per_active_year", "sum"),
        )
        .sort_values("detection_year")
    )
    cohort = cohort.merge(class_counts, on="detection_year", how="left")
    cohort = cohort.merge(projected, on="detection_year", how="left")
    cohort = cohort.merge(year_of_detection, on="detection_year", how="left")
    for column in ("groups_in_progress", "groups_closed"):
        cohort[column] = cohort[column].fillna(0).astype(int)
    cohort["effect_projected"] = cohort["effect_projected"].fillna(0.0)
    cohort["effect_year_of_detection"] = cohort["effect_year_of_detection"].fillna(0.0)
    return by_class, cohort


def totals_by_relevance(group_effect: pd.DataFrame) -> pd.DataFrame:
    """Итоги по классам релевантности: число групп и сумма эффекта.

    Args:
        group_effect: таблица эффекта по группам.

    Returns:
        Сводная таблица по классам.
    """
    if group_effect.empty:
        return pd.DataFrame(
            columns=["group_relevance", "groups", "net_loss_total", "effect_total"]
        )
    summary = (
        group_effect.groupby("group_relevance", as_index=False)
        .agg(
            groups=("group_id", "nunique"),
            net_loss_total=("net_loss_total", "sum"),
            mean_loss_per_active_year=("mean_loss_per_active_year", "sum"),
            effect_total=("effect_total", "sum"),
        )
        .rename(columns={"mean_loss_per_active_year": "prevented_loss_total"})
    )
    summary["share_of_effect"] = summary["effect_total"] / max(summary["effect_total"].sum(), 1e-9)
    return summary.sort_values("effect_total", ascending=False).reset_index(drop=True)


def reconcile(matrix: pd.DataFrame, group_effect: pd.DataFrame) -> dict[str, float]:
    """Сверяет итог матрицы с итогом по группам.

    Args:
        matrix: матрица эффекта (см. :func:`effect_matrix`).
        group_effect: таблица эффекта по группам.

    Returns:
        Словарь ``{'matrix_total': ..., 'groups_total': ..., 'delta': ...}``.
    """
    matrix_total = float(matrix["Итого"].sum()) if not matrix.empty else 0.0
    groups_total = float(group_effect["effect_total"].sum()) if not group_effect.empty else 0.0
    return {
        "matrix_total": matrix_total,
        "groups_total": groups_total,
        "delta": matrix_total - groups_total,
    }
