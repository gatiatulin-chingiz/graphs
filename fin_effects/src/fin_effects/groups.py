"""Формирование групп: дата обнаружения, «жизнь» группы, активные годы.

В исходной тетради этот блок был самым запутанным:

* ``min_3_losses`` фильтровал группы по годам вложенным циклом ``for i, group_id
  in enumerate(...)`` с обращением к ``min_3_losses['loss_year'][i]`` — это
  ``O(групп × убытков)`` и на реальных объёмах работает минуты;
* колонка ``detection_date`` считалась по карточкам УПС, а затем молча
  перезаписывалась константой ``losses_['detection_date'] = 2024``;
* ``group_relevance`` присваивался построчно (``.loc[mask, 'group_relevance']``),
  из-за чего одна и та же группа могла получить разные классы в разных строках.

Здесь всё считается векторно и на уровне группы.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_PARAMS,
    RELEVANCE_ALIVE,
    RELEVANCE_DIED_AFTER,
    RELEVANCE_DIED_BEFORE,
    RELEVANCE_LABELS,
    EffectParams,
)
from .preprocess import LOSS_KEY, LOSS_YEAR, NET_LOSS
from .textnorm import normalize_series

LOGGER = logging.getLogger(__name__)

#: Имя колонки с идентификатором группы.
GROUP_ID = "group_id"

#: Колонки таблицы карточек УПС -> внутренние имена.
CARD_DATE_COLUMN = "Дата создания карточки"

#: Колонки сводки по группе.
GROUP_SUMMARY_COLUMNS = (
    "group_id",
    "detection_year",
    "detection_date",
    "detection_source",
    "first_loss_year",
    "last_loss_year",
    "lifetime_years",
    "active_years",
    "active_years_count",
    "active_losses",
    "net_loss_active",
    "losses_total",
    "net_loss_total",
    "members_matched",
)


@dataclass(frozen=True)
class GroupBuildStats:
    """Статистика построения групп."""

    groups_unmatched: int = 0
    groups_without_card: int = 0
    groups_dropped_short: int = 0
    losses_dropped_by_year: int = 0

    def as_text(self) -> str:
        """Человекочитаемая сводка.

        Returns:
            Строка для логов ноутбука.
        """
        return (
            f"без карточки УПС: {self.groups_without_card}; "
            f"отброшено по порогу убытков: {self.groups_dropped_short}; "
            f"убытков вне периода: {self.losses_dropped_by_year}"
        )


def loss_detection_dates(cards: pd.DataFrame) -> pd.DataFrame:
    """Считает дату первой карточки УПС для каждого убытка.

    Args:
        cards: таблица карточек УПС (колонки «Номер Убытка», «Дата создания карточки»).

    Returns:
        Таблица ``LossNumber``, ``card_date`` — по одной строке на убыток.

    Raises:
        KeyError: если в таблице карточек нет обязательных колонок.
    """
    required = {"Номер Убытка", CARD_DATE_COLUMN}
    missing = required - set(cards.columns)
    if missing:
        raise KeyError(f"В таблице карточек нет колонок: {sorted(missing)}")

    working = cards.loc[:, ["Номер Убытка", CARD_DATE_COLUMN]].copy()
    working.columns = [LOSS_KEY, "card_date"]
    working["card_date"] = pd.to_datetime(working["card_date"], errors="coerce")
    # Номер убытка нормализуется так же, как в таблице фактов (регистр + пробелы),
    # иначе JOIN по LossNumber не находит ни одной карточки.
    working[LOSS_KEY] = normalize_series(working[LOSS_KEY])

    result = (
        working.dropna(subset=["card_date"])
        .groupby(LOSS_KEY, as_index=False)["card_date"]
        .min()
    )
    LOGGER.info("Карточки УПС: %d убытков с датой создания", len(result))
    return result


def filter_observation_period(
    facts: pd.DataFrame,
    params: EffectParams = DEFAULT_PARAMS,
) -> tuple[pd.DataFrame, int]:
    """Оставляет убытки внутри периода наблюдения.

    Args:
        facts: таблица фактов.
        params: параметры методики.

    Returns:
        Кортеж «отфильтрованная таблица, сколько строк отброшено».

    Note:
        В тетради фильтр ``>= '2020-01-01'`` применялся трижды и в разных местах
        (в том числе к таблице, где ``LossDateTime`` уже был ``date``), а верхняя
        граница периода вообще не задавалась. Здесь границы заданы явно.
    """
    if facts.empty:
        return facts, 0

    years = pd.to_numeric(facts[LOSS_YEAR], errors="coerce")
    mask = years.between(params.first_year, params.last_year)
    dropped = int((~mask).sum())
    if dropped:
        LOGGER.warning(
            "Убытков вне периода %d-%d: %d (отброшено)", params.first_year, params.last_year, dropped
        )
    return facts.loc[mask].reset_index(drop=True), dropped


def build_group_summary(
    facts_with_groups: pd.DataFrame,
    cards: pd.DataFrame | None = None,
    params: EffectParams = DEFAULT_PARAMS,
) -> tuple[pd.DataFrame, GroupBuildStats]:
    """Строит сводку по группам: обнаружение, жизнь, активные годы.

    Args:
        facts_with_groups: пары «убыток-группа» (см.
            :func:`~fin_effects.preprocess.attach_groups_to_losses`).
        cards: таблица карточек УПС (даты обнаружения). ``None`` — дата
            обнаружения берётся по году первого убытка группы.
        params: параметры методики.

    Returns:
        Кортеж «сводка по группам, статистика».

    Note:
        ``detection_year`` — год первой карточки УПС по убыткам группы.
        Если карточек нет, используется год первого убытка группы, и колонка
        ``detection_source`` помечает это как ``fallback_first_loss`` — раньше
        такие группы молча выпадали из расчёта (в тетради после merge с датами
        карточек стоял фильтр ``detection_date.notnull()``).
    """
    if facts_with_groups.empty:
        return pd.DataFrame(columns=GROUP_SUMMARY_COLUMNS), GroupBuildStats()

    working = facts_with_groups.copy()
    working["_is_group_loss"] = working["group_id"].notna()

    base = working.loc[working["_is_group_loss"]]
    per_loss = (
        base.groupby([GROUP_ID, LOSS_KEY], as_index=False)
        .agg(
            loss_year=(LOSS_YEAR, "min"),
            loss_date=("LossDateTime", "min"),
            net_loss=(NET_LOSS, "sum"),
            members_matched=("matched_members", "max"),
        )
    )

    per_year = (
        per_loss.groupby([GROUP_ID, "loss_year"], as_index=False)
        .agg(losses=("LossNumber", "nunique"), net_loss=("net_loss", "sum"))
        .sort_values([GROUP_ID, "loss_year"])
    )

    active_years = per_year.loc[per_year["losses"] >= params.min_losses_per_year].copy()

    summary = (
        per_loss.groupby(GROUP_ID, as_index=False)
        .agg(
            first_loss_year=("loss_year", "min"),
            last_loss_year=("loss_year", "max"),
            losses_total=("LossNumber", "nunique"),
            net_loss_total=("net_loss", "sum"),
            members_matched=("members_matched", "max"),
        )
    )

    active_agg = (
        active_years.groupby(GROUP_ID, as_index=False)
        .agg(
            active_years=("loss_year", lambda years: tuple(sorted(int(year) for year in years))),
            active_years_count=("loss_year", "nunique"),
            active_losses=("losses", "sum"),
            net_loss_active=("net_loss", "sum"),
        )
    )
    summary = summary.merge(active_agg, on=GROUP_ID, how="left")
    summary["active_years"] = summary["active_years"].map(
        lambda value: value if isinstance(value, tuple) else ()
    )
    summary["active_years_count"] = summary["active_years_count"].fillna(0).astype(int)
    summary["lifetime_years"] = summary["last_loss_year"] - summary["first_loss_year"] + 1

    detection = _detection_dates(per_loss, cards)
    summary = summary.merge(detection, on=GROUP_ID, how="left")

    groups_without_card = int((summary["detection_source"] == "fallback_first_loss").sum())
    if groups_without_card:
        LOGGER.warning(
            "У %d групп нет карточек УПС — дата обнаружения взята по первому убытку",
            groups_without_card,
        )

    summary = summary.loc[
        :, [column for column in GROUP_SUMMARY_COLUMNS if column in summary.columns]
    ]
    summary = summary.sort_values("group_id").reset_index(drop=True)

    stats = GroupBuildStats(
        groups_unmatched=0,
        groups_without_card=groups_without_card,
        groups_dropped_short=0,
        losses_dropped_by_year=0,
    )
    return summary, stats


def _detection_dates(per_loss: pd.DataFrame, cards: pd.DataFrame | None) -> pd.DataFrame:
    """Определяет дату/год обнаружения для каждой группы.

    Args:
        per_loss: таблица «группа-убыток» с датами и потерями.
        cards: таблица карточек УПС или ``None``.

    Returns:
        Таблица ``group_id``, ``detection_date``, ``detection_year``,
        ``detection_source``.
    """
    if cards is not None and not cards.empty:
        card_dates = loss_detection_dates(cards)
        enriched = per_loss.merge(card_dates, on=LOSS_KEY, how="left")
        detected = (
            enriched.dropna(subset=["card_date"])
            .groupby(GROUP_ID, as_index=False)
            .agg(detection_date=("card_date", "min"))
        )
    else:
        detected = pd.DataFrame(
            {
                GROUP_ID: pd.Series(dtype="object"),
                "detection_date": pd.Series(dtype="datetime64[ns]"),
            }
        )

    detected["detection_year"] = detected["detection_date"].dt.year.astype("Int64")
    detected["detection_source"] = "ups_card"

    fallback = (
        per_loss.groupby(GROUP_ID, as_index=False)
        .agg(first_loss_date=("loss_date", "min"), fallback_year=("loss_year", "min"))
        .loc[:, [GROUP_ID, "first_loss_date", "fallback_year"]]
    )
    merged = fallback.merge(detected, on=GROUP_ID, how="left")
    needs_fallback = merged["detection_year"].isna()
    merged.loc[needs_fallback, "detection_year"] = merged.loc[needs_fallback, "fallback_year"]
    merged.loc[needs_fallback, "detection_date"] = merged.loc[needs_fallback, "first_loss_date"]
    merged.loc[needs_fallback, "detection_source"] = "fallback_first_loss"
    merged["detection_year"] = merged["detection_year"].astype("Int64")
    return merged.loc[:, [GROUP_ID, "detection_date", "detection_year", "detection_source"]]


def classify_group_relevance(
    summary: pd.DataFrame,
    *,
    alive_year_shift: int = 0,
) -> pd.DataFrame:
    """Определяет класс релевантности группы относительно года обнаружения.

    Args:
        summary: сводка по группам (см. :func:`build_group_summary`).
        alive_year_shift: на сколько лет после обнаружения группа должна
            «доживать», чтобы считаться действующей (0 = достаточно года
            обнаружения).

    Returns:
        Копия сводки с колонкой ``group_relevance``.

    Классы:
        * ``Продолжает действовать`` — активность группы продолжается в год
          обнаружения и позже, то есть без вмешательства сервиса убытки шли бы
          и дальше;
        * ``Распалась после обнаружения`` — группа жила до обнаружения, но
          последний убыток приходится на год обнаружения (момент, когда сервис
          мог её «поймать»);
        * ``Распалась самостоятельно`` — группа перестала появляться **до** года
          обнаружения, эффект от сервиса по ней нулевой: убытки прекратились
          сами.

    Note:
        В исходной тетради класс присваивался построчно, и после этого строки
        «Распалась самостоятельно» с вероятностью 0.5 превращались в
        «Распалась после обнаружения» через ``np.random.choice`` — то есть часть
        эффекта создавалась генератором случайных чисел. Здесь случайности нет.
    """
    frame = summary.copy()
    detection = pd.to_numeric(frame["detection_year"], errors="coerce")
    last_year = pd.to_numeric(frame["last_loss_year"], errors="coerce")
    shift = int(alive_year_shift)

    # ``alive``: группа активна минимум через ``shift + 1`` год после обнаружения —
    # значит, без вмешательства сервиса убытки шли бы и дальше.
    # ``died_after``: последний убыток приходится ровно на год обнаружения
    # (с учётом требуемого запаса лет) — группа «закрыта» сервисом.
    # Остальное: ``died_before`` — убытки прекратились до обнаружения.
    alive_condition = last_year > (detection + shift)
    died_after_condition = last_year == (detection + shift)

    conditions = [alive_condition, died_after_condition]
    choices = [RELEVANCE_ALIVE, RELEVANCE_DIED_AFTER]
    frame["group_relevance"] = np.select(conditions, choices, default=RELEVANCE_DIED_BEFORE)

    unexpected = ~frame["group_relevance"].isin(RELEVANCE_LABELS)
    if unexpected.any():  # pragma: no cover - защита от новых значений
        LOGGER.warning("Неожиданные классы релевантности: %s", frame.loc[unexpected].head())
    return frame


def filter_groups_by_size(
    summary: pd.DataFrame,
    params: EffectParams = DEFAULT_PARAMS,
) -> tuple[pd.DataFrame, int]:
    """Отбрасывает «группы» из единичных убытков.

    Args:
        summary: сводка по группам.
        params: параметры методики.

    Returns:
        Кортеж «отфильтрованная сводка, сколько групп отброшено».

    Note:
        В тетради фильтр ``losses_count >= 3`` был закомментирован, а сама
        переменная называлась ``min_3_losses`` — то есть порог задумывался, но
        не применялся. Здесь он применяется (значение — ``EffectParams``).
    """
    if summary.empty:
        return summary, 0

    mask = summary["losses_total"] >= params.min_losses_total
    if "active_years_count" in summary.columns and params.min_years_for_baseline > 1:
        mask = mask & (summary["active_years_count"] >= params.min_years_for_baseline)
    dropped = int((~mask).sum())
    if dropped:
        LOGGER.info(
            "Отброшено групп фильтром размера (убытков >= %d, активных лет >= %d): %d",
            params.min_losses_total,
            params.min_years_for_baseline,
            dropped,
        )
    return summary.loc[mask].reset_index(drop=True), dropped


def year_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    """Разворачивает кортеж активных годов в бинарную матрицу «группа × год».

    Args:
        summary: сводка по группам.

    Returns:
        Таблица с колонками ``group_id``, ``group_relevance``, ``detection_year``
        и колонками-годами (1 = в этом году у группы были убытки).
    """
    if summary.empty:
        return pd.DataFrame(columns=["group_id", "group_relevance", "detection_year"])

    rows: list[dict[str, object]] = []
    for record in summary.itertuples(index=False):
        row: dict[str, object] = {
            "group_id": record.group_id,
            "group_relevance": record.group_relevance,
            "detection_year": record.detection_year,
        }
        for year in record.active_years:
            row[int(year)] = 1
        rows.append(row)

    frame = pd.DataFrame(rows).fillna(0)
    year_columns = sorted(column for column in frame.columns if isinstance(column, (int, np.integer)))
    return frame.loc[:, ["group_id", "group_relevance", "detection_year", *year_columns]]
