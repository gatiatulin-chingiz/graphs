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
    BENEFICIARY_KEY_COLUMNS,
    DEFAULT_PARAMS,
    RELEVANCE_CLOSED,
    RELEVANCE_IN_PROGRESS,
    RELEVANCE_LABELS,
    EffectParams,
)
from .preprocess import LOSS_KEY, LOSS_YEAR, NET_LOSS
from .textnorm import normalize_series, normalize_text

LOGGER = logging.getLogger(__name__)

#: Имя колонки с идентификатором группы.
GROUP_ID = "group_id"

#: Колонки таблицы карточек УПС -> внутренние имена.
CARD_DATE_COLUMN = "Дата создания карточки"
CARD_STAGE_COLUMN = "Стадия"
CARD_LOSS_COLUMN = "Номер Убытка"

#: Колонки сводки по группе.
GROUP_SUMMARY_COLUMNS = (
    "group_id",
    "detection_year",
    "detection_date",
    "detection_source",
    "card_stage",
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
        Таблица ``LossNumber``, ``card_date`` [, ``card_stage``] — по одной строке на убыток.

    Raises:
        KeyError: если в таблице карточек нет обязательных колонок.
    """
    required = {CARD_LOSS_COLUMN, CARD_DATE_COLUMN}
    missing = required - set(cards.columns)
    if missing:
        raise KeyError(f"В таблице карточек нет колонок: {sorted(missing)}")

    columns = [CARD_LOSS_COLUMN, CARD_DATE_COLUMN]
    rename = {CARD_LOSS_COLUMN: LOSS_KEY, CARD_DATE_COLUMN: "card_date"}
    if CARD_STAGE_COLUMN in cards.columns:
        columns.append(CARD_STAGE_COLUMN)
        rename[CARD_STAGE_COLUMN] = "card_stage_raw"

    working = cards.loc[:, columns].copy()
    working = working.rename(columns=rename)
    working["card_date"] = pd.to_datetime(working["card_date"], errors="coerce")
    working[LOSS_KEY] = normalize_series(working[LOSS_KEY])
    if "card_stage_raw" in working.columns:
        working["card_stage"] = working["card_stage_raw"].map(normalize_card_stage)
    else:
        working["card_stage"] = pd.NA

    working = working.dropna(subset=["card_date"])
    working = working.loc[working[LOSS_KEY].astype(bool)]
    if working.empty:
        return pd.DataFrame(columns=[LOSS_KEY, "card_date", "card_stage"])

    # На убыток — самая ранняя карточка; стадия — с приоритетом «в работе».
    working = working.sort_values("card_date")
    first = working.groupby(LOSS_KEY, as_index=False).agg(
        card_date=("card_date", "min"),
        card_stage=("card_stage", _prefer_stage),
    )
    LOGGER.info("Карточки УПС: %d убытков с датой создания", len(first))
    return first


def normalize_card_stage(value: object) -> str | None:
    """Приводит стадию карточки УПС к одному из двух статусов группы.

    Args:
        value: сырое значение колонки «Стадия».

    Returns:
        :data:`~fin_effects.config.RELEVANCE_IN_PROGRESS`,
        :data:`~fin_effects.config.RELEVANCE_CLOSED` или ``None``.
    """
    text = normalize_text(value)
    if not text:
        return None
    if "закрыт" in text:
        return RELEVANCE_CLOSED
    if "работ" in text or "открыт" in text:
        return RELEVANCE_IN_PROGRESS
    return None


def _prefer_stage(values: pd.Series) -> object:
    """При нескольких карточках предпочитает «в работе» закрытой."""
    stages = {normalize_card_stage(value) for value in values}
    stages.discard(None)
    if RELEVANCE_IN_PROGRESS in stages:
        return RELEVANCE_IN_PROGRESS
    if RELEVANCE_CLOSED in stages:
        return RELEVANCE_CLOSED
    return pd.NA


def link_cards_to_groups(
    cards: pd.DataFrame,
    *,
    facts_with_groups: pd.DataFrame | None = None,
    loss_facts: pd.DataFrame | None = None,
    beneficiaries: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Связывает карточки УПС с группами сервиса.

    Карточка часто ссылается на **один** убыток группы. Группа помечается
    как «в работе у УПС», если:

    * номер убытка из карточки уже есть в парах «убыток–группа»; или
    * из этого убытка извлекаются ФИО+ДР и матчятся с участниками Excel.

    Args:
        cards: выгрузка карточек УПС.
        facts_with_groups: пары убыток–группа (Excel).
        loss_facts: факты по убыткам с колонками ``_<ФИО>``.
        beneficiaries: участники групп с ``person_key``.

    Returns:
        Таблица ``group_id``, ``detection_date``, ``card_stage``, ``detection_source``.
    """
    if cards is None or cards.empty:
        return pd.DataFrame(
            columns=[GROUP_ID, "detection_date", "card_stage", "detection_source"]
        )

    card_dates = loss_detection_dates(cards)
    if card_dates.empty:
        return pd.DataFrame(
            columns=[GROUP_ID, "detection_date", "card_stage", "detection_source"]
        )

    parts: list[pd.DataFrame] = []

    if facts_with_groups is not None and not facts_with_groups.empty:
        by_loss = (
            facts_with_groups.loc[:, [LOSS_KEY, GROUP_ID]]
            .drop_duplicates()
            .merge(card_dates, on=LOSS_KEY, how="inner")
        )
        if not by_loss.empty:
            parts.append(
                by_loss.loc[:, [GROUP_ID, "card_date", "card_stage"]].assign(
                    detection_source="ups_card_loss"
                )
            )

    if (
        loss_facts is not None
        and not loss_facts.empty
        and beneficiaries is not None
        and not beneficiaries.empty
        and "person_key" in beneficiaries.columns
    ):
        key_cols = [column for column in BENEFICIARY_KEY_COLUMNS if column in loss_facts.columns]
        if key_cols:
            card_losses = card_dates.loc[:, [LOSS_KEY]].drop_duplicates()
            people = (
                loss_facts.loc[:, [LOSS_KEY, *key_cols]]
                .merge(card_losses, on=LOSS_KEY, how="inner")
                .melt(id_vars=[LOSS_KEY], value_vars=key_cols, value_name="person_key")
                .drop(columns=["variable"])
            )
            people["person_key"] = normalize_series(people["person_key"])
            people = people.loc[people["person_key"].astype(bool)].drop_duplicates()
            lookup = beneficiaries.loc[:, ["person_key", GROUP_ID]].copy()
            lookup["person_key"] = normalize_series(lookup["person_key"])
            lookup = lookup.loc[lookup["person_key"].astype(bool)].drop_duplicates()
            if not people.empty and not lookup.empty:
                by_person = (
                    people.merge(lookup, on="person_key", how="inner")
                    .loc[:, [LOSS_KEY, GROUP_ID]]
                    .drop_duplicates()
                    .merge(card_dates, on=LOSS_KEY, how="inner")
                )
                if not by_person.empty:
                    parts.append(
                        by_person.loc[:, [GROUP_ID, "card_date", "card_stage"]].assign(
                            detection_source="ups_card_person"
                        )
                    )

    if not parts:
        return pd.DataFrame(
            columns=[GROUP_ID, "detection_date", "card_stage", "detection_source"]
        )

    linked = pd.concat(parts, ignore_index=True)
    linked = linked.rename(columns={"card_date": "detection_date"})
    aggregated = (
        linked.sort_values("detection_date")
        .groupby(GROUP_ID, as_index=False)
        .agg(
            detection_date=("detection_date", "min"),
            card_stage=("card_stage", _prefer_stage),
            detection_source=("detection_source", "first"),
        )
    )
    LOGGER.info(
        "Карточки УПС привязаны к %d группам (через убыток/ФИО)",
        len(aggregated),
    )
    return aggregated


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
    *,
    loss_facts: pd.DataFrame | None = None,
    beneficiaries: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, GroupBuildStats]:
    """Строит сводку по группам: обнаружение по УПС, жизнь, активные годы.

    Args:
        facts_with_groups: пары «убыток-группа».
        cards: таблица карточек УПС. Без карточки группа в эффект не попадает.
        params: параметры методики.
        loss_facts: факты убытков (для расширения карточки через ФИО+ДР).
        beneficiaries: участники Excel (для матча ФИО+ДР).

    Returns:
        Кортеж «сводка только по группам с карточкой УПС, статистика».
    """
    if facts_with_groups.empty:
        return pd.DataFrame(columns=GROUP_SUMMARY_COLUMNS), GroupBuildStats()

    working = facts_with_groups.copy()
    base = working.loc[working["group_id"].notna()]
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

    all_groups = int(summary[GROUP_ID].nunique())
    detection = link_cards_to_groups(
        cards if cards is not None else pd.DataFrame(),
        facts_with_groups=facts_with_groups,
        loss_facts=loss_facts,
        beneficiaries=beneficiaries,
    )
    summary = summary.merge(detection, on=GROUP_ID, how="left")
    summary["detection_year"] = pd.to_datetime(
        summary["detection_date"], errors="coerce"
    ).dt.year.astype("Int64")

    has_card = summary["detection_date"].notna()
    groups_without_card = int((~has_card).sum())
    if groups_without_card:
        LOGGER.warning(
            "У %d из %d групп нет карточки УПС — в расчёт эффекта не входят",
            groups_without_card,
            all_groups,
        )
    summary = summary.loc[has_card].copy()

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


def classify_group_relevance(summary: pd.DataFrame) -> pd.DataFrame:
    """Статус группы по стадии карточки УПС: «в работе» или «закрыта».

    Args:
        summary: сводка с колонкой ``card_stage`` (см. :func:`build_group_summary`).

    Returns:
        Копия сводки с колонкой ``group_relevance``. Группы без стадии отбрасываются.
    """
    frame = summary.copy()
    if frame.empty:
        frame["group_relevance"] = pd.Series(dtype="object")
        return frame

    if "card_stage" in frame.columns:
        frame["group_relevance"] = frame["card_stage"].map(
            lambda value: value if value in RELEVANCE_LABELS else normalize_card_stage(value)
        )
    else:
        frame["group_relevance"] = pd.NA

    unknown = frame["group_relevance"].isna() | ~frame["group_relevance"].isin(RELEVANCE_LABELS)
    if unknown.any():
        LOGGER.warning(
            "У %d групп нет стадии УПС («в работе»/«закрыта») — отброшены",
            int(unknown.sum()),
        )
        frame = frame.loc[~unknown].copy()

    return frame.reset_index(drop=True)


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
