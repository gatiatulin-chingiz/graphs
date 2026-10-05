"""Предобработка: типы, нормализация, дедупликация, ключи сущностей.

Ключевые отличия от исходной тетради:

* **Дедупликация на уровне убытка.** В тетради таблица ``losses`` содержала
  несколько строк на один убыток (``LEFT JOIN`` к платежам и ПСР). Финансовые
  колонки после этого суммировались — то есть ``ValueRub`` (сумма по убытку)
  учитывался столько раз, сколько выплат по убытку есть в базе. Здесь денежные
  колонки сначала сводятся к одной строке на убыток, и только потом считаются
  метрики. Побочный эффект исходной логики: ``drop_duplicates('LossNumber')``
  в конце оставлял случайную строку, а не агрегат.
* **Явные типы.** Даты приводятся к ``datetime64``, деньги — к ``float``,
  идентификаторы — к нормализованным строкам.
* **Векторные операции.** Никаких ``apply(axis=1)`` по большим таблицам на этом
  шаге, кроме построения ключа группы.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    BENEFICIARY_KEY_COLUMNS,
    MONEY_COLUMNS,
    OBJECT_COLUMNS,
    PERSON_COLUMNS,
    PERSON_DATE_PAIRS,
)
from .textnorm import build_entity_keys, format_keys, normalize_series

LOGGER = logging.getLogger(__name__)

#: Внутренние имена колонок таблицы фактов.
LOSS_KEY = "LossNumber"
LOSS_DATE = "LossDateTime"
LOSS_YEAR = "loss_year"
GROUP_KEY = "group_key"
GROSS_LOSS = "gross_loss"
RECOVERIES = "recoveries"
NET_LOSS = "net_loss"

#: Разделитель для склейки «ФИО + ДР» в промежуточных колонках.
_NAME_DATE_SEPARATOR = "|"


@dataclass(frozen=True)
class LossPreprocessStats:
    """Статистика предобработки убытков."""

    rows_in: int = 0
    rows_out: int = 0
    duplicated_loss_numbers: int = 0
    rows_without_group: int = 0
    money_filled: int = 0

    def as_text(self) -> str:
        """Человекочитаемая сводка.

        Returns:
            Строка для логов ноутбука.
        """
        return (
            f"строк: {self.rows_in} -> {self.rows_out}; "
            f"дублей по {LOSS_KEY}: {self.duplicated_loss_numbers}; "
            f"без группы: {self.rows_without_group}"
        )


def normalize_losses(losses: pd.DataFrame) -> pd.DataFrame:
    """Приводит выгрузку убытков к каноничному виду.

    Args:
        losses: «сырая» выгрузка (из БД или файла).

    Returns:
        Копия таблицы, где:

        * ФИО и VIN нормализованы (регистр, пробелы, ``NaN``);
        * добавлены колонки ``_<ФИО-колонка>`` со склейкой «ФИО + ДР»;
        * ``LossDateTime`` — ``datetime64``;
        * денежные колонки — ``float``;
        * ``<VIN-колонка>`` нормализованы.

    Raises:
        KeyError: если в таблице нет ключевых колонок.
    """
    frame = losses.copy()

    if LOSS_KEY not in frame.columns:
        raise KeyError(f"В выгрузке убытков нет колонки {LOSS_KEY!r}")
    if LOSS_DATE not in frame.columns:
        raise KeyError(f"В выгрузке убытков нет колонки {LOSS_DATE!r}")

    for name_column, date_column in PERSON_DATE_PAIRS.items():
        if name_column not in frame.columns:
            LOGGER.debug("Колонка %s отсутствует в выгрузке — пропуск", name_column)
            continue
        date_series = frame[date_column] if date_column in frame.columns else pd.Series(
            "", index=frame.index
        )
        frame[f"_{name_column}"] = format_keys(frame[name_column], date_series)

    for column in OBJECT_COLUMNS:
        if column in frame.columns:
            frame[column] = normalize_series(frame[column])

    frame[LOSS_DATE] = pd.to_datetime(frame[LOSS_DATE], errors="coerce")
    frame[LOSS_KEY] = normalize_series(frame[LOSS_KEY])
    frame[LOSS_YEAR] = frame[LOSS_DATE].dt.year.astype("Int64")

    for column in MONEY_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            LOGGER.warning("Денежная колонка %s отсутствует — считаем нулями", column)
            frame[column] = 0.0

    return frame


def collapse_losses_to_facts(losses: pd.DataFrame) -> tuple[pd.DataFrame, LossPreprocessStats]:
    """Сводит убытки к одной строке на ``LossNumber`` и считает денежные метрики.

    Args:
        losses: нормализованная выгрузка (см. :func:`normalize_losses`).

    Returns:
        Кортеж «таблица фактов, статистика».

    Note:
        ``gross_loss`` = сумма выплат и претензий (то, что компания заплатила),
        ``recoveries`` = взыскано по ФУ и по иску, ``net_loss`` = разница.
        Именно ``net_loss`` — база для эффекта: взысканные деньги компания
        вернула, предотвращать их повторно не нужно.

        В исходной тетради считалось ``losses_money = ValueRub + претензии +
        взыскано_ФУ + взыскано_иск``, то есть возвраты **прибавлялись** к убытку
        и увеличивали «предотвращённый» ущерб. Это исправлено; сравнение
        результатов — в ноутбуке, блок «как было / как стало».
    """
    rows_in = len(losses)
    working = losses.copy()
    working["_gross"] = working["ValueRub"].fillna(0) + working["Сумма_выплат_по_претензиям"].fillna(0)
    working["_recoveries"] = working["Сумма_взыскано_по_ФУ"].fillna(0) + working[
        "Суммы_взыскано_по_иску"
    ].fillna(0)

    duplicated = int(working[LOSS_KEY].duplicated().sum()) if rows_in else 0

    group_columns = [column for column in BENEFICIARY_KEY_COLUMNS if column in working.columns]
    working[GROUP_KEY] = build_entity_keys(working, group_columns)

    if duplicated:
        LOGGER.warning(
            "В выгрузке %d строк-дублей по %s (размножение из-за JOIN платежей/ПСР); "
            "деньги будут агрегированы",
            duplicated,
            LOSS_KEY,
        )

    aggregations: dict[str, str] = {
        LOSS_DATE: "min",
        LOSS_YEAR: "min",
        "_gross": "sum",
        "_recoveries": "sum",
        GROUP_KEY: "first",
    }
    for column in group_columns:
        if column not in aggregations:
            aggregations[column] = "first"

    facts = (
        working.sort_values(LOSS_DATE)
        .groupby(LOSS_KEY, as_index=False, observed=True)
        .agg(aggregations)
    )
    facts = facts.rename(columns={"_gross": GROSS_LOSS, "_recoveries": RECOVERIES})
    facts[NET_LOSS] = facts[GROSS_LOSS] - facts[RECOVERIES]

    rows_without_group = int((facts[GROUP_KEY].map(len) == 0).sum())
    stats = LossPreprocessStats(
        rows_in=rows_in,
        rows_out=len(facts),
        duplicated_loss_numbers=duplicated,
        rows_without_group=rows_without_group,
        money_filled=int(facts[GROSS_LOSS].notna().sum()),
    )
    LOGGER.info("Предобработка убытков: %s", stats.as_text())
    return facts.reset_index(drop=True), stats


def _empty_pairs(facts: pd.DataFrame) -> pd.DataFrame:
    """Пустая таблица «убыток-группа» с полной схемой и сохранением типов.

    Args:
        facts: таблица фактов (используется как образец колонок и типов).

    Returns:
        Пустой ``DataFrame`` с колонками фактов, ``group_id`` и ``matched_members``.
    """
    empty = facts.iloc[0:0].copy()
    empty["group_id"] = pd.Series(dtype="object")
    empty["matched_members"] = pd.Series(dtype="int64")
    return empty


def build_beneficiary_lookup(beneficiaries: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только группы, чьи участники состоят в мошеннических группах сервиса.

    Args:
        beneficiaries: таблица участников (``group_id``, ``person_key``).

    Returns:
        Таблица с колонками ``person_key``, ``group_id``, ``members``: по одной
        строке на участника, ``members`` — размер группы.

    Note:
        Исходная тетрадь матчила убыток с участником через словарь
        ``band_list -> {2: group_number}``: при дублях ключа часть групп молча
        терялась, а размер группы не использовался. Здесь матч идёт по
        ``person_key``, а размер группы считается отдельно.
    """
    if beneficiaries.empty:
        return pd.DataFrame(columns=["person_key", "group_id", "members"])

    required = {"person_key", "group_id"}
    missing = required - set(beneficiaries.columns)
    if missing:
        raise KeyError(f"В таблице участников нет колонок: {sorted(missing)}")

    members = beneficiaries.groupby("group_id")["person_key"].transform("nunique")
    lookup = beneficiaries.loc[:, ["person_key", "group_id"]].copy()
    lookup["members"] = members
    return lookup.drop_duplicates().reset_index(drop=True)


def attach_groups_by_loss_number(
    facts: pd.DataFrame,
    beneficiaries: pd.DataFrame,
) -> pd.DataFrame:
    """Сопоставляет убытки с группами по номеру убытка из Excel.

    Args:
        facts: таблица фактов (колонка ``LossNumber`` уже нормализована).
        beneficiaries: участники с колонкой ``loss_number`` (prod-парсер).

    Returns:
        Таблица ``LossNumber``, ``group_id``, ``matched_members`` (1 на пару).
    """
    if beneficiaries.empty or "loss_number" not in beneficiaries.columns:
        return pd.DataFrame(columns=[LOSS_KEY, "group_id", "matched_members"])

    mapping = beneficiaries.loc[:, ["loss_number", "group_id"]].copy()
    mapping["loss_number"] = normalize_series(mapping["loss_number"])
    mapping = mapping.loc[mapping["loss_number"].astype(bool)]
    mapping = mapping.drop_duplicates(subset=["loss_number", "group_id"])
    if mapping.empty:
        return pd.DataFrame(columns=[LOSS_KEY, "group_id", "matched_members"])

    mapping = mapping.rename(columns={"loss_number": LOSS_KEY})
    membership = facts.loc[:, [LOSS_KEY]].drop_duplicates().merge(
        mapping, on=LOSS_KEY, how="inner"
    )
    membership["matched_members"] = 1
    return membership.loc[:, [LOSS_KEY, "group_id", "matched_members"]]


def _beneficiaries_have_loss_numbers(beneficiaries: pd.DataFrame) -> bool:
    """Есть ли в выгрузке сервиса номера убытков (prod Excel с колонкой «Убыток»)."""
    if beneficiaries.empty or "loss_number" not in beneficiaries.columns:
        return False
    series = beneficiaries["loss_number"]
    if series.empty:
        return False
    text = normalize_series(series.astype("object"))
    return bool(text.astype(bool).any())


def attach_groups_to_losses(
    facts: pd.DataFrame,
    beneficiaries: pd.DataFrame,
    *,
    min_members_matched: int = 1,
    match_mode: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Сопоставляет убытки с группами сервиса по участникам и/или номеру убытка.

    Args:
        facts: таблица фактов по убыткам (см. :func:`collapse_losses_to_facts`).
        beneficiaries: таблица участников.
        min_members_matched: сколько участников группы должно встречаться в
            убытке, чтобы убыток считался «групповым» (для матча по person_key).
        match_mode: ``auto`` — только ``loss_number``, если он есть в Excel
            (prod); иначе ``person_key``. ``union`` — объединять оба способа
            (legacy + доп. номера). ``loss_number`` / ``person_key`` — явно.

    Returns:
        Кортеж:

        * ``loss_group`` — таблица соответствий «убыток -> группа» (по одной
          строке на пару, поэтому убыток может попасть в несколько групп);
        * ``facts_with_groups`` — исходные факты, дополненные колонками
          ``group_id``, ``matched_members`` (одна строка на пару убыток-группа).

    Note:
        Prod-выгрузки содержат ``loss_number`` — прямой матч с ``LossNumber``.
        Join по ``person_key`` подтягивает **все** убытки участника из КАСКО и
        раздувает базу; в ``auto`` при наличии ``loss_number`` он не используется.
    """
    if match_mode not in {"auto", "loss_number", "person_key", "union"}:
        raise ValueError("match_mode: auto | loss_number | person_key | union")

    if beneficiaries.empty or facts.empty:
        return _empty_pairs(facts), _empty_pairs(facts)

    membership_parts: list[pd.DataFrame] = []
    use_loss = match_mode in {"auto", "loss_number", "union"}
    use_person = match_mode in {"person_key", "union"}
    if match_mode == "auto":
        has_loss_numbers = _beneficiaries_have_loss_numbers(beneficiaries)
        use_loss = has_loss_numbers
        use_person = not has_loss_numbers

    by_loss = pd.DataFrame(columns=[LOSS_KEY, "group_id", "matched_members"])
    if use_loss:
        by_loss = attach_groups_by_loss_number(facts, beneficiaries)
        if not by_loss.empty:
            membership_parts.append(by_loss)

    keys = [column for column in BENEFICIARY_KEY_COLUMNS if column in facts.columns]
    has_person = (
        use_person
        and "person_key" in beneficiaries.columns
        and beneficiaries["person_key"].astype(bool).any()
    )
    if keys and has_person:
        lookup = build_beneficiary_lookup(
            beneficiaries.loc[beneficiaries["person_key"].astype(bool)]
        )
        if not lookup.empty:
            id_vars = [LOSS_KEY, GROSS_LOSS, NET_LOSS, LOSS_YEAR]
            available_id = [column for column in id_vars if column in facts.columns]
            long_frame = (
                facts.loc[:, available_id + keys]
                .melt(id_vars=available_id, value_vars=keys, value_name="person_key")
                .drop(columns=["variable"])
            )
            long_frame = long_frame.loc[long_frame["person_key"].astype(bool)]
            long_frame = long_frame.drop_duplicates(subset=[LOSS_KEY, "person_key"])
            if not long_frame.empty:
                merged = long_frame.merge(lookup, on="person_key", how="inner")
                by_person = (
                    merged.groupby([LOSS_KEY, "group_id"], as_index=False)
                    .agg(
                        matched_members=("person_key", "nunique"),
                        members=("members", "max"),
                    )
                    .loc[lambda frame: frame["matched_members"] >= min_members_matched]
                    .drop(columns=["members"])
                )
                if not by_person.empty:
                    membership_parts.append(by_person)

    if not membership_parts:
        return _empty_pairs(facts), _empty_pairs(facts)

    membership = pd.concat(membership_parts, ignore_index=True)
    membership = (
        membership.groupby([LOSS_KEY, "group_id"], as_index=False)
        .agg(matched_members=("matched_members", "max"))
    )

    facts_with_groups = facts.merge(membership, on=LOSS_KEY, how="inner")
    by_person_pairs = 0
    if use_person and not by_loss.empty:
        loss_keys = set(by_loss[LOSS_KEY])
        by_person_pairs = int((~membership[LOSS_KEY].isin(loss_keys)).sum())
    elif use_person:
        by_person_pairs = len(membership)
    LOGGER.info(
        "Сопоставление (%s): убытков %d из %d; пар %d (по номеру: %d, по ФИО: %d)",
        match_mode,
        facts_with_groups[LOSS_KEY].nunique(),
        facts[LOSS_KEY].nunique(),
        len(facts_with_groups),
        len(by_loss) if not by_loss.empty else 0,
        by_person_pairs,
    )
    return membership, facts_with_groups


def add_group_number_column(
    facts_with_groups: pd.DataFrame,
    beneficiaries: pd.DataFrame,
    group_column: str = "group_number",
) -> pd.DataFrame:
    """Добавляет к парам «убыток-группа» номер группы из выгрузки сервиса.

    Args:
        facts_with_groups: результат :func:`attach_groups_to_losses`.
        beneficiaries: таблица участников.
        group_column: имя колонки с номером группы в выгрузке сервиса.

    Returns:
        Таблица с добавленной колонкой ``group_number`` (если она есть в выгрузке).
    """
    if group_column not in beneficiaries.columns or facts_with_groups.empty:
        return facts_with_groups

    mapping = (
        beneficiaries.loc[:, ["group_id", group_column]]
        .drop_duplicates()
        .groupby("group_id")[group_column]
        .first()
    )
    frame = facts_with_groups.copy()
    frame[group_column] = frame["group_id"].map(mapping)
    return frame


def money_columns_present(frame: pd.DataFrame, columns: Iterable[str] = MONEY_COLUMNS) -> list[str]:
    """Возвращает присутствующие денежные колонки.

    Args:
        frame: таблица.
        columns: проверяемые колонки.

    Returns:
        Список колонок, которые есть в таблице.
    """
    return [column for column in columns if column in frame.columns]


def aggregate_money_long(
    frame: pd.DataFrame,
    keys: Sequence[str],
    money_columns: Sequence[str] = MONEY_COLUMNS,
) -> pd.DataFrame:
    """Суммирует денежные колонки «в длинном формате» (утилита для отчётов).

    Args:
        frame: таблица.
        keys: колонки группировки.
        money_columns: денежные колонки.

    Returns:
        Сводная таблица «ключи + сумма по всем денежным колонкам».
    """
    present = money_columns_present(frame, money_columns)
    if not present:
        return frame.loc[:, list(keys)].drop_duplicates().assign(total=0.0)

    working = frame.copy()
    working["total"] = working.loc[:, present].fillna(0).sum(axis=1)
    return working.groupby(list(keys), as_index=False)["total"].sum()


def safe_divide(numerator: float, denominator: float, default: float = np.nan) -> float:
    """Деление с защитой от нуля.

    Args:
        numerator: числитель.
        denominator: знаменатель.
        default: значение при нулевом знаменателе.

    Returns:
        Результат деления или ``default``.
    """
    if denominator in (0, None) or pd.isna(denominator):
        return default
    return numerator / denominator
