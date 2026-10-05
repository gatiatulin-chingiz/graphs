"""Воспроизведение исходной (до рефакторинга) логики тетради.

Модуль нужен ровно для одного: показать в ноутбуке блок «как было / как стало»
и посчитать, насколько каждая правка меняет итоговую цифру. Здесь намеренно
сохранены все дефекты оригинала:

* ``losses_money = ValueRub + претензии + взыскано_ФУ + взыскано_иск`` — возвраты
  **прибавляются** к убытку;
* денежные колонки суммируются на уровне строк выгрузки, без свёртки до
  ``LossNumber``, поэтому ``ValueRub`` дублируется по числу платежей;
* ``detection_date`` жёстко равна :data:`LEGACY_DETECTION_YEAR`;
* группы с датой обнаружения «в будущем» относительно последнего убытка с
  вероятностью 0.5 переписываются в «Распалась после обнаружения» через
  ``np.random.choice`` — то есть часть эффекта создаётся генератором случайных
  чисел;
* ``mean_loss_money_by_year`` в pivot агрегируется через ``mean``, а не ``sum``;
* связь убытка с группой — через ``Series.map`` по строковому ключу, то есть
  максимум одна группа на убыток.

Ничего из этого **не** используется в рабочем расчёте.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .preprocess import GROUP_KEY, LOSS_DATE, LOSS_KEY, LOSS_YEAR
from .textnorm import build_entity_keys, format_keys, normalize_series

LOGGER = logging.getLogger(__name__)

#: Жёстко заданный год обнаружения — как в исходной тетради.
LEGACY_DETECTION_YEAR = 2024

#: Вероятность переписать «Распалась самостоятельно» на «Распалась после обнаружения».
LEGACY_RELEVANCE_FLIP_PROBABILITY = 0.5

#: Коэффициент сервиса из исходной тетради.
LEGACY_SERVICE_SHARE = 0.35

#: Классы релевантности в формулировках оригинала.
LEGACY_ALIVE = "Продолжает действовать"
LEGACY_DIED_AFTER = "Распалась после обнаружения"
LEGACY_DIED_BEFORE = "Распалась самостоятельно"


def legacy_beneficiaries_pipeline(raw_beneficiaries: pd.DataFrame) -> pd.DataFrame:
    """Обрабатывает выгрузку сервиса «как в тетради».

    Args:
        raw_beneficiaries: таблица с колонками ``0`` (ФИО), ``1`` (дата рождения),
            ``2`` (номер группы).

    Returns:
        Таблица с колонками ``0``, ``1``, ``2`` и ``band_list``.

    Note:
        Повторяет шаги оригинала: очистка текста -> фильтр шапок по точному
        совпадению ``'фио'``/``'др'``/``'group_number'`` -> отбрасывание пустых
        строк -> ``band_list = ФИО + ' ' + дата``.
    """
    frame = raw_beneficiaries.copy()
    for column in (0, 1, 2):
        frame[column] = normalize_series(frame[column])

    # Шапки отсекаются до разбора дат: иначе pandas сыплет предупреждениями
    # о смешанных форматах («др» — не дата).
    header_mask = (frame[0] == "фио") | (frame[1] == "др") | (frame[2] == "group_number")
    frame = frame.loc[~header_mask]

    frame[1] = pd.to_datetime(frame[1], errors="coerce").dt.normalize()
    frame = frame.loc[(frame[0] != "") & (frame[2] != "")]
    frame["band_list"] = frame[0] + " " + frame[1].dt.strftime("%Y-%m-%d").fillna("")
    frame["band_list"] = frame["band_list"].str.strip()
    return frame.reset_index(drop=True)


def _person_key_for_losses(losses: pd.DataFrame, name_column: str) -> pd.Series:
    """Строит строковый ключ «ФИО ДР» так, как это делал оригинал.

    Args:
        losses: выгрузка убытков.
        name_column: колонка ФИО.

    Returns:
        Серия ключей (нормализованные ФИО и дата, склеенные пробелом).

    Note:
        В тетради ключ собирался как ``column + ' ' + losses[date_column].astype(str)``
        и не нормализовался, поэтому половина убытков не матчилась с группами.
        Здесь ключ приведён к тому же виду, что и ``band_list``, иначе
        демонстрация «как было» превращается в демонстрацию пустого DataFrame.
    """
    from .config import PERSON_DATE_PAIRS

    date_column = PERSON_DATE_PAIRS.get(name_column)
    if date_column is None or date_column not in losses.columns:
        return normalize_series(losses[name_column])

    with_date = format_keys(losses[name_column], losses[date_column])
    return with_date


def legacy_join_groups(
    losses: pd.DataFrame,
    legacy_beneficiaries: pd.DataFrame,
) -> pd.DataFrame:
    """Привязывает группы к убыткам через ``Series.map`` по ключу участника.

    Args:
        losses: выгрузка убытков (строки могут дублироваться по платежам).
        legacy_beneficiaries: результат :func:`legacy_beneficiaries_pipeline`.

    Returns:
        Копия ``losses`` с колонками ``groups_list`` (ключ группы),
        ``losses_money`` (сумма по формуле оригинала) и ``group_number``.

    Note:
        Воспроизводится ключевое ограничение оригинала: используется ``map``,
        поэтому убыток привязывается **максимум к одной** группе, и при дублях
        ключа в словаре часть групп теряется.
    """
    frame = losses.copy()
    person_keys: dict[str, pd.Series] = {}
    for name_column in _person_columns():
        if name_column not in frame.columns:
            continue
        key = _person_key_for_losses(frame, name_column)
        frame[f"_{name_column}"] = key
        person_keys[name_column] = key

    frame[GROUP_KEY] = build_entity_keys(frame, [f"_{name}" for name in person_keys])
    frame["groups_list"] = frame[GROUP_KEY].map(lambda key: ", ".join(key) if key else "")

    mapping = _group_number_mapping(legacy_beneficiaries)
    membership = pd.Series(pd.NA, index=frame.index, dtype="object")
    for keys in person_keys.values():
        membership = membership.where(membership.notna(), keys.map(mapping))

    frame["group_number"] = membership
    frame = frame.loc[frame["groups_list"] != ""]
    frame = frame.drop_duplicates(LOSS_KEY)
    frame["losses_money"] = legacy_losses_money(frame)
    return frame.reset_index(drop=True)


def _person_columns() -> tuple[str, ...]:
    """Колонки ФИО из схемы выгрузки.

    Returns:
        Кортеж имён колонок.
    """
    from .config import PERSON_COLUMNS

    return PERSON_COLUMNS


def _group_number_mapping(legacy_beneficiaries: pd.DataFrame) -> dict[str, str]:
    """Словарь «ключ участника -> номер группы» (при дублях — последний).

    Args:
        legacy_beneficiaries: обработанная выгрузка сервиса.

    Returns:
        Словарь отображения.
    """
    return dict(
        zip(
            legacy_beneficiaries["band_list"],
            legacy_beneficiaries[2],
            strict=False,
        )
    )


def legacy_losses_money(losses: pd.DataFrame) -> pd.Series:
    """Считает ``losses_money`` по формуле оригинала.

    Args:
        losses: выгрузка убытков с денежными колонками.

    Returns:
        Серия сумм (возвраты прибавляются).
    """
    total = pd.Series(0.0, index=losses.index)
    for column in (
        "ValueRub",
        "Сумма_выплат_по_претензиям",
        "Сумма_взыскано_по_ФУ",
        "Суммы_взыскано_по_иску",
    ):
        if column in losses.columns:
            total = total + pd.to_numeric(losses[column], errors="coerce").fillna(0)
    return total


def legacy_detection_and_relevance(
    losses_: pd.DataFrame,
    cards: pd.DataFrame | None = None,
    *,
    seed: int = 42,
    detection_year: int = LEGACY_DETECTION_YEAR,
) -> pd.DataFrame:
    """Повторяет блок «Идентификаторы группы + деньги» из тетради.

    Args:
        losses_: таблица убытков с колонками ``groups_list``, ``losses_money``.
        cards: карточки УПС. В оригинале дата обнаружения бралась отсюда, но затем
            перезаписывалась константой; параметр нужен только для честного
            сравнения сценариев.
        seed: зерно генератора для переприсвоения класса релевантности.
        detection_year: жёстко заданный год обнаружения.

    Returns:
        Таблица с колонками ``detection_date``, ``first_lossnumber_by``,
        ``last_lossnumber_by``, ``group_relevance``, ``mean_loss_money_by_year``.

    Note:
        ``detection_year`` по умолчанию равен :data:`LEGACY_DETECTION_YEAR`
        (2024) — ровно как в тетради, независимо от данных карточек.
    """
    frame = losses_.copy()
    frame[LOSS_KEY] = normalize_series(frame[LOSS_KEY])
    frame[LOSS_DATE] = pd.to_datetime(frame[LOSS_DATE], errors="coerce")
    frame[LOSS_YEAR] = frame[LOSS_DATE].dt.year

    # Как в тетради: detection_date сначала из карточек...
    if cards is not None and not cards.empty and "Номер Убытка" in cards.columns:
        card_dates = (
            cards.assign(**{LOSS_KEY: normalize_series(cards["Номер Убытка"])})
            .assign(card_date=lambda data: pd.to_datetime(data["Дата создания карточки"], errors="coerce"))
            .groupby(LOSS_KEY, as_index=False)["card_date"]
            .min()
        )
        frame = frame.merge(card_dates, on=LOSS_KEY, how="left")
        frame["detection_date"] = frame["card_date"]
    else:
        frame["detection_date"] = pd.NaT

    # ...а затем молча перезаписывается константой.
    frame["detection_date"] = detection_year

    bounds = (
        frame.groupby("groups_list")[LOSS_YEAR]
        .agg(first_lossnumber_by="min", last_lossnumber_by="max")
        .reset_index()
    )
    frame = frame.merge(bounds, on="groups_list", how="left")

    frame.loc[
        frame["detection_date"] == frame["last_lossnumber_by"], "group_relevance"
    ] = LEGACY_DIED_AFTER
    frame.loc[
        frame["detection_date"] < frame["last_lossnumber_by"], "group_relevance"
    ] = LEGACY_ALIVE
    frame.loc[
        frame["detection_date"] > frame["last_lossnumber_by"], "group_relevance"
    ] = LEGACY_DIED_BEFORE

    rng = np.random.default_rng(seed)
    died_alone = frame["group_relevance"] == LEGACY_DIED_BEFORE
    count = int(died_alone.sum())
    if count:
        frame.loc[died_alone, "group_relevance"] = rng.choice(
            [LEGACY_DIED_BEFORE, LEGACY_DIED_AFTER],
            size=count,
            p=[1 - LEGACY_RELEVANCE_FLIP_PROBABILITY, LEGACY_RELEVANCE_FLIP_PROBABILITY],
        )

    year_counts = (
        frame.groupby("groups_list")[LOSS_YEAR].nunique().reset_index().rename(
            columns={LOSS_YEAR: "year_count"}
        )
    )
    totals = (
        frame.groupby("groups_list", as_index=False)["losses_money"]
        .sum()
        .rename(columns={"losses_money": "groups_total"})
        .merge(year_counts, on="groups_list", how="left")
    )
    totals["mean_loss_money_by_year"] = totals["groups_total"] / totals["year_count"].replace(0, np.nan)
    return frame.merge(
        totals.loc[:, ["groups_list", "mean_loss_money_by_year"]], on="groups_list", how="left"
    )


def legacy_effect_totals(losses_: pd.DataFrame) -> dict[str, float]:
    """Считает эффект по формулам конца тетради.

    Args:
        losses_: результат :func:`legacy_detection_and_relevance`.

    Returns:
        Словарь с ключами ``pivot_total`` (сумма среднего годового убытка по pivot)
        и ``effect`` (эффект с коэффициентом 0.35).
    """
    if "group_relevance" not in losses_.columns:
        return {"pivot_total": 0.0, "effect": 0.0}

    subset = losses_.loc[losses_["group_relevance"] == LEGACY_DIED_AFTER]
    if subset.empty:
        return {"pivot_total": 0.0, "effect": 0.0}

    total = pd.pivot_table(
        subset,
        values="mean_loss_money_by_year",
        index=["groups_list"],
        columns=[LOSS_YEAR],
        aggfunc="mean",
    ).fillna(0)
    total = total.replace([np.inf, -np.inf], 0)
    pivot_total = float(total.to_numpy().sum())
    return {"pivot_total": pivot_total, "effect": pivot_total * LEGACY_SERVICE_SHARE}


def legacy_naive_double_count(raw_losses: pd.DataFrame) -> dict[str, float]:
    """Показывает эффект «двойного учёта» денежных колонок.

    Args:
        raw_losses: выгрузка убытков «как пришла» (с дублями строк).

    Returns:
        Словарь: сумма ``losses_money`` по строкам и сумма после свёртки до
        ``LossNumber``, а также их разница.
    """
    row_level = float(legacy_losses_money(raw_losses).sum())
    collapsed = (
        raw_losses.assign(_money=legacy_losses_money(raw_losses))
        .groupby(LOSS_KEY, as_index=False)["_money"]
        .sum()
    )
    collapsed_total = float(collapsed["_money"].sum())
    return {
        "row_level_total": row_level,
        "collapsed_total": collapsed_total,
        "inflation": row_level - collapsed_total,
    }


def legacy_vs_corrected_money(raw_losses: pd.DataFrame) -> pd.DataFrame:
    """Сравнивает «как было» и «как стало» по денежной базе.

    Args:
        raw_losses: выгрузка убытков со дублями строк.

    Returns:
        Таблица со строками «исходная формула», «свёртка до убытка» и
        «корректно: выплаты минус возвраты».
    """
    from .preprocess import collapse_losses_to_facts, normalize_losses

    normalized = normalize_losses(raw_losses)
    facts, _ = collapse_losses_to_facts(normalized)

    legacy_row_level = float(legacy_losses_money(raw_losses).sum())
    legacy_collapsed = float(
        raw_losses.assign(_money=legacy_losses_money(raw_losses))
        .groupby(LOSS_KEY, as_index=False)["_money"]
        .sum()["_money"]
        .sum()
    )
    corrected = float(facts["net_loss"].sum())
    gross = float(facts["gross_loss"].sum())
    recoveries = float(facts["recoveries"].sum())

    table = pd.DataFrame(
        [
            {
                "Вариант": "Исходный: строки выгрузки, возвраты прибавляются",
                "Сумма, ₽": legacy_row_level,
                "Выплаты, ₽": np.nan,
                "Возвраты, ₽": np.nan,
            },
            {
                "Вариант": "Свёртка до убытка, возвраты прибавляются",
                "Сумма, ₽": legacy_collapsed,
                "Выплаты, ₽": np.nan,
                "Возвраты, ₽": np.nan,
            },
            {
                "Вариант": "Корректно: выплаты минус возвраты, до убытка",
                "Сумма, ₽": corrected,
                "Выплаты, ₽": gross,
                "Возвраты, ₽": recoveries,
            },
        ]
    )
    table["Множитель к корректной"] = (
        table["Сумма, ₽"] / corrected if corrected else np.nan
    )
    return table


def legacy_pipeline_summary(
    raw_losses: pd.DataFrame,
    raw_beneficiaries: pd.DataFrame,
    cards: pd.DataFrame | None = None,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Прогоняет исходную логику целиком и возвращает её «эффект».

    Args:
        raw_losses: выгрузка убытков с дублями строк.
        raw_beneficiaries: выгрузка сервиса (колонки 0, 1, 2).
        cards: карточки УПС (в оригинале перезаписываются константой).
        seed: зерно переприсвоения класса релевантности.

    Returns:
        Таблица «Вариант расчёта -> эффект, ₽» для сравнения с корректным.
    """
    processed = legacy_beneficiaries_pipeline(raw_beneficiaries)
    joined = legacy_join_groups(raw_losses, processed)
    final = legacy_detection_and_relevance(joined, cards, seed=seed)
    totals = legacy_effect_totals(final)
    classes = final["group_relevance"].value_counts().to_dict()

    rows = [
        {"Вариант расчёта": "Исходная логика тетради", "Эффект, ₽": totals["effect"]},
        {"Вариант расчёта": "База эффекта (до доли сервиса)", "Эффект, ₽": totals["pivot_total"]},
    ]
    for label, count in classes.items():
        rows.append({"Вариант расчёта": f"  групп: {label}", "Эффект, ₽": float(count)})
    return pd.DataFrame(rows)
