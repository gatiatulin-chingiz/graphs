"""Дымовой прогон пайплайна на синтетических данных.

Запуск::

    python scripts/run_synthetic.py [--groups N] [--seed S] [--legacy]

Скрипт генерирует синтетику, прогоняет пайплайн, печатает ключевые таблицы и
(опционально) сравнивает с исходной логикой из ``fin_effects.legacy``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from fin_effects.effect import effect_matrix  # noqa: E402
from fin_effects.legacy import (  # noqa: E402
    legacy_beneficiaries_pipeline,
    legacy_detection_and_relevance,
    legacy_effect_totals,
    legacy_join_groups,
    legacy_vs_corrected_money,
)
from fin_effects.pipeline import run_pipeline  # noqa: E402
from fin_effects.reporting import (  # noqa: E402
    effect_by_cohort_table,
    effect_by_year_table,
    headcount_table,
    top_groups_table,
)
from fin_effects.synth import SynthConfig, generate_synthetic_data  # noqa: E402


def _configure_logging(verbose: bool) -> None:
    """Настраивает логирование.

    Args:
        verbose: если ``True`` — уровень DEBUG.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _print_frame(title: str, frame: pd.DataFrame, rows: int = 15) -> None:
    """Печатает таблицу с заголовком.

    Args:
        title: заголовок.
        frame: таблица.
        rows: сколько строк показать.
    """
    print(f"\n=== {title} ===")
    if frame is None or frame.empty:
        print("(пусто)")
        return
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(frame.head(rows).to_string())


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки.

    Returns:
        Namespace с аргументами.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=48, help="число синтетических групп")
    parser.add_argument("--noise", type=int, default=400, help="убытков вне групп")
    parser.add_argument("--seed", type=int, default=20240517, help="зерно генератора")
    parser.add_argument("--share", type=float, default=0.35, help="доля вклада сервиса")
    parser.add_argument("--legacy", action="store_true", help="сравнить с исходной логикой")
    parser.add_argument("--verbose", action="store_true", help="подробные логи")
    return parser.parse_args()


def main() -> int:
    """Точка входа.

    Returns:
        Код возврата.
    """
    args = parse_args()
    _configure_logging(args.verbose)

    started = time.perf_counter()
    config = SynthConfig(groups=args.groups, noise_losses=args.noise, seed=args.seed)
    data = generate_synthetic_data(config)
    print(data.summary())
    generation_time = time.perf_counter() - started

    started = time.perf_counter()
    result = run_pipeline(
        data.beneficiaries,
        data.losses,
        data.cards,
        persist=False,
    )
    pipeline_time = time.perf_counter() - started

    print(f"\nГенерация: {generation_time:.2f} c | Пайплайн: {pipeline_time:.2f} c")
    print(result.as_text())

    _print_frame("Сводка по группам", result.summary)
    _print_frame("Эффект по группам", result.group_effect)
    _print_frame("Классы релевантности", headcount_table(result.summary))
    _print_frame("Эффект по годам", effect_by_year_table(result.group_effect, result.params))
    _print_frame("Эффект по когортам", effect_by_cohort_table(result.group_effect, result.params))
    _print_frame("Топ групп", top_groups_table(result.group_effect))
    _print_frame("Сверка", pd.DataFrame([result.reconciliation]))

    print(f"\nИТОГОВЫЙ ЭФФЕКТ: {result.total_effect:,.0f} руб.".replace(",", " "))
    print(
        f"Предотвращённый убыток (до доли сервиса): "
        f"{result.total_prevented_loss:,.0f} руб.".replace(",", " ")
    )

    if args.legacy:
        _run_legacy_comparison(data, result)

    truth = data.groups
    if not truth.empty and not result.summary.empty:
        merged = result.summary.merge(
            truth.loc[:, ["group_id", "scenario", "expected_relevance"]], on="group_id", how="left"
        )
        match = (merged["group_relevance"] == merged["expected_relevance"]).mean()
        print(f"\nСовпадение класса релевантности с эталоном: {match:.1%}")
        mismatched = merged.loc[merged["group_relevance"] != merged["expected_relevance"]]
        if not mismatched.empty:
            _print_frame("Расхождения с эталоном", mismatched)

    _ = effect_matrix  # используется в тестах
    return 0


def _run_legacy_comparison(data, result) -> None:
    """Сравнивает корректный расчёт с логикой исходной тетради.

    Args:
        data: синтетические данные.
        result: результат корректного прогона.
    """
    print("\n\n########## СРАВНЕНИЕ С ИСХОДНОЙ ЛОГИКОЙ ##########")

    money = legacy_vs_corrected_money(data.losses)
    money["Сумма, млн ₽"] = money["Сумма, ₽"] / 1_000_000
    _print_frame(
        "Денежная база: как было / как стало",
        money.loc[:, ["Вариант", "Сумма, млн ₽", "Множитель к корректной"]],
    )

    raw_beneficiaries = pd.DataFrame(
        {
            0: data.beneficiaries["full_name"].tolist(),
            1: data.beneficiaries["birth_date"].astype(str).tolist(),
            2: data.beneficiaries["group_number"].astype(str).tolist(),
        }
    )
    legacy_beneficiaries = legacy_beneficiaries_pipeline(raw_beneficiaries)
    legacy_joined = legacy_join_groups(data.losses, legacy_beneficiaries)
    legacy_final = legacy_detection_and_relevance(legacy_joined)
    legacy_totals = legacy_effect_totals(legacy_final)

    print(f"\nАктуальный (исправленный) эффект: {result.total_effect:,.0f} руб.".replace(",", " "))
    print(f"Исходная логика, эффект:         {legacy_totals['effect']:,.0f} руб.".replace(",", " "))
    print(f"Исходная логика, база:           {legacy_totals['pivot_total']:,.0f} руб.".replace(",", " "))
    ratio = legacy_totals["effect"] / result.total_effect if result.total_effect else float("nan")
    print(f"Отношение исходной к исправленной: {ratio:.2f}x")


if __name__ == "__main__":
    raise SystemExit(main())
