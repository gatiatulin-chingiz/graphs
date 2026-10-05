"""Замер производительности: исходная логика против оптимизированной.

Запуск::

    python scripts/benchmark.py [--rows 20000] [--groups 200]

Скрипт считает время по шагам пайплайна и сравнивает:

* ``legacy_loop_filter`` — исходный вложенный цикл тетради
  (``for group in groups: losses[losses['groups_list'] == group]``);
* векторизованные операции модуля :mod:`fin_effects.preprocess` /
  :mod:`fin_effects.groups`;
* чтение выгрузки сервиса в одном процессе и в пуле.

Результат сохраняется в ``outputs/reports/benchmark.csv`` и
``outputs/figures/benchmark.png``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fin_effects.config import FIGURE_DIR, REPORT_DIR, VSK_ACCENT, VSK_BLUE, EffectParams  # noqa: E402
from fin_effects.excel_beneficiaries import load_beneficiaries  # noqa: E402
from fin_effects.groups import build_group_summary, classify_group_relevance  # noqa: E402
from fin_effects.preprocess import (  # noqa: E402
    attach_groups_to_losses,
    collapse_losses_to_facts,
    normalize_losses,
)
from fin_effects.synth import SynthConfig, generate_synthetic_data  # noqa: E402

LOGGER = logging.getLogger("benchmark")


@dataclass
class Measurement:
    """Результат одного замера.

    Attributes:
        step: название шага.
        seconds: время выполнения.
        rows: число строк на входе (для контекста).
    """

    step: str
    seconds: float
    rows: int

    def as_dict(self) -> dict[str, object]:
        """Представление для таблицы.

        Returns:
            Словарь с полями шага.
        """
        return {"Шаг": self.step, "Секунд": round(self.seconds, 3), "Строк": self.rows}


def time_call(step: str, rows: int, function, *args, **kwargs) -> tuple[object, Measurement]:
    """Замеряет время вызова.

    Args:
        step: название шага.
        rows: число строк на входе.
        function: вызываемая функция.
        *args: позиционные аргументы.
        **kwargs: именованные аргументы.

    Returns:
        Кортеж «результат, замер».
    """
    started = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed = time.perf_counter() - started
    LOGGER.info("%-38s %8.3f c", step, elapsed)
    return result, Measurement(step=step, seconds=elapsed, rows=rows)


def legacy_loop_filter(losses: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Исходная логика отбора строк группы вложенным циклом.

    Args:
        losses: таблица убытков с колонкой ``groups_list``.
        groups: список идентификаторов групп.

    Returns:
        Склеенная таблица по всем группам.

    Note:
        Это буквальный перенос кода тетради::

            losses_ = pd.DataFrame()
            for i, group_id in enumerate(min_3_losses['groups_list']):
                j = losses[(losses['groups_list'] == str(group_id)) & ...]
                losses_ = pd.concat([losses_, j], axis=0)

        Сложность ``O(групп × убытков)`` плюс копирование на каждом шаге.
    """
    result = pd.DataFrame()
    for group_id in groups:
        subset = losses.loc[losses["groups_list"] == str(group_id)]
        result = pd.concat([result, subset], axis=0)
    return result


def benchmark_pipeline(rows: int, groups: int) -> list[Measurement]:
    """Прогоняет замеры по всем шагам пайплайна.

    Args:
        rows: целевое число строк убытков (регулируется числом «обычных» убытков).
        groups: число синтетических групп.

    Returns:
        Список замеров.
    """
    measurements: list[Measurement] = []
    data_config = SynthConfig(
        groups=groups,
        noise_losses=max(0, rows - groups * 20),
        output_dir=PROJECT_ROOT / "data" / "benchmark",
        write_workbooks=False,
    )
    data = generate_synthetic_data(data_config)
    losses = data.losses
    LOGGER.info("Данные: %d строк убытков, %d групп", len(losses), groups)

    normalized, measurement = time_call("normalize_losses", len(losses), normalize_losses, losses)
    measurements.append(measurement)

    collapsed, measurement = time_call(
        "collapse_losses_to_facts", len(normalized), collapse_losses_to_facts, normalized
    )
    measurements.append(measurement)
    facts, _collapse_stats = collapsed

    paired, measurement = time_call(
        "attach_groups_to_losses", len(facts), attach_groups_to_losses, facts, data.beneficiaries
    )
    measurements.append(measurement)

    # attach_groups_to_losses возвращает (соответствия, пары «убыток-группа»).
    _membership, pairs = paired

    built, measurement = time_call(
        "build_group_summary", len(pairs), build_group_summary, pairs, data.cards
    )
    measurements.append(measurement)
    summary, _group_stats = built

    classified, measurement = time_call(
        "classify_group_relevance", len(summary), classify_group_relevance, summary
    )
    measurements.append(measurement)
    del classified

    pairs = pairs.copy()
    pairs["groups_list"] = pairs["group_id"].astype(str)
    group_ids = pairs["group_id"].dropna().unique().tolist()

    legacy_sample = group_ids[: min(200, len(group_ids))]
    _, measurement = time_call(
        f"legacy_loop_filter ({len(legacy_sample)} групп)",
        len(pairs),
        legacy_loop_filter,
        pairs,
        legacy_sample,
    )
    measurements.append(measurement)

    _, measurement = time_call(
        "groupby вместо цикла (те же группы)",
        len(pairs),
        lambda frame, ids: frame[frame["group_id"].isin(ids)],
        pairs,
        legacy_sample,
    )
    measurements.append(measurement)

    for workers in (1, 4):
        (_, _), measurement = time_call(
            f"load_beneficiaries (workers={workers})",
            len(data.beneficiary_files),
            _load_beneficiaries_timed,
            data.beneficiary_files,
            workers,
        )
        measurements.append(measurement)

    return measurements


def _load_beneficiaries_timed(paths, workers: int):
    """Обёртка для замера чтения выгрузок сервиса.

    Args:
        paths: пути к файлам.
        workers: число процессов.

    Returns:
        Результат :func:`load_beneficiaries`.
    """
    return load_beneficiaries(paths, workers=workers)


def build_report(measurements: list[Measurement]) -> pd.DataFrame:
    """Сохраняет таблицу замеров и график.

    Args:
        measurements: список замеров.

    Returns:
        Таблица замеров.
    """
    table = pd.DataFrame([measurement.as_dict() for measurement in measurements])
    table["Доля от суммы, %"] = (table["Секунд"] / table["Секунд"].sum() * 100).round(1)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORT_DIR / "benchmark.csv", index=False)

    figure, axis = plt.subplots(figsize=(10, max(4.0, 0.32 * len(table))))
    ordered = table.iloc[::-1]
    colors = [VSK_ACCENT if "legacy" in step else VSK_BLUE for step in ordered["Шаг"]]
    axis.barh(ordered["Шаг"], ordered["Секунд"], color=colors)
    axis.set_xlabel("Секунды")
    axis.set_title("Замеры шагов пайплайна (красный — исходная логика)")
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / "benchmark.png", dpi=130, bbox_inches="tight")
    plt.close(figure)
    return table


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки.

    Returns:
        Namespace с аргументами.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000, help="сколько строк убытков")
    parser.add_argument("--groups", type=int, default=200, help="число групп")
    return parser.parse_args()


def main() -> int:
    """Точка входа.

    Returns:
        Код возврата.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = parse_args()

    started = time.perf_counter()
    measurements = benchmark_pipeline(args.rows, args.groups)
    table = build_report(measurements)

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("\n=== Замеры ===")
        print(table.to_string(index=False))

    total = table["Секунд"].sum()
    print(f"\nИтого: {total:.2f} c (общее время бенчмарка {time.perf_counter() - started:.2f} c)")
    legacy_rows = table.loc[table["Шаг"].str.contains("legacy_loop_filter")]
    if not legacy_rows.empty:
        print(
            "Исходный цикл против векторизованного отбора: "
            f"{legacy_rows['Секунд'].iloc[0]:.3f} c против "
            f"{table.loc[table['Шаг'].str.contains('groupby вместо'), 'Секунд'].iloc[0]:.3f} c"
        )
    _ = EffectParams
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
