"""Общие фикстуры тестов: синтетические данные и результаты прогона."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fin_effects import synth  # noqa: E402
from fin_effects.config import EffectParams  # noqa: E402
from fin_effects.pipeline import PipelineResult, run_pipeline  # noqa: E402

#: Параметры методики для тестов (короткий горизонт, мягкие пороги).
TEST_PARAMS = EffectParams(
    service_share=0.35,
    min_losses_total=3,
    min_losses_per_year=1,
    first_year=2020,
    last_year=2026,
    project_years=2,
)


def build_synthetic(seed: int = 7, groups: int = 24, noise: int = 80, output_dir: Path | None = None):
    """Генерирует небольшой синтетический набор данных.

    Args:
        seed: зерно генератора.
        groups: число групп.
        noise: число «обычных» убытков.
        output_dir: каталог вывода (по умолчанию — временный).

    Returns:
        :class:`~fin_effects.synth.SynthResult`.
    """
    config = synth.SynthConfig(seed=seed, groups=groups, noise_losses=noise, output_dir=output_dir)
    return synth.generate_synthetic_data(config)


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory: pytest.TempPathFactory):
    """Сессионная фикстура с синтетическими данными (xlsx пишутся во временный каталог)."""
    output_dir = tmp_path_factory.mktemp("synthetic")
    return build_synthetic(output_dir=output_dir)


@pytest.fixture(scope="session")
def pipeline_result(synthetic) -> PipelineResult:
    """Результат прогона пайплайна на синтетических данных."""
    return run_pipeline(
        synthetic.beneficiaries,
        synthetic.losses,
        synthetic.cards,
        TEST_PARAMS,
    )


@pytest.fixture(scope="session")
def loss_numbers_with_multiple_payments(synthetic) -> list[str]:
    """Номера убытков, у которых больше одной строки в выгрузке.

    Args:
        synthetic: синтетические данные.

    Returns:
        Список номеров убытков.
    """
    counts = synthetic.losses["LossNumber"].value_counts()
    return counts[counts > 1].index.tolist()


@pytest.fixture
def small_losses() -> pd.DataFrame:
    """Мини-выгрузка убытков из двух строк одного убытка.

    Returns:
        Таблица убытков для точечных тестов.
    """
    return pd.DataFrame(
        {
            "LossNumber": ["L1", "L1"],
            "LossDateTime": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01")],
            "Виновник.Полн наименование": ["Иванов Иван", "Иванов Иван"],
            "Виновник.Дата рождения": [pd.Timestamp("1980-01-01")] * 2,
            "Потерпевший.Полн наименование": ["Петров Пётр", "Петров Пётр"],
            "Потерпевший.Дата рождения": [pd.Timestamp("1990-05-05")] * 2,
            "ValueRub": [100_000.0, 50_000.0],
            "Сумма_выплат_по_претензиям": [0.0, 0.0],
            "Сумма_взыскано_по_ФУ": [10_000.0, 0.0],
            "Суммы_взыскано_по_иску": [0.0, 0.0],
            "Объект виновника.Идентификатор": ["XTA0000000000001"] * 2,
            "Объект потерпевшего.Идентификатор": ["XTA0000000000002"] * 2,
            "Убыток.Объект страхователя.Идентификатор": ["XTA0000000000002"] * 2,
        }
    )
