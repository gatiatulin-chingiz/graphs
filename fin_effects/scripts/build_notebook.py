"""Сборка итогового ноутбука: KPI, эффект по годам, HTML-отчёт.

Запуск::

    python scripts/build_notebook.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import nbformat as nbf  # noqa: E402

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "financial_effect.ipynb"


def build_notebook() -> nbf.NotebookNode:
    """Собирает компактный ноутбук с расчётом и отчётом."""
    notebook = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []
    markdown = lambda text: cells.append(nbf.v4.new_markdown_cell(text))  # noqa: E731
    code = lambda text: cells.append(nbf.v4.new_code_cell(text))  # noqa: E731

    markdown(
        """# Финансовый эффект сервиса графовой аналитики

Краткий прогон: загрузка данных → пайплайн → KPI → эффект по годам → HTML-отчёт.
Логика расчёта — в `src/fin_effects` (`pytest`)."""
    )

    markdown("## 1. Окружение и параметры")
    code(
        """import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fin_effects.config import EffectParams, FIGURE_DIR, REPORT_DIR
from fin_effects.pipeline import run_pipeline
from fin_effects.reporting import (
    build_figures,
    effect_by_year_table,
    headcount_table,
    pipeline_overview_table,
    write_html_report,
)
from fin_effects.synth import SynthConfig, generate_synthetic_data

pd.options.display.float_format = "{:,.1f}".format

SERVICE_SHARE = 0.35
PROJECT_YEARS = 3
MIN_LOSSES_TOTAL = 3
MIN_LOSSES_PER_YEAR = 3
FIRST_YEAR = 2020
LAST_YEAR = 2026

PARAMS = EffectParams(
    service_share=SERVICE_SHARE,
    project_years=PROJECT_YEARS,
    min_losses_total=MIN_LOSSES_TOTAL,
    min_losses_per_year=MIN_LOSSES_PER_YEAR,
    first_year=FIRST_YEAR,
    last_year=LAST_YEAR,
    baseline_window="full",
)
print("Проект:", PROJECT_ROOT.resolve())
PARAMS"""
    )

    markdown(
        """## 2. Данные

* **Синтетика** (`SYNTHETIC=True`) — для проверки без БД.
* **Прод** — Excel в `data/raw/<YYYY>/`, убытки/карточки из parquet или SQL.
* `REFRESH_SQL` / `REFRESH_EXCEL` — принудительно перечитать источники."""
    )
    code(
        """SYNTHETIC = True
REFRESH_SQL = False
REFRESH_EXCEL = False

if SYNTHETIC:
    data = generate_synthetic_data(
        SynthConfig(
            seed=20240517,
            groups=120,
            noise_losses=1500,
            year_range=(FIRST_YEAR, LAST_YEAR),
            write_workbooks=False,
        )
    )
    BENEFICIARIES, LOSSES, CARDS = data.beneficiaries, data.losses, data.cards
    DATA_LABEL = "синтетика"
else:
    from fin_effects.config import ARTIFACT_BENEFICIARIES
    from fin_effects.excel_beneficiaries import load_beneficiaries_from_dir
    from fin_effects.loaders import load_cards, load_losses

    BENEFICIARIES, _stats = load_beneficiaries_from_dir(
        PROJECT_ROOT / "data" / "raw",
        refresh=REFRESH_EXCEL,
        cache_path=ARTIFACT_BENEFICIARIES,
    )
    LOSSES = load_losses(refresh=REFRESH_SQL)
    CARDS = load_cards(refresh=REFRESH_SQL)
    DATA_LABEL = "прод (Excel + кэш/SQL)"
    print(_stats.as_text())

print(f"Участников: {len(BENEFICIARIES):,}, групп: {BENEFICIARIES['group_id'].nunique():,}")
print(f"Убытков (строк): {len(LOSSES):,}, уник.: {LOSSES['LossNumber'].nunique():,}")"""
    )

    markdown(
        """## 3. Расчёт

Сопоставление убытков с группами: если в Excel есть номера убытков (`loss_number`),
матч **только по номеру** (не по всем убыткам участника из КАСКО)."""
    )
    code(
        """result = run_pipeline(BENEFICIARIES, LOSSES, CARDS, PARAMS, persist=True)
print(result.as_text())
pipeline_overview_table(result)"""
    )

    markdown(
        """## 4. Классы групп и эффект по годам

* **Распалась после обнаружения** — последний убыток ровно в год обнаружения (редко на проде).
* **Продолжает действовать** — убытки шли и после обнаружения → эффект на `project_years` лет.
* Колонка «Убытков (Σ по группам)» суммирует по группам и **может завышать**; смотрите «уник.»."""
    )
    code(
        """headcount_table(result.summary, result.facts_with_groups)"""
    )
    code(
        """effect_by_year_table(result.group_effect, PARAMS)"""
    )

    markdown(
        """## 5. HTML-отчёт

Файл по умолчанию: **`outputs/reports/financial_effect.html`** (от корня `fin_effects`)."""
    )
    code(
        """figures = build_figures(result.group_effect, result.summary, PARAMS, FIGURE_DIR)
html_path = write_html_report(
    result,
    REPORT_DIR / "financial_effect.html",
    figures=figures,
    source_label=DATA_LABEL,
)
print("HTML-отчёт:", html_path.resolve())"""
    )

    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    return notebook


def main() -> int:
    """Точка входа."""
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), str(NOTEBOOK_PATH))
    print(f"Ноутбук собран: {NOTEBOOK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
