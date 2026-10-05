"""Прогон пайплайна на синтетике и сборка HTML-отчёта по фин. эффекту.

Запуск::

    python scripts/build_html_report.py [--groups N] [--seed S]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fin_effects.config import EffectParams, REPORT_DIR  # noqa: E402
from fin_effects.pipeline import run_pipeline  # noqa: E402
from fin_effects.reporting import build_figures, write_html_report  # noqa: E402
from fin_effects.synth import SynthConfig, generate_synthetic_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=120, help="число синтетических групп")
    parser.add_argument("--noise", type=int, default=1500, help="убытков вне групп")
    parser.add_argument("--seed", type=int, default=20240517, help="зерно генератора")
    parser.add_argument("--share", type=float, default=0.35, help="доля вклада сервиса")
    parser.add_argument("--project-years", type=int, default=3, help="горизонт эффекта, лет")
    parser.add_argument("--first-year", type=int, default=2020)
    parser.add_argument("--last-year", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR / "financial_effect.html",
        help="путь к HTML-отчёту",
    )
    return parser.parse_args()


def main() -> int:
    """Точка входа: синтетика -> пайплайн -> HTML."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s", stream=sys.stdout)
    args = parse_args()

    params = EffectParams(
        service_share=args.share,
        project_years=args.project_years,
        first_year=args.first_year,
        last_year=args.last_year,
        min_losses_total=3,
        min_losses_per_year=3,
        baseline_window="full",
    )
    config = SynthConfig(
        seed=args.seed,
        groups=args.groups,
        noise_losses=args.noise,
        year_range=(args.first_year, args.last_year),
        write_workbooks=False,
    )

    started = time.perf_counter()
    data = generate_synthetic_data(config)
    result = run_pipeline(data.beneficiaries, data.losses, data.cards, params, persist=True)
    figures = build_figures(result.group_effect, result.summary, params)
    report_path = write_html_report(
        result,
        args.output,
        figures=figures,
        source_label=(
            f"синтетика (groups={args.groups}, noise={args.noise}, seed={args.seed})"
        ),
    )
    elapsed = time.perf_counter() - started

    print(result.as_text())
    print(f"\nHTML-отчёт: {report_path}")
    print(f"Время: {elapsed:.1f} c")
    print(f"Итоговый эффект: {result.total_effect / 1e6:,.1f} млн руб.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
