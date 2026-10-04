"""Точка входа CLI."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from graph_analytics.application import DataSource, GraphAnalyticsApplication
from graph_analytics.config import Settings


def build_parser() -> argparse.ArgumentParser:
    """Собрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Анализ связей по страховым убыткам и визуализация графов.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Каталог с .sql или .xlsx (по умолчанию INPUT_DIR из env)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Каталог результатов (по умолчанию OUTPUT_DIR из env)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    pipeline = subparsers.add_parser("pipeline", help="Полный цикл обработки")
    pipeline.add_argument(
        "source",
        choices=[DataSource.SQL.value, DataSource.EXCEL.value],
        help="Источник данных",
    )
    pipeline.add_argument(
        "--group",
        type=int,
        default=None,
        help="Номер группы для HTML-визуализации (1..N)",
    )

    return parser


def _resolve_settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    if args.input_dir:
        settings = replace(settings, input_dir=args.input_dir)
    if args.output_dir:
        settings = replace(settings, output_dir=args.output_dir)
    return settings


def main(argv: list[str] | None = None) -> int:
    """Запуск CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "pipeline":
        parser.error("Доступна команда: pipeline")
        return 1

    app = GraphAnalyticsApplication(_resolve_settings(args))
    app.run_pipeline(
        DataSource(args.source),
        args.input_dir,
        visualize_group=args.group,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
