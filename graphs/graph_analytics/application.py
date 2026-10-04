"""Оркестрация этапов анализа графа убытков."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import pandas as pd

from graph_analytics.config import Settings
from graph_analytics.links import LinkBuilder
from graph_analytics.loader import DataLoader
from graph_analytics.preprocessor import DataPreprocessor
from graph_analytics.state import PipelineState
from graph_analytics.statistics import StatisticsService
from graph_analytics.visualizer import GraphVisualizer


class DataSource(str, Enum):
    """Источник сырых данных."""

    SQL = "sql"
    EXCEL = "excel"


class GraphAnalyticsApplication:
    """Приложение для загрузки, анализа и визуализации связей по убыткам."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.state = PipelineState()
        self._loader = DataLoader(self.settings)
        self._preprocessor = DataPreprocessor(self.settings)
        self._link_builder = LinkBuilder()
        self._statistics = StatisticsService(self.settings)
        self._visualizer = GraphVisualizer()

    def load(self, source: DataSource, directory: Path | None = None) -> pd.DataFrame:
        """Загрузить и предобработать данные."""
        if source == DataSource.SQL:
            raw = self._loader.load_sql(directory)
        else:
            raw = self._loader.load_excel(directory)
        self._preprocessor.run(self.state, raw)
        return self.state.require_data()

    def build_links(self) -> pd.DataFrame:
        """Построить рёбра графа."""
        self._link_builder.run(self.state)
        return self.state.require_links()

    def compute_statistics(self) -> pd.DataFrame:
        """Найти группы и записать Excel."""
        return self._statistics.run(self.state)

    def visualize(self, group_number: int, output_path: Path | None = None) -> Path:
        """Сохранить HTML-визуализацию группы."""
        if output_path is None:
            output_path = self.settings.output_dir / f"Group_{group_number}_visualisation.html"
        return self._visualizer.render_group(self.state, group_number, output_path)

    def run_pipeline(
        self,
        source: DataSource,
        directory: Path | None = None,
        visualize_group: int | None = None,
    ) -> None:
        """Полный цикл: загрузка → связи → статистика → (опционально) визуализация."""
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        self.load(source, directory)
        self.build_links()
        statistics = self.compute_statistics()
        print(f"Найдено групп: {len(statistics)}")
        print(f"Статистика: {self.settings.output_dir / 'data.xlsx'}")
        if visualize_group is not None:
            path = self.visualize(visualize_group)
            print(f"Визуализация: {path}")
