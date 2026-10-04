"""Состояние пайплайна в памяти."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from graph_analytics.columns import LossColumns


@dataclass
class PipelineState:
    """Данные и результаты этапов обработки."""

    columns: LossColumns = field(default_factory=LossColumns)
    data: pd.DataFrame | None = None
    people: pd.Series | None = None
    vin: pd.Series | None = None
    objects: pd.Series | None = None
    links: pd.DataFrame | None = None
    big_groups: list[set[int]] = field(default_factory=list)
    statistics: pd.DataFrame | None = None

    @property
    def people_count(self) -> int:
        """Число узлов-людей (индексы VIN начинаются после них)."""
        if self.people is None:
            return 0
        return len(self.people)

    def require_data(self) -> pd.DataFrame:
        if self.data is None:
            raise RuntimeError("Данные не загружены. Выполните load.")
        return self.data

    def require_links(self) -> pd.DataFrame:
        if self.links is None:
            raise RuntimeError("Связи не построены. Выполните build_links.")
        return self.links

    def require_groups(self) -> list[set[int]]:
        if not self.big_groups:
            raise RuntimeError("Группы не найдены. Выполните compute_statistics.")
        return self.big_groups
