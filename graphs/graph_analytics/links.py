"""Построение рёбер графа по убыткам."""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from graph_analytics.columns import LossColumns
from graph_analytics.state import PipelineState


class LinkBuilder:
    """Связи между участниками одного убытка."""

    def __init__(self) -> None:
        self._cols = LossColumns()

    def run(self, state: PipelineState) -> None:
        data = state.require_data()
        objects = state.objects
        if objects is None:
            raise RuntimeError("Объекты не сформированы. Выполните preprocess.")

        link_fields = self._cols.link_fields()
        object_index = pd.Series(data=objects.index, index=objects.values)
        edges: list[list[object]] = []

        for row_idx in tqdm(range(len(data)), desc="Связи"):
            row = data.iloc[row_idx]
            for left_idx, left in enumerate(link_fields):
                for right in link_fields[left_idx + 1 :]:
                    idx_left = object_index.get(row[left])
                    idx_right = object_index.get(row[right])
                    if pd.notna(idx_left) and pd.notna(idx_right):
                        edges.append([idx_left, idx_right, row_idx, f"{left}_{right}"])

        state.links = pd.DataFrame(
            edges,
            columns=["obj1", "obj2", "Loss_idx", "link_type"],
        )
