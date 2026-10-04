"""Поиск кластеров и экспорт статистики."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
from tqdm import tqdm

from graph_analytics.columns import LossColumns
from graph_analytics.config import Settings
from graph_analytics.state import PipelineState


class StatisticsService:
    """Анализ связных компонент и выгрузка в Excel."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cols = LossColumns()

    def run(self, state: PipelineState) -> pd.DataFrame:
        """Назначить группы убыткам и сформировать сводную таблицу."""
        data = state.require_data()
        links = state.require_links()
        objects = state.objects
        if objects is None:
            raise RuntimeError("Объекты не сформированы.")

        big_groups = self._find_big_groups(links)
        state.big_groups = big_groups
        self._assign_groups(data, objects, big_groups)
        statistics = self._build_statistics_table(data, links, big_groups)
        state.statistics = statistics
        self._export_excel(data, statistics)
        return statistics

    def _find_big_groups(self, links: pd.DataFrame) -> list[set[int]]:
        graph = nx.from_pandas_edgelist(links, "obj1", "obj2", create_using=nx.Graph())
        graph.remove_edges_from(nx.selfloop_edges(graph))
        graph.remove_nodes_from([node for node in graph.nodes if graph.degree[node] <= 1])

        components = list(nx.connected_components(graph))
        if components:
            largest = max(components, key=len)
            graph.remove_nodes_from(largest)

        bound = self._settings.group_size_bound
        return [
            set(component)
            for component in nx.connected_components(graph)
            if len(component) > bound
        ]

    def _assign_groups(
        self,
        data: pd.DataFrame,
        objects: pd.Series,
        big_groups: list[set[int]],
    ) -> None:
        data["Группа"] = None
        objects_df = pd.DataFrame({"label": objects})

        for group_num, group in enumerate(tqdm(big_groups, desc="Группы"), start=1):
            labels = objects_df.loc[list(group), "label"].values
            mask = False
            for field in self._cols.link_fields():
                mask = mask | data[field].isin(labels)
            data.loc[mask, "Группа"] = group_num

    def _build_statistics_table(
        self,
        data: pd.DataFrame,
        links: pd.DataFrame,
        big_groups: list[set[int]],
    ) -> pd.DataFrame:
        rows = []
        for group_num, group in enumerate(big_groups, start=1):
            loss_count = (data["Группа"] == group_num).sum()
            rows.append(
                {
                    "Группа": group_num,
                    "Число убытков": loss_count,
                    "Число объектов": len(group),
                    "Число связей": links["obj1"].isin(group).sum(),
                    "Список ID участников группы": group,
                    "Средняя выплата": data.loc[
                        data["Группа"] == group_num, self._cols.amount
                    ].mean(),
                    "Средняя доля отказов": data.loc[
                        data["Группа"] == group_num, "Reject"
                    ].mean(),
                }
            )

        stat = pd.DataFrame(rows)
        stat["Плотность связей"] = stat["Число связей"] / stat["Число убытков"]
        stat["Плотность убытков"] = stat["Число убытков"] / stat["Число объектов"]
        return stat

    def _export_excel(self, data: pd.DataFrame, statistics: pd.DataFrame) -> Path:
        output_dir = self._settings.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "data.xlsx"
        grouped = data[data["Группа"].notnull()]

        if len(grouped) <= 1_000_000:
            with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
                grouped.to_excel(writer, index=False, sheet_name="data")
                statistics.to_excel(writer, index=False, sheet_name="statistics")
            return output_path

        for chunk_id, start in enumerate(range(0, len(grouped), 1_000_000)):
            chunk = grouped.iloc[start : start + 1_000_000]
            chunk_path = output_dir / f"data_{chunk_id}.xlsx"
            with pd.ExcelWriter(chunk_path, engine="xlsxwriter") as writer:
                writer.book.use_zip64()
                chunk.to_excel(writer, index=False, sheet_name="data")
                statistics.to_excel(writer, index=False, sheet_name="statistics")
        return output_dir / "data_0.xlsx"
