"""Интерактивная визуализация группы в HTML."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
from pyvis.network import Network
from tqdm import tqdm

from graph_analytics.columns import LossColumns
from graph_analytics.state import PipelineState


class GraphVisualizer:
    """Отрисовка подграфа выбранной группы."""

    def __init__(self) -> None:
        self._cols = LossColumns()

    def render_group(
        self,
        state: PipelineState,
        group_number: int,
        output_path: Path | None = None,
    ) -> Path:
        """
        Построить HTML для группы (нумерация с 1).

        :param state: состояние пайплайна после compute_statistics
        :param group_number: номер группы в списке big_groups
        :param output_path: путь к HTML; по умолчанию output/Group_N.html
        """
        groups = state.require_groups()
        if group_number < 1 or group_number > len(groups):
            raise ValueError(f"Номер группы должен быть от 1 до {len(groups)}")

        group_index = group_number - 1
        group_nodes = groups[group_index]
        data = state.require_data()
        links = state.require_links()
        objects = state.objects
        people_count = state.people_count

        if objects is None:
            raise RuntimeError("Объекты не сформированы.")

        subgraph_links = links[
            links["obj1"].isin(group_nodes) & links["obj2"].isin(group_nodes)
        ]
        graph = nx.from_pandas_edgelist(
            subgraph_links, "obj1", "obj2", create_using=nx.Graph()
        )
        graph.remove_edges_from(nx.selfloop_edges(graph))

        self._style_nodes(graph, links, objects, people_count)
        self._style_edges(graph, links, data)
        graph = self._relabel_nodes(graph, objects, people_count)

        network = Network(
            "100vh",
            "100%",
            bgcolor="#ffffff",
            font_color="black",
            select_menu=True,
            filter_menu=True,
            cdn_resources="remote",
        )
        network.from_nx(graph)
        network.show_buttons(filter_=True)

        if output_path is None:
            output_path = Path("./output") / f"Group_{group_number}_visualisation.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        network.save_graph(str(output_path))
        return output_path

    def _style_nodes(
        self,
        graph: nx.Graph,
        links: pd.DataFrame,
        objects: pd.Series,
        people_count: int,
    ) -> None:
        for node in tqdm(graph.nodes, desc="Узлы"):
            types = links[(links["obj1"] == node) | (links["obj2"] == node)]["link_type"].unique()
            label = str(objects.iloc[node])
            if any(t in types for t in ("Victim_Culprit", "Culprit_Victim")):
                graph.nodes[node]["color"] = "red"
            elif node >= people_count:
                graph.nodes[node]["color"] = "black"
            else:
                graph.nodes[node]["color"] = "grey"
            graph.nodes[node]["title"] = (
                f"VIN: {label}" if node >= people_count else f"ФИО+ДР: {label}"
            )

    def _style_edges(self, graph: nx.Graph, links: pd.DataFrame, data: pd.DataFrame) -> None:
        loss_id_col = self._cols.loss_id
        for edge in tqdm(graph.edges, desc="Рёбра"):
            left, right = sorted(edge)
            types = links[
                ((links["obj1"] == left) & (links["obj2"] == right))
                | ((links["obj2"] == left) & (links["obj1"] == right))
            ]["link_type"].unique()
            loss_ids = self._loss_ids_for_edge(links, data, left, right, loss_id_col)

            if any(t in types for t in ("Culprit_VINv", "Victim_VINc")):
                color, weight, width = "black", 0, 1
            elif any(t in types for t in ("VINv_VINc", "VINc_VINv")):
                color, weight, width = "black", 2, 2
            elif any(t in types for t in ("Victim_VINv", "Culprit_VINc")):
                color, weight, width = "black", 3, 3
            elif any(t in types for t in ("Victim_Culprit", "Culprit_Victim")):
                color, weight, width = "red", 4, 4
            else:
                color, weight, width = "grey", 0, 1

            graph[left][right]["color"] = color
            graph[left][right]["weight"] = weight
            graph[left][right]["width"] = width
            graph[left][right]["title"] = ", ".join(loss_ids)

    def _loss_ids_for_edge(
        self,
        links: pd.DataFrame,
        data: pd.DataFrame,
        left: int,
        right: int,
        loss_id_col: str,
    ) -> list[str]:
        mask = (
            ((links["obj1"] == left) & (links["obj2"] == right))
            | ((links["obj2"] == left) & (links["obj1"] == right))
        ) & links["Loss_idx"].notna()
        indices = links.loc[mask, "Loss_idx"].unique()
        return [str(value) for value in data.loc[indices, loss_id_col].unique()]

    def _relabel_nodes(
        self,
        graph: nx.Graph,
        objects: pd.Series,
        people_count: int,
    ) -> nx.Graph:
        people_nodes = list(set(range(people_count)) & set(graph.nodes))
        vin_nodes = list(set(graph.nodes) - set(range(people_count)))

        labels: dict[int, str] = {}
        if people_nodes:
            names = (
                objects.iloc[people_nodes]
                .str.split()
                .apply(lambda parts: parts[0].lower().capitalize() if parts else "")
            )
            labels.update(names.to_dict())
        if vin_nodes:
            labels.update(objects.iloc[vin_nodes].to_dict())

        frame = pd.DataFrame(pd.Series(labels), columns=["base"])
        frame["label"] = frame["base"]
        duplicated = frame["base"].duplicated(keep=False)
        frame.loc[duplicated, "label"] = (
            frame.loc[duplicated, "base"]
            + " "
            + frame.groupby("base").cumcount().add(1).astype(str)
        )
        mapping = pd.Series(frame["label"].values, index=frame.index).to_dict()
        return nx.relabel_nodes(graph, mapping)
