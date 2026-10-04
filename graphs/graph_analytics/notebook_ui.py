"""Интерфейс для Jupyter Notebook (ipywidgets)."""

from __future__ import annotations

from pathlib import Path

import ipywidgets as widgets
from IPython.display import IFrame, display

from graph_analytics.application import DataSource, GraphAnalyticsApplication


class NotebookUI:
    """Панель управления анализом графа для бизнес-пользователей в Jupyter."""

    def __init__(self, app: GraphAnalyticsApplication | None = None) -> None:
        self.app = app or GraphAnalyticsApplication()
        self._log = widgets.Output(layout={"border": "1px solid #ccc", "padding": "8px"})

    def display(self) -> None:
        """Показать панель кнопок в ячейке ноутбука."""
        self.app.settings.output_dir.mkdir(parents=True, exist_ok=True)
        display(self._build_panel())

    def _build_panel(self) -> widgets.VBox:
        input_dir = widgets.Text(
            value=str(self.app.settings.input_dir),
            description="Вход:",
            style={"description_width": "60px"},
            layout=widgets.Layout(width="98%"),
        )
        output_dir = widgets.Text(
            value=str(self.app.settings.output_dir),
            description="Выход:",
            style={"description_width": "60px"},
            layout=widgets.Layout(width="98%"),
        )

        btn_load_sql = widgets.Button(description="1. Загрузить из SQL", button_style="primary")
        btn_load_excel = widgets.Button(description="1. Загрузить из Excel", button_style="primary")
        btn_links = widgets.Button(description="2. Создать связи")
        btn_stats = widgets.Button(description="3. Статистика → Excel")
        btn_show_stats = widgets.Button(description="Показать таблицу групп")

        group_num = widgets.IntText(value=1, description="Группа №", style={"description_width": "70px"})
        btn_viz = widgets.Button(description="4. Визуализация → HTML", button_style="success")
        btn_open_html = widgets.Button(description="Открыть HTML в ноутбуке")

        btn_load_sql.on_click(lambda _: self._on_load(DataSource.SQL, input_dir, output_dir))
        btn_load_excel.on_click(lambda _: self._on_load(DataSource.EXCEL, input_dir, output_dir))
        btn_links.on_click(lambda _: self._on_links())
        btn_stats.on_click(lambda _: self._on_stats())
        btn_show_stats.on_click(lambda _: self._on_show_stats())
        btn_viz.on_click(lambda _: self._on_visualize(group_num))
        btn_open_html.on_click(lambda _: self._on_open_html(group_num))

        header = widgets.HTML(
            "<h3>Анализ связей по убыткам</h3>"
            "<p>Шаги выполняйте по порядку. Файлы .sql / .xlsx — во входном каталоге.</p>"
        )
        steps = widgets.HBox([btn_load_sql, btn_load_excel, btn_links, btn_stats])
        viz_row = widgets.HBox([group_num, btn_viz, btn_open_html, btn_show_stats])

        return widgets.VBox(
            [
                header,
                input_dir,
                output_dir,
                steps,
                viz_row,
                widgets.HTML("<b>Журнал:</b>"),
                self._log,
            ]
        )

    def _sync_dirs(self, input_widget: widgets.Text, output_widget: widgets.Text) -> None:
        from dataclasses import replace

        self.app.settings = replace(
            self.app.settings,
            input_dir=Path(input_widget.value.strip() or "."),
            output_dir=Path(output_widget.value.strip() or "./output"),
        )
        self.app.settings.output_dir.mkdir(parents=True, exist_ok=True)

    def _on_load(
        self,
        source: DataSource,
        input_widget: widgets.Text,
        output_widget: widgets.Text,
    ) -> None:
        with self._log:
            self._log.clear_output()
            try:
                self._sync_dirs(input_widget, output_widget)
                label = "SQL" if source == DataSource.SQL else "Excel"
                print(f"Загрузка из {label}: {self.app.settings.input_dir} …")
                data = self.app.load(source, self.app.settings.input_dir)
                print(f"Готово. Строк: {len(data)}, объектов: {len(self.app.state.objects)}")
            except Exception as exc:
                print(f"Ошибка: {exc}")

    def _on_links(self) -> None:
        with self._log:
            self._log.clear_output()
            try:
                if self.app.state.data is None:
                    raise RuntimeError("Сначала загрузите данные (шаг 1).")
                print("Построение связей…")
                links = self.app.build_links()
                print(f"Готово. Рёбер: {len(links)}")
            except Exception as exc:
                print(f"Ошибка: {exc}")

    def _on_stats(self) -> None:
        with self._log:
            self._log.clear_output()
            try:
                if self.app.state.links is None:
                    raise RuntimeError("Сначала создайте связи (шаг 2).")
                print("Расчёт групп и запись Excel…")
                stat = self.app.compute_statistics()
                path = self.app.settings.output_dir / "data.xlsx"
                print(f"Готово. Групп: {len(stat)}")
                print(f"Файл: {path}")
            except Exception as exc:
                print(f"Ошибка: {exc}")

    def _on_show_stats(self) -> None:
        with self._log:
            try:
                if self.app.state.statistics is None:
                    raise RuntimeError("Сначала выполните шаг 3 (статистика).")
                display(self.app.state.statistics)
            except Exception as exc:
                print(f"Ошибка: {exc}")

    def _on_visualize(self, group_widget: widgets.IntText) -> None:
        with self._log:
            self._log.clear_output()
            try:
                if not self.app.state.big_groups:
                    raise RuntimeError("Сначала выполните шаг 3 (статистика).")
                group = int(group_widget.value)
                print(f"Визуализация группы {group}…")
                path = self.app.visualize(group)
                print(f"Готово: {path}")
                print("Нажмите «Открыть HTML в ноутбуке» для просмотра.")
            except Exception as exc:
                print(f"Ошибка: {exc}")

    def _on_open_html(self, group_widget: widgets.IntText) -> None:
        with self._log:
            try:
                group = int(group_widget.value)
                path = self.app.settings.output_dir / f"Group_{group}_visualisation.html"
                if not path.is_file():
                    raise FileNotFoundError(f"Файл не найден: {path}. Сначала выполните шаг 4.")
                display(IFrame(src=str(path.resolve()), width="100%", height="700px"))
            except Exception as exc:
                print(f"Ошибка: {exc}")


def run_notebook_ui(app: GraphAnalyticsApplication | None = None) -> NotebookUI:
    """
    Создать и показать панель в ноутбуке.

    Пример в ячейке::

        from graph_analytics.notebook_ui import run_notebook_ui
        run_notebook_ui()
    """
    ui = NotebookUI(app)
    ui.display()
    return ui
