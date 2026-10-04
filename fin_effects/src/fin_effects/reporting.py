"""Сводные таблицы, графики и экспорт итогов.

В ноутбуке остаются только вызовы этого модуля и визуализация: вся логика
построения таблиц живёт здесь, чтобы её можно было тестировать.
"""

from __future__ import annotations

import base64
import html
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import (
    DEFAULT_PARAMS,
    FIGURE_DIR,
    REPORT_DIR,
    RELEVANCE_ALIVE,
    RELEVANCE_LABELS,
    VSK_ACCENT,
    VSK_BLUE,
    VSK_BLUE_DARK,
    VSK_BLUE_MID,
    VSK_INK,
    VSK_MUTED,
    VSK_PALETTE,
    VSK_RELEVANCE_COLORS,
    VSK_SKY,
    VSK_SKY_SOFT,
    EffectParams,
)
from .effect import effect_long
from .io_utils import format_money

if TYPE_CHECKING:
    from .pipeline import PipelineResult

LOGGER = logging.getLogger(__name__)

#: Цвета классов релевантности (брендбук ВСК).
RELEVANCE_COLORS = dict(VSK_RELEVANCE_COLORS)

_PALETTE = VSK_PALETTE


def _style_axes(ax) -> None:
    """Применяет базовый стиль осей в цветах ВСК.

    Args:
        ax: оси matplotlib.
    """
    ax.set_facecolor("white")
    ax.tick_params(colors=VSK_INK)
    ax.xaxis.label.set_color(VSK_INK)
    ax.yaxis.label.set_color(VSK_INK)
    ax.title.set_color(VSK_BLUE_DARK)
    for spine in ax.spines.values():
        spine.set_color("#D5DCE8")
    ax.grid(axis="y", color=VSK_SKY_SOFT, alpha=1.0)


def money_table(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Форматирует денежные колонки в млн ₽ (возвращает копию).

    Args:
        frame: таблица.
        columns: колонки для конвертации; ``None`` — только ``float``-колонки.

    Returns:
        Копия таблицы с суммами в миллионах рублей.
    """
    result = frame.copy()
    target = columns or [
        column for column in result.columns if pd.api.types.is_float_dtype(result[column])
    ]
    for column in target:
        if column in result.columns:
            result[column] = result[column] / 1_000_000
    return result


def effect_by_year_table(
    group_effect: pd.DataFrame,
    params: EffectParams = DEFAULT_PARAMS,
    *,
    by_class: bool = False,
) -> pd.DataFrame:
    """Эффект по годам.

    Args:
        group_effect: таблица эффекта по группам.
        params: параметры методики.
        by_class: если ``True`` — дополнительно разрез по классу релевантности.

    Returns:
        Таблица «год [, класс], групп, эффект, накопленный эффект (в млн ₽)».
    """
    long_frame = effect_long(group_effect, params)
    if long_frame.empty:
        columns = (
            ["Год", "Класс группы", "Групп", "Эффект, млн ₽", "Накопленный эффект, млн ₽"]
            if by_class
            else ["Год", "Групп", "Эффект, млн ₽", "Накопленный эффект, млн ₽"]
        )
        return pd.DataFrame(columns=columns)

    if by_class:
        grouped = (
            long_frame.groupby(["year", "group_relevance"], as_index=False)
            .agg(groups=("group_id", "nunique"), effect=("effect", "sum"))
            .sort_values(["year", "group_relevance"])
            .rename(columns={"year": "Год", "group_relevance": "Класс группы", "groups": "Групп"})
        )
        value_columns = ["Год", "Класс группы", "Групп", "Эффект, млн ₽", "Накопленный эффект, млн ₽"]
    else:
        grouped = (
            long_frame.groupby("year", as_index=False)
            .agg(groups=("group_id", "nunique"), effect=("effect", "sum"))
            .sort_values("year")
            .rename(columns={"year": "Год", "groups": "Групп"})
        )
        value_columns = ["Год", "Групп", "Эффект, млн ₽", "Накопленный эффект, млн ₽"]

    grouped["Эффект, млн ₽"] = grouped["effect"] / 1_000_000
    grouped["Накопленный эффект, млн ₽"] = grouped["Эффект, млн ₽"].cumsum()
    return grouped.loc[:, value_columns]


def effect_by_cohort_table(
    group_effect: pd.DataFrame,
    params: EffectParams = DEFAULT_PARAMS,
) -> pd.DataFrame:
    """Эффект по когортам года обнаружения.

    Args:
        group_effect: таблица эффекта по группам.
        params: параметры методики.

    Returns:
        Таблица с числом групп и эффектом по каждой когорте (в млн ₽).
    """
    if group_effect.empty:
        return pd.DataFrame(columns=["Год обнаружения", "Групп", "Эффект, млн ₽"])

    long_frame = effect_long(group_effect, params)
    projected = (
        long_frame.groupby("detection_year", as_index=False)["effect"].sum()
        if not long_frame.empty
        else pd.DataFrame(columns=["detection_year", "effect"])
    )

    per_group = (
        group_effect.groupby("detection_year", as_index=False)
        .agg(
            groups=("group_id", "nunique"),
            net_loss_total=("net_loss_total", "sum"),
            baseline_total=("mean_loss_per_active_year", "sum"),
            effect_total=("effect_total", "sum"),
        )
        .sort_values("detection_year")
        .merge(projected, on="detection_year", how="left")
        .fillna({"effect": 0.0})
    )

    result = per_group.rename(
        columns={
            "detection_year": "Год обнаружения",
            "groups": "Групп",
            "net_loss_total": "Убыток групп, млн ₽",
            "baseline_total": "Предотвращённый убыток, млн ₽",
            "effect_total": "Эффект, млн ₽",
            "effect": "Эффект в горизонте, млн ₽",
        }
    )
    money_columns = [
        "Убыток групп, млн ₽",
        "Предотвращённый убыток, млн ₽",
        "Эффект, млн ₽",
        "Эффект в горизонте, млн ₽",
    ]
    for column in money_columns:
        if column in result.columns:
            result[column] = result[column] / 1_000_000
    return result.loc[:, ["Год обнаружения", "Групп", *money_columns]].reset_index(drop=True)


def top_groups_table(group_effect: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Топ групп по эффекту.

    Args:
        group_effect: таблица эффекта по группам.
        top_n: сколько строк показать.

    Returns:
        Таблица «группа, класс, год обнаружения, убыток, база, эффект».
    """
    if group_effect.empty:
        return pd.DataFrame(columns=["group_id"])

    top = group_effect.sort_values("effect_total", ascending=False).head(top_n).copy()
    result = top.loc[
        :,
        [
            "group_id",
            "group_relevance",
            "detection_year",
            "losses_total",
            "net_loss_total",
            "mean_loss_per_active_year",
            "effect_total",
        ],
    ].rename(
        columns={
            "group_id": "Группа",
            "group_relevance": "Класс",
            "detection_year": "Год обнаружения",
            "losses_total": "Убытков",
            "net_loss_total": "Убыток, млн ₽",
            "mean_loss_per_active_year": "База в год, млн ₽",
            "effect_total": "Эффект, млн ₽",
        }
    )
    for column in ("Убыток, млн ₽", "База в год, млн ₽", "Эффект, млн ₽"):
        result[column] = result[column] / 1_000_000
    return result.reset_index(drop=True)


def sensitivity_table(
    group_effect: pd.DataFrame,
    params: EffectParams = DEFAULT_PARAMS,
    shares: tuple[float, ...] = (0.2, 0.35, 0.5, 0.7),
) -> pd.DataFrame:
    """Чувствительность эффекта к доле вклада сервиса.

    Args:
        group_effect: таблица эффекта по группам.
        params: параметры методики.
        shares: проверяемые значения доли.

    Returns:
        Таблица «доля сервиса -> эффект, млн ₽».
    """
    base_total = float(group_effect["effect_total"].sum()) if not group_effect.empty else 0.0
    if not params.service_share:
        return pd.DataFrame(columns=["Доля сервиса", "Эффект, млн ₽"])
    rows = [
        {
            "Доля сервиса": share,
            "Эффект, млн ₽": base_total * share / params.service_share / 1_000_000,
        }
        for share in shares
    ]
    return pd.DataFrame(rows)


def headcount_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Сводка по классам релевантности: сколько групп и какие годы.

    Args:
        summary: сводка по группам.

    Returns:
        Таблица «класс, групп, убытков, медиана лет жизни».
    """
    if summary.empty:
        return pd.DataFrame(columns=["Класс", "Групп"])

    grouped = (
        summary.groupby("group_relevance", as_index=False)
        .agg(
            groups=("group_id", "nunique"),
            losses=("losses_total", "sum"),
            median_lifetime=("lifetime_years", "median"),
            mean_active_years=("active_years_count", "mean"),
            net_loss=("net_loss_total", "sum"),
        )
        .rename(
            columns={
                "group_relevance": "Класс",
                "groups": "Групп",
                "losses": "Убытков",
                "median_lifetime": "Медиана лет жизни",
                "mean_active_years": "Активных лет (среднее)",
                "net_loss": "Убыток, млн ₽",
            }
        )
        .sort_values("Групп", ascending=False)
    )
    grouped["Убыток, млн ₽"] = grouped["Убыток, млн ₽"] / 1_000_000
    return grouped.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------


def _finish_figure(fig: plt.Figure, path: Path, *, dpi: int = 130) -> Path:
    """Сохраняет фигуру и закрывает её.

    Args:
        fig: фигура.
        path: путь сохранения.
        dpi: разрешение.

    Returns:
        Путь к файлу.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("График сохранён: %s", path)
    return path


def plot_effect_by_year(
    group_effect: pd.DataFrame,
    params: EffectParams = DEFAULT_PARAMS,
    path: Path | None = None,
) -> Path:
    """Столбцы эффекта по годам с разбивкой по классам релевантности.

    Args:
        group_effect: таблица эффекта по группам.
        params: параметры методики.
        path: куда сохранить (по умолчанию ``outputs/figures/effect_by_year.png``).

    Returns:
        Путь к картинке.
    """
    long_frame = effect_long(group_effect, params)
    target = path or FIGURE_DIR / "effect_by_year.png"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if long_frame.empty:
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center")
        return _finish_figure(fig, target)

    by_year = (
        long_frame.groupby("year", as_index=False)["effect"]
        .sum()
        .sort_values("year")
    )
    years = by_year["year"].astype(str)
    values = by_year["effect"].to_numpy() / 1_000_000
    ax.bar(years, values, color=VSK_BLUE)

    for position, total in enumerate(values):
        ax.text(
            position,
            total,
            format_money(total * 1_000_000, "млн ₽"),
            ha="center",
            va="bottom",
            fontsize=9,
            color=VSK_BLUE_DARK,
        )

    ax.set_title("Финансовый эффект по годам")
    ax.set_xlabel("Год")
    ax.set_ylabel("Эффект, млн ₽")
    _style_axes(ax)
    return _finish_figure(fig, target)


def plot_effect_by_cohort(
    group_effect: pd.DataFrame,
    params: EffectParams = DEFAULT_PARAMS,
    path: Path | None = None,
) -> Path:
    """Эффект по когортам года обнаружения.

    Args:
        group_effect: таблица эффекта по группам.
        params: параметры методики.
        path: куда сохранить.

    Returns:
        Путь к картинке.
    """
    target = path or FIGURE_DIR / "effect_by_cohort.png"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if group_effect.empty:
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center")
        return _finish_figure(fig, target)

    table = effect_by_cohort_table(group_effect, params)
    positions = np.arange(len(table))
    ax.bar(
        positions - 0.2,
        table["Эффект, млн ₽"],
        width=0.4,
        label="Эффект (год обнаружения)",
        color=VSK_BLUE,
    )
    ax.bar(
        positions + 0.2,
        table["Эффект в горизонте, млн ₽"],
        width=0.4,
        label=f"Эффект (горизонт {params.project_years} года)",
        color=VSK_ACCENT,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(table["Год обнаружения"].astype(str))
    ax.set_title("Эффект по когортам обнаружения")
    ax.set_xlabel("Год обнаружения")
    ax.set_ylabel("Эффект, млн ₽")
    ax.legend(fontsize=8, frameon=False)
    _style_axes(ax)
    return _finish_figure(fig, target)


def plot_relevance_split(summary: pd.DataFrame, path: Path | None = None) -> Path:
    """Круговая диаграмма распределения групп по классам релевантности.

    Args:
        summary: сводка по группам.
        path: куда сохранить.

    Returns:
        Путь к картинке.
    """
    target = path or FIGURE_DIR / "relevance_split.png"
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if summary.empty:
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center")
        return _finish_figure(fig, target)

    counts = summary["group_relevance"].value_counts().reindex(RELEVANCE_LABELS).fillna(0)
    colors = [RELEVANCE_COLORS[label] for label in RELEVANCE_LABELS]
    ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.0f%%",
        colors=colors,
        textprops={"fontsize": 8, "color": VSK_INK},
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    ax.set_title("Группы по классам релевантности", color=VSK_BLUE_DARK)
    return _finish_figure(fig, target)


def _has_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    """Проверяет, что таблица непуста и содержит нужные колонки.

    Args:
        frame: таблица.
        columns: обязательные колонки.

    Returns:
        ``True``, если можно строить график.
    """
    return not frame.empty and set(columns) <= set(frame.columns)


def plot_top_groups(group_effect: pd.DataFrame, top_n: int = 15, path: Path | None = None) -> Path:
    """Горизонтальные столбцы: топ групп по эффекту.

    Args:
        group_effect: таблица эффекта по группам.
        top_n: сколько групп показать.
        path: куда сохранить.

    Returns:
        Путь к картинке.
    """
    target = path or FIGURE_DIR / "top_groups.png"
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.32 * top_n)))
    if not _has_columns(group_effect, ("group_id", "effect_total", "group_relevance")):
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center")
        return _finish_figure(fig, target)

    top = group_effect.nlargest(top_n, "effect_total").iloc[::-1]
    colors = [RELEVANCE_COLORS.get(label, VSK_BLUE) for label in top["group_relevance"]]
    ax.barh(top["group_id"], top["effect_total"] / 1_000_000, color=colors)
    ax.set_title(f"Топ-{top_n} групп по эффекту")
    ax.set_xlabel("Эффект, млн ₽")
    _style_axes(ax)
    ax.grid(axis="x", color=VSK_SKY_SOFT, alpha=1.0)
    ax.grid(axis="y", visible=False)
    return _finish_figure(fig, target)


def plot_baseline_distribution(group_effect: pd.DataFrame, path: Path | None = None) -> Path:
    """Распределение среднего годового убытка группы.

    Args:
        group_effect: таблица эффекта по группам.
        path: куда сохранить.

    Returns:
        Путь к картинке.
    """
    target = path or FIGURE_DIR / "baseline_distribution.png"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if not _has_columns(group_effect, ("mean_loss_per_active_year",)):
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center")
        return _finish_figure(fig, target)

    values = (
        group_effect.loc[group_effect["mean_loss_per_active_year"] > 0, "mean_loss_per_active_year"]
        / 1_000_000
    )
    if values.empty:
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center")
        return _finish_figure(fig, target)

    ax.hist(
        values,
        bins=min(20, max(5, len(values) // 2)),
        color=VSK_BLUE,
        edgecolor="white",
    )
    ax.set_title("Средний годовой убыток группы (база эффекта)")
    ax.set_xlabel("млн ₽ в год")
    ax.set_ylabel("Групп")
    _style_axes(ax)
    return _finish_figure(fig, target)


def build_figures(
    group_effect: pd.DataFrame,
    summary: pd.DataFrame,
    params: EffectParams = DEFAULT_PARAMS,
    figures_dir: Path = FIGURE_DIR,
) -> dict[str, Path]:
    """Строит полный комплект графиков.

    Args:
        group_effect: таблица эффекта по группам.
        summary: сводка по группам.
        params: параметры методики.
        figures_dir: каталог для картинок.

    Returns:
        Словарь «имя графика -> путь».
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {
        "effect_by_year": plot_effect_by_year(group_effect, params, figures_dir / "effect_by_year.png"),
        "effect_by_cohort": plot_effect_by_cohort(
            group_effect, params, figures_dir / "effect_by_cohort.png"
        ),
        "relevance_split": plot_relevance_split(summary, figures_dir / "relevance_split.png"),
        "top_groups": plot_top_groups(group_effect, 15, figures_dir / "top_groups.png"),
        "baseline_distribution": plot_baseline_distribution(
            group_effect, figures_dir / "baseline_distribution.png"
        ),
    }


def summary_kpis(group_effect: pd.DataFrame, params: EffectParams = DEFAULT_PARAMS) -> dict[str, float]:
    """Ключевые цифры для шапки отчёта.

    Args:
        group_effect: таблица эффекта по группам.
        params: параметры методики.

    Returns:
        Словарь основных показателей.
    """
    if group_effect.empty:
        return {
            "groups": 0,
            "effect_total": 0.0,
            "effect_first_year": 0.0,
            "prevented_total": 0.0,
            "net_loss_total": 0.0,
        }
    return {
        "groups": float(group_effect["group_id"].nunique()),
        "effect_total": float(group_effect["effect_total"].sum()),
        "effect_first_year": float(group_effect["effect_first_year"].sum()),
        "prevented_total": float(group_effect["mean_loss_per_active_year"].sum()),
        "net_loss_total": float(group_effect["net_loss_total"].sum()),
        "service_share": float(params.service_share),
    }


def _df_to_html(frame: pd.DataFrame, *, float_fmt: str = "{:,.2f}") -> str:
    """Рендерит таблицу в HTML с базовым форматированием чисел.

    Args:
        frame: таблица.
        float_fmt: формат для float-колонок.

    Returns:
        HTML-фрагмент ``<table>...</table>``.
    """
    if frame is None or frame.empty:
        return "<p class=\"muted\">Нет данных</p>"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value, fmt=float_fmt: "—" if pd.isna(value) else fmt.format(value)
            )
    return display.to_html(index=False, escape=True, border=0, classes="data")


def _image_to_data_uri(path: Path) -> str:
    """Кодирует PNG в data-URI для встраивания в HTML.

    Args:
        path: путь к картинке.

    Returns:
        Строка ``data:image/png;base64,...`` или пустая строка.
    """
    if not path.exists():
        return ""
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def write_html_report(
    result: PipelineResult,
    path: str | Path | None = None,
    *,
    figures: dict[str, Path] | None = None,
    title: str = "Финансовый эффект сервиса графовой аналитики",
    source_label: str = "синтетические данные",
) -> Path:
    """Собирает самодостаточный HTML-отчёт по результатам пайплайна.

    Args:
        result: результат :func:`~fin_effects.pipeline.run_pipeline`.
        path: куда сохранить HTML; по умолчанию ``outputs/reports/financial_effect.html``.
        figures: словарь путей к PNG (если ``None`` — графики строятся заново).
        title: заголовок отчёта.
        source_label: подпись источника данных в шапке.

    Returns:
        Путь к сохранённому HTML-файлу.
    """
    target = Path(path) if path is not None else REPORT_DIR / "financial_effect.html"
    target.parent.mkdir(parents=True, exist_ok=True)

    params = result.params
    kpis = summary_kpis(result.group_effect, params)
    figure_paths = figures or build_figures(
        result.group_effect, result.summary, params, FIGURE_DIR
    )

    def _figure_section(key: str, title: str) -> str:
        """Одна секция с одним графиком на всю ширину."""
        if key not in figure_paths:
            return ""
        uri = _image_to_data_uri(Path(figure_paths[key]))
        if not uri:
            return ""
        return (
            f"<section class=\"figure-section\"><h2>{html.escape(title)}</h2>"
            f'<figure class="full"><img src="{uri}" alt="{html.escape(title)}"/></figure>'
            f"</section>"
        )

    year_table = effect_by_year_table(result.group_effect, params, by_class=False)
    sections_html = [
        (
            "<section><h2>Топ групп по эффекту</h2>"
            f"{_df_to_html(top_groups_table(result.group_effect, top_n=20))}</section>"
        ),
        _figure_section("top_groups", "Топ групп по эффекту — график"),
        (
            "<section><h2>Эффект по годам</h2>"
            f"{_df_to_html(year_table)}</section>"
        ),
        _figure_section("effect_by_year", "Эффект по годам — график"),
        (
            "<section><h2>Чувствительность к доле сервиса</h2>"
            f"{_df_to_html(sensitivity_table(result.group_effect, params))}</section>"
        ),
        _figure_section("relevance_split", "Распределение групп по классам"),
        _figure_section("baseline_distribution", "Распределение базы эффекта"),
    ]
    body_sections = "\n".join(block for block in sections_html if block)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    delta = result.reconciliation.get("delta", 0.0)

    document = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      /* Палитра САО ВСК: синий / светло-синий (ТЗ), логотип #2D2D81 */
      --vsk-blue: {VSK_BLUE};
      --vsk-blue-dark: {VSK_BLUE_DARK};
      --vsk-blue-mid: {VSK_BLUE_MID};
      --vsk-sky: {VSK_SKY};
      --vsk-sky-soft: {VSK_SKY_SOFT};
      --vsk-accent: {VSK_ACCENT};
      --ink: {VSK_INK};
      --muted: {VSK_MUTED};
      --line: #D5DCE8;
      --bg: #F4F7FC;
      --card: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PT Sans", "Helvetica Neue", Arial, sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, var(--vsk-sky-soft) 0%, var(--bg) 220px);
      line-height: 1.45;
    }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 64px; }}
    .brand {{
      font-size: 12px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--vsk-blue);
      font-weight: 700;
      margin-bottom: 6px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 1.8rem;
      letter-spacing: -0.02em;
      color: var(--vsk-blue-dark);
      border-bottom: 3px solid var(--vsk-blue);
      padding-bottom: 12px;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 1.2rem;
      color: var(--vsk-blue-dark);
      border-left: 4px solid var(--vsk-sky);
      padding-left: 10px;
    }}
    .meta {{ color: #5A6275; margin-bottom: 24px; }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 20px 0 28px;
    }}
    .kpi {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 14px 16px;
      border-top: 3px solid var(--vsk-blue);
    }}
    .kpi .label {{ color: #5A6275; font-size: 0.85rem; }}
    .kpi .value {{
      font-size: 1.35rem;
      font-weight: 650;
      margin-top: 4px;
      color: var(--vsk-blue);
    }}
    .kpi:last-child {{ border-top-color: var(--vsk-accent); }}
    .kpi:last-child .value {{ color: var(--vsk-accent); }}
    section {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 18px 18px 8px;
      margin-bottom: 18px;
      overflow-x: auto;
    }}
    table.data {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
      margin-bottom: 12px;
    }}
    table.data th, table.data td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      white-space: nowrap;
    }}
    table.data th {{
      background: var(--vsk-blue);
      color: #fff;
      font-weight: 600;
    }}
    table.data tr:hover td {{ background: var(--vsk-sky-soft); }}
    section.figure-section {{ padding-bottom: 18px; }}
    figure.full {{ margin: 0 0 4px; }}
    figure.full img {{
      display: block;
      width: 100%;
      max-width: 960px;
      height: auto;
      margin: 0 auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    .muted {{ color: #5A6275; }}
    footer {{ color: #5A6275; font-size: 0.85rem; margin-top: 8px; }}
  </style>
</head>
<body>
  <main>
    <div class="brand">САО ВСК</div>
    <h1>{html.escape(title)}</h1>
    <p class="meta">
      Источник: {html.escape(source_label)}.
      Сформировано: {html.escape(generated_at)}.
      Параметры: доля сервиса {params.service_share:.0%},
      горизонт {params.project_years} г.,
      период {params.first_year}–{params.last_year},
      baseline={html.escape(params.baseline_window)}.
    </p>
    <div class="kpis">
      <div class="kpi"><div class="label">Групп в расчёте</div>
        <div class="value">{kpis["groups"]:.0f}</div></div>
      <div class="kpi"><div class="label">Убыток групп, млн руб.</div>
        <div class="value">{kpis["net_loss_total"] / 1e6:,.1f}</div></div>
      <div class="kpi"><div class="label">Предотвращённый убыток (база), млн руб.</div>
        <div class="value">{kpis["prevented_total"] / 1e6:,.1f}</div></div>
      <div class="kpi"><div class="label">Финансовый эффект, млн руб.</div>
        <div class="value">{kpis["effect_total"] / 1e6:,.1f}</div></div>
    </div>
    {body_sections}
    <footer>
      Сверка матрицы с итогом по группам: расхождение {delta:,.2f} руб.
      Итоговый эффект: {result.total_effect:,.0f} руб.
    </footer>
  </main>
</body>
</html>
"""
    target.write_text(document, encoding="utf-8")
    LOGGER.info("HTML-отчёт сохранён: %s", target)
    return target
