"""Сборка итогового ноутбука: только аналитика поверх пакета ``fin_effects``.

Запуск::

    python scripts/build_notebook.py

Скрипт создаёт ``notebooks/financial_effect.ipynb``. Смысл: код расчётов живёт в
``src/fin_effects`` и покрыт тестами, а ноутбук содержит только витрину —
таблицы, графики и итоговый финансовый эффект.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import nbformat as nbf  # noqa: E402

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "financial_effect.ipynb"


def build_notebook() -> nbf.NotebookNode:
    """Собирает ноутбук с аналитикой.

    Returns:
        Объект ноутбука ``nbformat``.
    """
    notebook = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []
    markdown = lambda text: cells.append(nbf.v4.new_markdown_cell(text))  # noqa: E731
    code = lambda text: cells.append(nbf.v4.new_code_cell(text))  # noqa: E731

    # ------------------------------------------------------------------
    markdown(
        """# Финансовый эффект от сервиса графовой аналитики

Ноутбук содержит **только итоговую аналитику**: проверку исходных данных,
сводные таблицы, графики и итоговый финансовый эффект. Вся логика расчёта
лежит в пакете `src/fin_effects` и покрыта тестами (`pytest`).

## Методика в одном абзаце

Сервис графовой аналитики находит группы связанных лиц, которые совершают
убытки. Для каждой найденной группы считается **средний убыток за активный год**
(база). Если бы сервис группу не нашёл, этот убыток повторялся бы каждый
следующий год. Сервис предотвращает часть этого убытка — она определяется
коэффициентом вклада сервиса (`SERVICE_SHARE`, в исходной тетради `0.35`).

Эффект по группе: `база × доля сервиса × число лет предотвращения`, где число
лет зависит от того, что произошло с группой после обнаружения."""
    )

    # ------------------------------------------------------------------
    markdown("## 1. Настройка окружения")
    code(
        """import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fin_effects.config import (
    DEFAULT_PARAMS,
    FIGURE_DIR,
    REPORT_DIR,
    RELEVANCE_ALIVE,
    RELEVANCE_DIED_AFTER,
    RELEVANCE_DIED_BEFORE,
    EffectParams,
)
from fin_effects.effect import effect_matrix, effect_long, totals_by_relevance
from fin_effects.legacy import (
    LEGACY_SERVICE_SHARE,
    legacy_pipeline_summary,
    legacy_vs_corrected_money,
)
from fin_effects.pipeline import run_pipeline, run_pipeline_from_files
from fin_effects.reporting import (
    build_figures,
    effect_by_cohort_table,
    effect_by_year_table,
    headcount_table,
    sensitivity_table,
    summary_kpis,
    top_groups_table,
)
from fin_effects.synth import SynthConfig, generate_synthetic_data

pd.options.display.float_format = "{:,.1f}".format
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)

print("pandas", pd.__version__)
print("Проект:", PROJECT_ROOT)"""
    )

    # ------------------------------------------------------------------
    markdown(
        """## 2. Параметры расчёта

Все допущения собраны в одном месте и видны в отчёте — в исходной версии это
были «магические» числа, разбросанные по ячейкам.

* `service_share` — **доля вклада (заслуги) сервиса** в предотвращённый убыток;
* `min_losses_total` — минимальное число убытков, чтобы считать группу реальной;
* `min_losses_per_year` — сколько убытков должно быть в году, чтобы считать год активным;
* `project_years` — на сколько лет вперёд проецируется эффект для действующих групп."""
    )
    code(
        """SERVICE_SHARE = 0.35        # доля вклада (заслуги) сервиса
PROJECT_YEARS = 3           # горизонт проецирования эффекта, лет
MIN_LOSSES_TOTAL = 3        # порог «группа, а не разовый убыток»
MIN_LOSSES_PER_YEAR = 3     # порог активности года
FIRST_YEAR = 2020
LAST_YEAR = 2026
BASELINE_WINDOW = "full"    # "full" | "before_detection"

PARAMS = EffectParams(
    service_share=SERVICE_SHARE,
    project_years=PROJECT_YEARS,
    min_losses_total=MIN_LOSSES_TOTAL,
    min_losses_per_year=MIN_LOSSES_PER_YEAR,
    first_year=FIRST_YEAR,
    last_year=LAST_YEAR,
    baseline_window=BASELINE_WINDOW,
)
PARAMS"""
    )

    # ------------------------------------------------------------------
    markdown(
        """## 3. Источник данных

В продовом контуре убытки и карточки УПС берутся из OISUU:

```python
from fin_effects.loaders import load_losses_from_db, load_cards_from_db
losses = load_losses_from_db()   # SQL из fin_effects/queries.py
cards = load_cards_from_db()
```

Выгрузки сервиса (xlsx с участниками групп) читаются функцией
`load_beneficiaries` / `run_pipeline_from_files`.

Здесь, чтобы ноутбук был воспроизводим без доступа к БД, работает
синтетический генератор: он создаёт выгрузки в том же формате, что и продовые."""
    )
    code(
        """SYNTHETIC = True   # False -> читать продовые выгрузки из data/raw

if SYNTHETIC:
    config = SynthConfig(
        seed=20240517,
        groups=120,
        noise_losses=1500,
        year_range=(FIRST_YEAR, LAST_YEAR),
        scenarios={
            "alive": 0.45,
            "died_after_detection": 0.20,
            "died_before_detection": 0.15,
            "no_card": 0.10,
            "single_loss": 0.10,
        },
    )
    data = generate_synthetic_data(config)
    print(data.summary())
    BENEFICIARIES = data.beneficiaries
    LOSSES = data.losses
    CARDS = data.cards
    TRUTH = data.groups
    BENEFICIARY_FILES = data.beneficiary_files
else:
    from fin_effects.excel_beneficiaries import load_beneficiaries_from_dir
    from fin_effects.loaders import load_table

    BENEFICIARIES, _stats = load_beneficiaries_from_dir(
        PROJECT_ROOT / "data" / "raw", years=[LAST_YEAR], workers=4
    )
    LOSSES = load_table(PROJECT_ROOT / "data" / "raw" / "losses.pkl")
    CARDS = load_table(PROJECT_ROOT / "data" / "raw" / "cards.pkl")
    TRUTH = None
    BENEFICIARY_FILES = []

LOSSES.shape, CARDS.shape"""
    )

    # ------------------------------------------------------------------
    markdown(
        """### 3.1. Проверка сырых данных

Контроль качества: сколько строк пришло из БД, сколько из них — дубли по
`LossNumber` (следствие `LEFT JOIN` к платежам и ПСР), нет ли убытков без
идентификаторов участников."""
    )
    code(
        """duplicates = int(LOSSES["LossNumber"].duplicated().sum())
                       
quality = pd.DataFrame(
    [
        {"Показатель": "Строк в выгрузке убытков", "Значение": len(LOSSES)},
        {"Показатель": "Уникальных убытков", "Значение": LOSSES["LossNumber"].nunique()},
        {"Показатель": "Строк-дублей по LossNumber", "Значение": duplicates},
        {"Показатель": "Групп в выгрузке сервиса", "Значение": BENEFICIARIES["group_id"].nunique()},
        {"Показатель": "Участников в выгрузке сервиса", "Значение": BENEFICIARIES["person_key"].nunique()},
        {"Показатель": "Карточек УПС", "Значение": len(CARDS)},
        {"Показатель": "Период убытков", "Значение": f"{LOSSES['LossDateTime'].min():%Y-%m} .. {LOSSES['LossDateTime'].max():%Y-%m}"},
    ]
)
quality"""
    )
    code(
        """COVERAGE = pd.DataFrame(
    {
        "Метрика": ["Убытков с виновником", "Убытков с потерпевшим", "Убытков с VIN"],
        "Доля, %": [
            (LOSSES["Виновник.Полн наименование"].notna().mean() * 100),
            (LOSSES["Потерпевший.Полн наименование"].notna().mean() * 100),
            (LOSSES["Объект виновника.Идентификатор"].notna().mean() * 100),
        ],
    }
)
COVERAGE"""
    )

    # ------------------------------------------------------------------
    markdown(
        """### 3.2. Сверка чтения xlsx с эталоном

Генератор сохраняет исходный список участников, а пайплайн читает те же данные
из xlsx-файлов. Совпадение подтверждает, что парсер выгрузки сервиса не теряет
и не искажает участников."""
    )
    code(
        """if SYNTHETIC:
    from fin_effects.excel_beneficiaries import load_beneficiaries

    parsed, stats = load_beneficiaries(BENEFICIARY_FILES, workers=4)
    expected = set(TRUTH.loc[:, ["group_id", "person_key"]].itertuples(index=False, name=None))
    actual = set(parsed.loc[:, ["group_id", "person_key"]].itertuples(index=False, name=None))

    print(stats.as_text())
    parity = pd.DataFrame(
        [
            {"Проверка": "Листов пропущено", "Значение": stats.sheets_skipped},
            {"Проверка": "Ошибок чтения", "Значение": len(stats.errors)},
            {"Проверка": "Участников прочитано", "Значение": len(actual)},
            {"Проверка": "Участников в эталоне", "Значение": len(expected)},
            {"Проверка": "Потеряно при чтении", "Значение": len(expected - actual)},
            {"Проверка": "Лишних при чтении", "Значение": len(actual - expected)},
        ]
    )
    display(parity)
else:
    print("Продовый режим: сверка с эталоном недоступна")"""
    )

    # ------------------------------------------------------------------
    markdown(
        """## 4. Расчёт пайплайна

Один вызов `run_pipeline` заменяет всю последовательность ячеек исходной
тетради: нормализация → свёртка до уровня убытка → сопоставление с группами →
сводка по группам → расчёт эффекта."""
    )
    code(
        """result = run_pipeline(
    BENEFICIARIES,
    LOSSES,
    CARDS,
    PARAMS,
    persist=True,
)
print(result.as_text())"""
    )
    code(
        """pd.DataFrame(
    [
        {"Этап": "Строк убытков на входе", "Значение": result.loss_stats.rows_in},
        {"Этап": "Убытков-фактов после свёртки", "Значение": result.loss_stats.rows_out},
        {"Этап": "Строк-дублей по LossNumber", "Значение": result.loss_stats.duplicated_loss_numbers},
        {"Этап": "Пар «убыток-группа»", "Значение": len(result.facts_with_groups)},
        {"Этап": "Групп в расчёте", "Значение": len(result.summary)},
        {"Этап": "Групп без карточки УПС (fallback)", "Значение": result.group_stats.groups_without_card},
        {"Этап": "Групп отброшено порогом убытков", "Значение": result.group_stats.groups_dropped_short},
        {"Этап": "Убытков вне периода наблюдения", "Значение": result.group_stats.losses_dropped_by_year},
    ]
)"""
    )

    # ------------------------------------------------------------------
    markdown(
        """## 5. Денежная база: «как было» и «как стало»

Две правки меняют денежную базу эффекта:

1. **Свёртка до уровня убытка.** В выгрузке убыток встречается столько раз,
   сколько у него платежей и строк ПСР (`LEFT JOIN`). Суммировать деньги по
   строкам нельзя — получится завышение.
2. **Возвраты вычитаются.** Взыскано по ФУ и по иску — это деньги, которые
   компания вернула. В исходной тетради они **прибавлялись** к убытку, то есть
   увеличивали «предотвращённый» ущерб.

Ниже — прямое сравнение формул на одних и тех же данных."""
    )
    code(
        """money = legacy_vs_corrected_money(LOSSES)
money_display = money.copy()
for column in ("Сумма, ₽", "Выплаты, ₽", "Возвраты, ₽"):
    money_display[column] = money_display[column] / 1e6
money_display = money_display.rename(
    columns={
        "Сумма, ₽": "Сумма, млн ₽",
        "Выплаты, ₽": "Выплаты, млн ₽",
        "Возвраты, ₽": "Возвраты, млн ₽",
    }
)
money_display"""
    )

    # ------------------------------------------------------------------
    markdown(
        """### 5.1. Итоговый эффект: исходная логика против исправленной

Исходная логика дополнительно страдала тем, что:

* год обнаружения группы был **жёстко равен 2024** (`losses_['detection_date'] = 2024`);
* группам с признаком «распалась самостоятельно» класс **переприсваивался
  случайно** (`np.random.choice` с вероятностью 0.5), то есть часть эффекта
  создавалась генератором случайных чисел;
* в pivot эффект по году усреднялся (`aggfunc="mean"`), а затем суммировался —
  это занижает результат пропорционально числу групп."""
    )
    code(
        """raw_beneficiaries = pd.DataFrame(
    {
        0: BENEFICIARIES["full_name"].tolist(),
        1: BENEFICIARIES["birth_date"].astype(str).tolist(),
        2: BENEFICIARIES["group_number"].astype(str).tolist(),
    }
)
legacy_table = legacy_pipeline_summary(LOSSES, raw_beneficiaries, CARDS)
legacy_table"""
    )
    code(
        """comparison = pd.DataFrame(
    [
        {
            "Вариант расчёта": "Исходная логика тетради (доля сервиса 0.35)",
            "Эффект, млн ₽": legacy_table.loc[0, "Эффект, ₽"] / 1e6,
        },
        {
            "Вариант расчёта": "Исправленная методика (доля сервиса 0.35)",
            "Эффект, млн ₽": result.total_effect / 1e6,
        },
    ]
)
base = comparison["Эффект, млн ₽"].iloc[1]
comparison["Отношение к исправленной"] = comparison["Эффект, млн ₽"] / base
comparison"""
    )

    # ------------------------------------------------------------------
    markdown(
        """## 6. Портфель групп

Класс релевантности определяет, даёт группа эффект или нет:

* **Продолжает действовать** — убытки идут и после обнаружения: сервис
  предотвращает базу в каждом году горизонта;
* **Распалась после обнаружения** — убытки прекратились в год обнаружения:
  предотвращается один годовой убыток;
* **Распалась самостоятельно** — группа исчезла до обнаружения: эффект нулевой."""
    )
    code(
        """headcount_table(result.summary)"""
    )
    code(
        """summary_view = result.summary.copy()
summary_view["active_years"] = summary_view["active_years"].map(
    lambda years: ", ".join(str(year) for year in years) if len(years) else "—"
)
summary_view.loc[
    :,
    [
        "group_id", "group_relevance", "detection_year", "detection_source",
        "first_loss_year", "last_loss_year", "active_years",
        "losses_total", "net_loss_total", "net_loss_active", "members_matched",
    ],
].head(20)"""
    )
    code(
        """build_figures(result.group_effect, result.summary, PARAMS, FIGURE_DIR)"""
    )
    code(
        """from IPython.display import Image, display

display(Image(filename=str(FIGURE_DIR / "relevance_split.png")))
display(Image(filename=str(FIGURE_DIR / "baseline_distribution.png")))"""
    )

    # ------------------------------------------------------------------
    markdown(
        """## 7. Финансовый эффект

### 7.1. Эффект по группам

`mean_loss_per_active_year` — база (средний убыток за активный год),
`effect_first_year` — эффект года обнаружения, `effect_total` — эффект за весь
горизонт (для действующих групп — `project_years` лет, для остальных — один год)."""
    )
    code(
        """effect_view = result.group_effect.copy()
for column in ("net_loss_total", "mean_loss_per_active_year", "effect_first_year", "effect_total"):
    effect_view[column] = effect_view[column] / 1e6
effect_view = effect_view.rename(
    columns={
        "mean_loss_per_active_year": "База в год, млн ₽",
        "net_loss_total": "Убыток группы, млн ₽",
        "effect_first_year": "Эффект года обнаружения, млн ₽",
        "effect_total": "Эффект всего, млн ₽",
    }
)
effect_view.sort_values("Эффект всего, млн ₽", ascending=False).head(25)"""
    )

    markdown("### 7.2. Эффект по годам")
    code(
        """effect_by_year_table(result.group_effect, PARAMS)"""
    )
    code(
        """display(Image(filename=str(FIGURE_DIR / "effect_by_year.png")))"""
    )

    markdown(
        """### 7.3. Эффект по когортам обнаружения

Когорта — год, в котором сервис обнаружил группу. Эффект когорты нельзя
сравнивать с эффектом года: обнаруженная в 2025 году группа даёт эффект и в
2026-м."""
    )
    code(
        """effect_by_cohort_table(result.group_effect, PARAMS)"""
    )
    code(
        """display(Image(filename=str(FIGURE_DIR / "effect_by_cohort.png")))"""
    )

    markdown("### 7.4. Топ групп по эффекту")
    code(
        """top_groups_table(result.group_effect, top_n=20)"""
    )
    code(
        """display(Image(filename=str(FIGURE_DIR / "top_groups.png")))"""
    )

    # ------------------------------------------------------------------
    markdown("## 8. Итоговый финансовый эффект")
    code(
        """KPIS = summary_kpis(result.group_effect, PARAMS)

prevented = KPIS["prevented_total"]
effect = KPIS["effect_total"]

final = pd.DataFrame(
    [
        {"Показатель": "Групп в расчёте", "Значение": f"{KPIS['groups']:.0f}"},
        {"Показатель": "Убыток групп за период, млн ₽", "Значение": f"{KPIS['net_loss_total'] / 1e6:,.1f}"},
        {"Показатель": "Предотвращённый убыток (база), млн ₽", "Значение": f"{prevented / 1e6:,.1f}"},
        {"Показатель": "Доля вклада сервиса", "Значение": f"{PARAMS.service_share:.0%}"},
        {
            "Показатель": f"Финансовый эффект сервиса, млн ₽ (горизонт {PARAMS.project_years} года)",
            "Значение": f"{effect / 1e6:,.1f}",
        },
    ]
)
final"""
    )
    code(
        """relevance_totals = totals_by_relevance(result.group_effect).copy()
relevance_totals["prevented_loss_total"] = relevance_totals["prevented_loss_total"] / 1e6
relevance_totals["effect_total"] = relevance_totals["effect_total"] / 1e6
relevance_totals = relevance_totals.rename(
    columns={
        "group_relevance": "Класс группы",
        "groups": "Групп",
        "net_loss_total": "Убыток групп, млн ₽",
        "prevented_loss_total": "Предотвращённый убыток, млн ₽",
        "effect_total": "Эффект, млн ₽",
        "share_of_effect": "Доля эффекта",
    }
)
relevance_totals["Доля эффекта"] = relevance_totals["Доля эффекта"].map("{:.1%}".format)
relevance_totals"""
    )
    code(
        """matrix = result.effect_matrix.copy()
matrix_display = matrix / 1e6
matrix_display.columns = [f"{column}, млн ₽" if column != "Итого" else "Итого, млн ₽" for column in matrix_display.columns]
matrix_display.index.name = "Год"
matrix_display"""
    )
    code(
        """print(f"Сверка матрицы эффекта с итогом по группам: расхождение {result.reconciliation['delta']:,.2f} ₽")"""
    )

    # ------------------------------------------------------------------
    markdown(
        """## 9. Чувствительность к доле вклада сервиса

`0.35` — договорная величина. Ниже — как меняется эффект при других значениях:
это готовый аргумент для торга с бизнесом о том, какая доля заслуги сервиса
признаётся корректной."""
    )
    code(
        """sensitivity_table(result.group_effect, PARAMS, shares=(0.2, 0.3, 0.35, 0.5, 0.7))"""
    )

    code(
        """def effect_for_share(share: float) -> float:
    \"\"\"Пересчитывает эффект для другой доли вклада сервиса.

    Args:
        share: новая доля вклада сервиса.

    Returns:
        Итоговый эффект, ₽.
    \"\"\"
    alternative = run_pipeline(
        BENEFICIARIES, LOSSES, CARDS, replace(PARAMS, service_share=share)
    )
    return alternative.total_effect


for share in (0.2, 0.35, 0.5):
    print(f"Доля сервиса {share:.0%}: эффект {effect_for_share(share) / 1e6:,.1f} млн ₽")"""
    )

    # ------------------------------------------------------------------
    markdown(
        """## 10. Выгрузка результатов"""
    )
    code(
        """from fin_effects.io_utils import write_excel_report

artifacts = {
    "KPI": final,
    "Классы групп": relevance_totals,
    "Эффект по годам": effect_by_year_table(result.group_effect, PARAMS),
    "Эффект по когортам": effect_by_cohort_table(result.group_effect, PARAMS),
    "Эффект по группам": effect_view,
    "Топ групп": top_groups_table(result.group_effect, top_n=50),
    "Денежная база было-стало": money_display,
    "Исходная логика": legacy_table,
}
report_file = write_excel_report(artifacts, REPORT_DIR / "financial_effect.xlsx")
print("Отчёт сохранён:", report_file)

result.group_effect.to_parquet(REPORT_DIR / "group_effect.parquet", index=False)
print("Таблица эффекта по группам:", REPORT_DIR / "group_effect.parquet")"""
    )

    # ------------------------------------------------------------------
    markdown(
        """## 11. Допущения и ограничения

Их полезно проговорить в записке к расчёту — каждая завышает или занижает эффект.

1. **База считается по всей наблюдаемой истории группы** (`baseline_window="full"`).
   Если считать только по годам до обнаружения (`"before_detection"`), база
   меняется: для групп, которые после обнаружения «разогнались», она ниже.
2. **Сопоставление убытка с группой идёт по ФИО + дате рождения.** Разное
   написание ФИО, смена фамилии, опечатка в дате — группа не соберётся.
   Дополнительно: если один убыток связывает участников двух разных групп, его
   набор идентификаторов не совпадёт ни с одной из них.
3. **Горизонт проецирования** (по умолчанию 3 года) — управляемое допущение:
   чем он больше, тем больше эффект. В исходной тетради горизонт фактически
   складывался из двух захардкоженных годов.
4. **Доля вклада сервиса** (`0.35`) — не измеряемая из данных величина; см.
   блок чувствительности.
5. **Контрфактическое допущение «убыток повторялся бы каждый год»** — самая
   сильная гипотеза методики. Более строгая версия — сравнение с контрольной
   группой похожих клиентов без признаков связи.
6. **Взыскания** (по ФУ и по иску) вычитаются из базы. Если по практике часть
   взысканий приходит уже после обнаружения группы, база немного занижена.
7. **Отсечение периода** (`first_year`/`last_year`) применяется к дате убытка,
   а не к дате обнаружения."""
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
    """Точка входа.

    Returns:
        Код возврата.
    """
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    nbf.write(notebook, str(NOTEBOOK_PATH))
    code_cells = sum(1 for cell in notebook.cells if cell.cell_type == "code")
    markdown_cells = sum(1 for cell in notebook.cells if cell.cell_type == "markdown")
    print(f"Ноутбук собран: {NOTEBOOK_PATH}")
    print(f"Ячеек: {len(notebook.cells)} (код {code_cells}, markdown {markdown_cells})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
