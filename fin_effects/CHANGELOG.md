# Changelog

## 2026-10-05

- Что: классы групп — «в работе» / «закрыта» по стадии карточки УПС; без карточки группа вне эффекта; карточка по 1 убытку расширяется на всю группу через ФИО+ДР.
- Зачем: одна карточка ≠ один убыток; эффект только по группам, которыми реально занимаются.
- Проверка: `python scripts/build_html_report.py` → `outputs/reports/financial_effect.html`.

- Что: prod-матч убытков к группам: при `loss_number` в Excel — только по номеру (без union с ФИО); компактный ноутбук + `write_html_report` в ячейке; KPI `pipeline_overview_table`.
- Зачем: union с `person_key` тянул сотни тысяч убытков КАСКО → млрд эффекта; HTML не вызывался из ноутбука.
- Проверка: перезапуск прод-ноутбука — пары «убыток–группа» сильно меньше; `outputs/reports/financial_effect.html`.

- Что: ускорение: prod Excel (`iter_rows`, быстрый цвет, стоп на пустых); кэш `beneficiaries.parquet`; `EXCEL_WORKERS` по умолчанию; synth `write_workbooks=False` в скриптах/ноутбуке (без сотен xlsx).
- Зачем: повторные прогоны не должны минутами читать 100+ xlsx/SQL и писать синтетические книги.
- Проверка: `run_synthetic` / HTML без xlsx; prod — `REFRESH_EXCEL/SQL=True` один раз, дальше parquet.

- Что: `load_losses` / `load_cards` читают parquet-кэш (`data/interim/losses.parquet`, `cards.parquet`), SQL — только при отсутствии файла или `refresh=True`.
- Зачем: не тянуть ~1.5M строк losses из OISUU на каждый прогон ноутбука.
- Проверка: первый `load_losses()` пишет кэш; второй — из parquet без SQL.

- Что: prod-парсер Excel (`excel_loss_groups`): листы с колонкой «Убыток» (нормализация заголовков), цветные строки (заливка/шрифт), 1 файл = 1 группа; матч убытков по `loss_number` + `person_key`.
- Зачем: боевые выгрузки — широкие таблицы убытков, а не 3 колонки ФИО/ДР; старый парсер почти ничего не матчил.
- Проверка: `pytest tests/test_excel_loss_groups.py`; на сервере `SYNTHETIC=False` — рост «Пар убыток-группа».

- Что: в `legacy_detection_and_relevance` `LossNumber` нормализуется до merge с карточками.
- Зачем: из SQL ключ приходит как `float64`, после `normalize_series` у карточек — `object`, pandas падал на merge.
- Проверка: ячейка legacy-сравнения при `SYNTHETIC = False`.

- Что: из `LOSSES_QUERY` убран JOIN к `oisuu81_t_IncidentCharacteristics` / колонка `IncidentNumber`.
- Зачем: на текущем OISUU объекта нет (`Invalid object name`), а в пайплайне эффекта `IncidentNumber` не используется.
- Проверка: `load_losses_from_db()` при `SYNTHETIC = False`.

- Что: в legacy-сравнении ноутбука колонка группы берётся как `group_id` (а не `group_number`).
- Зачем: в продовом Excel после `load_beneficiaries` есть только `group_id`; `group_number` есть лишь у синтетики.
- Проверка: ячейка «исходная логика против исправленной» при `SYNTHETIC = False`.

- Что: добавлен `env_template`; `DbConfig` читает `.env`; ODBC для OISUU перебирает `.so` на диске и `pyodbc.drivers()` (как в monitoring: Driver 18 / `libmsodbcsql-18.so`).
- Зачем: на Jupyter нет Driver 17, из‑за этого падал `load_losses_from_db`.
- Проверка: `cp env_template .env`, заполнить UID/PWD, при необходимости `OISUU_ODBC_DRIVER=/usr/lib64/libmsodbcsql-18.so`, restart kernel, `SYNTHETIC = False`.

- Что: продовая загрузка в ноутбуке универсальна — Excel из всех `data/raw/<YYYY>/`, `losses`/`cards` из SQL (`load_losses_from_db` / `load_cards_from_db`); `discover_workbooks` сам находит годовые каталоги; `read_sql` склеивает чанки в DataFrame.
- Зачем: не требовать ручной `losses.pkl` и не жёстко привязываться к одному году вроде 2026.
- Проверка: `SYNTHETIC = False`, положить xlsx в `data/raw/2024` (и/или другие годы), задать `OISUU_UID`/`OISUU_PWD`, выполнить ячейку источника данных.

- Что: добавлен HTML-отчёт по финансовому эффекту (`write_html_report`, скрипт `scripts/build_html_report.py`).
- Зачем: быстрый самодостаточный отчёт после прогона на синтетике (таблицы + встроенные графики).
- Проверка: `python scripts/build_html_report.py` → `outputs/reports/financial_effect.html`.

- Что: графики и HTML переведены на палитру брендбука САО ВСК (`#2D2D81`, `#6BA3D9`, акцент `#E0922F`).
- Зачем: единый визуальный стиль с корпоративными отчётами.
- Проверка: пересобрать `python scripts/build_html_report.py` и открыть HTML.

- Что: структура HTML-отчёта упрощена: топ групп в начале, эффект по годам без разреза по классам, без когорт/классов; графики — каждый в своей секции.
- Зачем: убрать дублирующие срезы, сделать отчёт читаемее.
- Проверка: `python scripts/build_html_report.py`.

## 2026-10-01

- Что: рефакторинг расчёта финансового эффекта из `by_group_set_2-Copy1 (1).ipynb` в пакет `src/fin_effects`; ноутбук `notebooks/financial_effect.ipynb` оставлен только с аналитикой; добавлены синтетические данные, тесты, бенчмарк и сравнение с legacy-логикой.
- Зачем: воспроизводимый и быстрый пайплайн вместо разрозненных ячеек с магическими константами и циклами.
- Проверка: `python -m pytest`, `python scripts/run_synthetic.py`, `python scripts/benchmark.py`.

### Исправления при доводке

- `effect_by_cohort`: merge по году обнаружения больше не размножает строки, если в когорте несколько групп; `effect_year_of_detection` считается по релевантным классам с долей сервиса.
- `effect_total` усекается `last_year`, чтобы сходиться с матрицей эффекта.
- `total_prevented_loss` считает только группы, дающие эффект.
- `min_years_for_baseline` реально применяется в фильтре групп.
- `attach_groups_to_losses` переведён на `melt` вместо цикла + `concat`.
- Логи/скрипты на Windows: символ `₽` заменён на `руб.`, чтобы не падать на cp1251.
