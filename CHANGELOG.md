# Changelog

## 2026-09-14

- Fraud-пайплайн (`src/fraud.py`): аудит хабов → whitelist/NaN exclude → кластеры от suspect seeds → скоринг/типы (колотуны, битки, соло, юрлицо-хаб) → HTML в `./output/html/fraud/`.
- `hub_keywords`: матч **целого слова/фразы** (регистр не важен); каталог ОПФ/ИП + бренды; `cluster_mode='fraud'`.
- Runner: `show_hub_audit` → `show_fraud_candidates` → `visualize_fraud`.

## 2026-09-11

- Срез хабов: `hub_keywords` среди топ-хабов; удаляется хаб и соседи (legacy при `cluster_mode='legacy'`).
- Отчёт: топ-хабов только **после** среза; `legal_forms_remove` заменён на `hub_keywords`.
- Пайплайн: `run_pipeline()` по умолчанию **без** отрисовки → сначала `show_degree_report()`, затем `visualize()`.
- HTML на диск только для компонент размером ≥ `bound`.
- HTML: подписи ФИО/VIN **внутри** узлов (`ellipse`), не сбоку от точки.
- Viz: перед HTML/Gephi применяется срез юрлиц/хабов из config (статистика Excel без изменений).
- Viz: короткие `label` (ФИО→«Фамилия И.О. ДР», рёбра→до 3 номеров убытков) + полные `title` (hover в HTML); то же для Gephi (Label).
- Gephi: `weight` всегда ≥ 1 (раньше 0 → рёбра игнорировались).
- Кэш: если Excel/SQL (и `bound`) не менялись — загрузка артефактов из `./vars` с уведомлением; иначе полный пересчёт и новый `input_fingerprint.json`.
- Пайплайн без кнопок/GUI: `run()` сразу считает или поднимает кэш, затем визуализация.
- Удалён неиспользуемый код: ipywidgets/кнопки, sociohub, matplotlib/requests, дубли импортов, закомментированные ветки GUI.
- `config.viz_top_n` — сколько крупнейших групп выгружать в HTML/Gephi (по умолчанию 200).
- Логи: шапка, шаги ▸, детали ·, успех ✓ / ошибки ✗.
- Папка `html/group_N/` только если у группы >1 компоненты; иначе один файл `Group_visualisation{N}.html`.
- В тетрадке: `show_degree_report(0)` — степени/хабы/сценарии среза для выбора N.
- Gephi: полные графы в `./output/gephi/` (без разбиения на компоненты).
- Исправлен запуск в Jupyter: пакет `src/`, вызов `run_pipeline()` вместо `run()` (конфликт с IPython `%run`).
- Перед `run_pipeline()` создаются `./vars`, `./output`, `./output/html`, `./output/gephi`; старые HTML/Gephi удаляются.
- Визуализация: без physics/menu; title рёбер ≤ 3 убытков.
- Выгрузка только `./output/statistics.xlsx`; колонки «ФИО участников группы» и «Номера убытков».
- Ускорение `create_links` / `create_statistics` / индексы для viz; параллельное чтение Excel.
- Проверка: повторный `run()` без смены файлов → сообщение про кэш; в `output/gephi` появляются `.gexf`.
