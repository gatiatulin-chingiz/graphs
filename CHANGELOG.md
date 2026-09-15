# Changelog

## 2026-09-15

- Витрина/HTML: Top-N по `score` **внутри каждого типа** (`fraud_top_n_per_type`); убраны дубли `viz_top_n` / `fraud_viz_top_n`.
- Runner: `view, bundle = show_fraud_candidates()` → `visualize_fraud()`; полный список в `bundle['ranked']`.

## 2026-09-14

- Fraud-пайплайн (`src/fraud.py`): аудит хабов → whitelist/NaN exclude → кластеры от suspect seeds → скоринг/типы (колотуны, битки, соло, юрлицо-хаб) → HTML в `./output/html/fraud/`.
- `hub_keywords`: матч **целого слова/фразы** (регистр не важен); каталог ОПФ/ИП + бренды.
- Runner: `show_hub_audit` → `show_fraud_candidates` → `visualize_fraud`.
- Удалён legacy-срез хабов+соседей (`collect_cut_nodes` / `apply_group_cuts` / `log_cut_preview` / `show_degree_report` / старый Gephi-HTML viz); убраны `cluster_mode`, `hub_degree_n`. `visualize()` → alias на `visualize_fraud`.
- Ускорение fraud: кэш кластеров/ranked между ячейками; compile `hub_keywords`; предрасчёт meta узлов; индекс links группы + `nx.k_core(2)`; `quick_fraud_slices` вынесен в `show_quick_slices()` (не в каждом `show_fraud_candidates`).

## 2026-09-11

- HTML: подписи ФИО/VIN **внутри** узлов (`ellipse`), не сбоку от точки.
- Viz: короткие `label` + полные `title` (hover); Gephi `weight` ≥ 1.
- Кэш: если Excel/SQL (и `bound`) не менялись — загрузка из `./vars`; иначе полный пересчёт.
- Пайплайн без GUI: `run_pipeline()`; логи шапка/шаги; пакет `src/`.
- Выгрузка `./output/statistics.xlsx`; ускорение `create_links` / `create_statistics`.
