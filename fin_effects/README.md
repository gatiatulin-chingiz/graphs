# fin-effects

Расчёт финансового эффекта от сервиса графовой аналитики (поиск мошеннических групп).

## Быстрый старт

```bash
pip install -e ".[dev]"
python scripts/run_synthetic.py --groups 48
python -m pytest
```

Аналитика — в `notebooks/financial_effect.ipynb`. Логика — в `src/fin_effects`.

## Структура

- `src/fin_effects/` — пайплайн (препроцессинг, группы, эффект, отчёты, синтетика)
- `notebooks/financial_effect.ipynb` — таблицы, графики, итоговый эффект
- `scripts/run_synthetic.py` — дымовой прогон
- `scripts/benchmark.py` — замер производительности
- `data/synthetic/` — синтетические выгрузки (создаются генератором)

Подключение к OISUU опционально: `pip install -e ".[db]"`.
