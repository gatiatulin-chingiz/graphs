"""Загрузка данных из SQL и Excel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pymssql
from tqdm import tqdm

from graph_analytics.config import Settings


class DataLoader:
    """Чтение убытков из файлов в рабочей директории."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def load_sql(self, directory: Path | None = None) -> pd.DataFrame:
        """Выполнить все .sql в каталоге и объединить результаты."""
        self._settings.validate_db()
        root = directory or self._settings.input_dir
        sql_files = sorted(root.glob("*.sql"))
        if not sql_files:
            raise FileNotFoundError(f"В каталоге {root} нет файлов .sql")

        frames: list[pd.DataFrame] = []
        for path in tqdm(sql_files, desc="SQL-запросы"):
            query = path.read_text(encoding="utf-8")
            with pymssql.connect(
                server=self._settings.db_server,
                user=self._settings.db_user,
                password=self._settings.db_password,
                database=self._settings.db_name,
                port=self._settings.db_port,
            ) as connection:
                frames.append(pd.read_sql_query(query, connection))

        return pd.concat(frames, axis=0, ignore_index=True)

    def load_excel(self, directory: Path | None = None) -> pd.DataFrame:
        """Объединить все .xlsx в каталоге."""
        root = directory or self._settings.input_dir
        excel_files = sorted(root.glob("*.xlsx"))
        if not excel_files:
            raise FileNotFoundError(f"В каталоге {root} нет файлов .xlsx")

        frames = [pd.read_excel(path) for path in tqdm(excel_files, desc="Excel")]
        return pd.concat(frames, axis=0, ignore_index=True)
