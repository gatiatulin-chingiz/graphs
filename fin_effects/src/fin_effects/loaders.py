"""Загрузка таблиц ``losses`` / ``cards`` — из БД OISUU или из файлов.

Ноутбук не должен уметь ходить в продовую БД: расчёты берут данные из
``data/interim`` (артефакты) или из ``data/raw`` (выгрузки). Работа с БД живёт в
``loaders.py`` и вызывается отдельным явным шагом обновления данных.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from . import queries
from .config import DbConfig
from .io_utils import read_pickle, write_pickle

LOGGER = logging.getLogger(__name__)

#: Колонка-ключ убытка.
LOSS_KEY = "LossNumber"

#: Колонка-ключ карточки УПС.
CARD_LOSS_KEY = "Номер Убытка"

READABLE_SUFFIXES = (".pkl", ".pickle", ".parquet", ".csv", ".xlsx", ".xls")


def _require_pyodbc() -> Any:
    """Импортирует ``pyodbc`` с понятной ошибкой.

    Returns:
        Модуль ``pyodbc``.

    Raises:
        ImportError: если драйвер/пакет не установлен.
    """
    try:
        import pyodbc  # noqa: PLC0415 - опциональная зависимость
    except ImportError as error:  # pragma: no cover - зависит от окружения
        raise ImportError(
            "Для чтения из OISUU установите pyodbc и ODBC Driver 17 for SQL Server "
            "(pip install pyodbc)"
        ) from error
    return pyodbc


def read_sql(
    query: str,
    db_config: DbConfig | None = None,
    *,
    database: str | None = None,
    chunksize: int | None = 200_000,
) -> pd.DataFrame:
    """Выполняет SQL-запрос и возвращает ``DataFrame``.

    Args:
        query: текст запроса.
        db_config: параметры подключения.
        database: переопределение БД.
        chunksize: размер батча; ``None`` — читать одним куском.

    Returns:
        Результат запроса.
    """
    pyodbc = _require_pyodbc()
    config = db_config or DbConfig()
    connection_string = config.connection_string(database)
    LOGGER.info("Чтение из OISUU: %s (database=%s)", config.server, database or config.database)

    with pyodbc.connect(connection_string) as connection:
        return pd.read_sql_query(query, connection, chunksize=chunksize)


def load_losses_from_db(db_config: DbConfig | None = None) -> pd.DataFrame:
    """Выгружает убытки (КАСКО) из OISUU.

    Args:
        db_config: параметры подключения.

    Returns:
        Таблица убытков «как есть» (без нормализации).
    """
    return read_sql(queries.LOSSES_QUERY, db_config, database="OISUU_report")


def load_cards_from_db(db_config: DbConfig | None = None) -> pd.DataFrame:
    """Выгружает карточки УПС из OISUU.

    Args:
        db_config: параметры подключения.

    Returns:
        Таблица карточек УПС.
    """
    return read_sql(queries.CARDS_QUERY, db_config, database="oisuu81")


def read_table(path: str | Path) -> pd.DataFrame:
    """Читает таблицу из файла по расширению.

    Args:
        path: путь к ``.pkl``/``.parquet``/``.csv``/``.xlsx``.

    Returns:
        Таблица.

    Raises:
        ValueError: если расширение не поддерживается.
    """
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        obj = read_pickle(resolved)
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(f"В {resolved} лежит не DataFrame, а {type(obj).__name__}")
        return obj
    if suffix == ".parquet":
        return pd.read_parquet(resolved)
    if suffix == ".csv":
        return pd.read_csv(resolved)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(resolved)
    raise ValueError(f"Неподдерживаемое расширение: {resolved.suffix}")


def load_table(
    source: str | Path | pd.DataFrame,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Загружает таблицу из файла/``DataFrame`` с опциональным кэшем.

    Args:
        source: путь к файлу, либо уже готовая таблица.
        cache_path: куда положить pickle (только если читали из файла).

    Returns:
        Таблица.

    Note:
        Заменяет пары ячеек тетради «прочитать X из БД -> сохранить pickle ->
        прочитать pickle». Теперь источник данных задаётся одним аргументом.
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()

    frame = read_table(source)
    if cache_path is not None:
        saved = write_pickle(frame, cache_path)
        LOGGER.info("Артефакт сохранён: %s", saved)
    return frame


def make_synthetic_source(frame: pd.DataFrame) -> pd.DataFrame:
    """Возвращает таблицу-заглушку «источник данных» для синтетического прогона.

    Args:
        frame: синтетическая таблица.

    Returns:
        Та же таблица (копия).
    """
    return frame.copy()


def find_source_file(directory: str | Path, stems: Iterable[str]) -> Path | None:
    """Ищет первый существующий файл по списку имён (без расширения).

    Args:
        directory: каталог поиска.
        stems: имена файлов без расширения.

    Returns:
        Путь к файлу или ``None``.
    """
    base = Path(directory)
    if not base.exists():
        return None
    for stem in stems:
        for suffix in READABLE_SUFFIXES:
            candidate = base / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None
