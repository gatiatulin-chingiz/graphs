"""Загрузка таблиц ``losses`` / ``cards`` — из БД OISUU или из файлов.

В продовом контуре убытки и карточки УПС читаются SQL-запросами
(:func:`load_losses_from_db`, :func:`load_cards_from_db`). Файловый путь
(:func:`load_table`) нужен для кэша/синтетики, а не как обязательный ручной шаг.
"""

from __future__ import annotations

import glob
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from . import queries
from .config import DbConfig, load_dotenv
from .io_utils import read_pickle, write_pickle

LOGGER = logging.getLogger(__name__)

#: Колонка-ключ убытка.
LOSS_KEY = "LossNumber"

#: Колонка-ключ карточки УПС.
CARD_LOSS_KEY = "Номер Убытка"

READABLE_SUFFIXES = (".pkl", ".pickle", ".parquet", ".csv", ".xlsx", ".xls")

#: Типичные пути к libmsodbcsql на Jupyter / RHEL (как в monitoring).
_SO_GLOBS = (
    "/usr/lib64/libmsodbcsql*.so",
    "/usr/lib/libmsodbcsql*.so",
    "/opt/microsoft/msodbcsql*/lib64/libmsodbcsql*.so",
    "/opt/microsoft/msodbcsql*/lib64/libmsodbcsql-*.so*",
)


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
            "Для чтения из OISUU установите pyodbc "
            "(pip install pyodbc) и ODBC-драйвер SQL Server / libmsodbcsql"
        ) from error
    return pyodbc


def list_available_odbc_drivers() -> list[str]:
    """Возвращает драйверы из ``pyodbc.drivers()``."""
    try:
        pyodbc = _require_pyodbc()
        return list(pyodbc.drivers())
    except Exception:  # pragma: no cover - зависит от окружения
        return []


def find_msodbcsql_libraries() -> list[str]:
    """Ищет реальные ``libmsodbcsql*.so`` на диске (обход битого odbcinst)."""
    found: list[str] = []
    for pattern in _SO_GLOBS:
        for path in sorted(glob.glob(pattern)):
            if path not in found and Path(path).is_file():
                found.append(path)
    return found


def candidate_odbc_drivers(explicit: str | None = None) -> list[str]:
    """Собирает список кандидатов: явный → `.so` на диске → SQL Server из pyodbc."""
    drivers: list[str] = []

    def _add(name: str | None) -> None:
        if name and name not in drivers:
            drivers.append(name)

    _add((explicit or "").strip() or None)
    for so_path in find_msodbcsql_libraries():
        _add(so_path)
    for registered in list_available_odbc_drivers():
        low = registered.lower()
        if any(token in low for token in ("sql server", "msodbcsql", "freetds", "tds")):
            _add(registered)
    return drivers


def connect_oisuu(
    db_config: DbConfig | None = None,
    *,
    database: str | None = None,
) -> Any:
    """Открывает соединение с OISUU, перебирая рабочие ODBC-драйверы.

    Args:
        db_config: параметры подключения.
        database: переопределение БД.

    Returns:
        Живое ``pyodbc``-соединение (вызывающий закрывает сам).

    Raises:
        RuntimeError: если ни один кандидат не подключился.
    """
    load_dotenv()
    pyodbc = _require_pyodbc()
    config = db_config or DbConfig()
    available = list_available_odbc_drivers()
    libs = find_msodbcsql_libraries()
    candidates = candidate_odbc_drivers(config.driver)
    LOGGER.info(
        "OISUU ODBC: drivers=%s libs=%s candidates=%s",
        available,
        libs,
        candidates,
    )
    if not candidates:
        raise RuntimeError(
            "Не найден ODBC-драйвер SQL Server.\n"
            f"pyodbc.drivers() = {available}\n"
            f"libmsodbcsql*.so = {libs}\n"
            "Задай в .env (см. env_template):\n"
            "  OISUU_ODBC_DRIVER=/usr/lib64/libmsodbcsql-18.so"
        )

    last_error: Exception | None = None
    tried: list[str] = []
    for driver in candidates:
        tried.append(driver)
        conn_str = config.connection_string(database, driver=driver)
        try:
            connection = pyodbc.connect(conn_str)
            LOGGER.info("OISUU ODBC OK: %s", driver)
            return connection
        except Exception as error:  # pragma: no cover - зависит от окружения
            last_error = error
            LOGGER.warning("OISUU ODBC fail (%s): %s", driver, error)

    raise RuntimeError(
        "Не удалось подключиться к OISUU.\n"
        f"pyodbc.drivers() = {available}\n"
        f"libmsodbcsql on disk = {libs}\n"
        f"Пробовали: {tried}\n"
        "Если Driver 18 в списке, но 'file not found' — укажи реальный .so в "
        "OISUU_ODBC_DRIVER (см. env_template).\n"
        f"Последняя ошибка: {last_error}"
    ) from last_error


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
    load_dotenv()
    config = db_config or DbConfig()
    LOGGER.info("Чтение из OISUU: %s (database=%s)", config.server, database or config.database)

    with connect_oisuu(config, database=database) as connection:
        if chunksize is None:
            return pd.read_sql_query(query, connection)
        chunks = list(pd.read_sql_query(query, connection, chunksize=chunksize))
        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks, ignore_index=True)


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
