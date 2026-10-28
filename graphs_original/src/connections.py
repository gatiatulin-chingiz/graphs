"""Параметры подключений к БД (не бизнес-настройки пайплайна)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Сначала .env рядом с graphs_original, затем cwd (Jupyter / сервер).
load_dotenv(_PROJECT_ROOT / '.env')
load_dotenv()

oisuu_conn = {
    'host': os.getenv('OISUU_CONN_HOST', default='oisuu-db-biz.vsk.ru'),
    'username': os.getenv('OISUU_CONN_USERNAME'),
    'password': os.getenv('OISUU_CONN_PASSWORD'),
    'database': os.getenv('OISUU_CONN_DATABASE', default='OISUU_report'),
    'driver': os.getenv('OISUU_ODBC_DRIVER', default=''),
}

actuar_conn = {
    'host': os.getenv('ACTUAR_CONN_HOST', default='actuar2.vsk.ru'),
    'username': os.getenv('ACTUAR_CONN_USERNAME'),
    'password': os.getenv('ACTUAR_CONN_PASSWORD'),
    'database': os.getenv('ACTUAR_CONN_DATABASE', default='Motor'),
}


def _require_pyodbc() -> Any:
    """Импортирует pyodbc с понятной ошибкой."""
    try:
        import pyodbc  # noqa: PLC0415
    except ImportError as error:
        raise ImportError(
            'Для выгрузки из SQL установите pyodbc: pip install pyodbc'
        ) from error
    return pyodbc


def _odbc_driver_candidates(explicit: str | None = None) -> list[str]:
    """Кандидаты ODBC-драйвера SQL Server."""
    pyodbc = _require_pyodbc()
    candidates: list[str] = []
    preferred = (explicit or oisuu_conn.get('driver') or '').strip()
    if preferred:
        candidates.append(preferred)
    for name in pyodbc.drivers():
        lower = name.lower()
        if 'sql server' in lower or 'msodbcsql' in lower:
            if name not in candidates:
                candidates.append(name)
    # Типичные имена, если drivers() пуст (Linux с .so).
    for fallback in (
        'ODBC Driver 18 for SQL Server',
        'ODBC Driver 17 for SQL Server',
        '/usr/lib64/libmsodbcsql-18.so',
        '/opt/microsoft/msodbcsql18/lib64/libmsodbcsql-18.so',
    ):
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates


def _connection_string(driver: str, database: str | None = None) -> str:
    """Собирает ODBC-строку для OISUU."""
    host = oisuu_conn.get('host') or ''
    user = oisuu_conn.get('username') or ''
    password = oisuu_conn.get('password') or ''
    db = database or oisuu_conn.get('database') or 'OISUU_report'
    if not host or not user:
        raise RuntimeError(
            'Не заданы OISUU_CONN_HOST / OISUU_CONN_USERNAME. '
            'Скопируйте env_template → .env и заполните.'
        )
    return (
        f'DRIVER={{{driver}}};'
        f'SERVER={host};'
        f'DATABASE={db};'
        f'UID={user};'
        f'PWD={password};'
        'TrustServerCertificate=yes;'
    )


def connect_oisuu(database: str | None = None) -> Any:
    """Открывает pyodbc-соединение с OISUU_report.

    Returns:
        Живое соединение (вызывающий закрывает сам).
    """
    pyodbc = _require_pyodbc()
    last_error: Exception | None = None
    tried: list[str] = []
    for driver in _odbc_driver_candidates():
        tried.append(driver)
        try:
            return pyodbc.connect(_connection_string(driver, database=database))
        except Exception as error:  # pragma: no cover - зависит от окружения
            last_error = error
    raise RuntimeError(
        'Не удалось подключиться к OISUU.\n'
        f'Пробовали драйверы: {tried}\n'
        'Задайте OISUU_ODBC_DRIVER в .env при необходимости.\n'
        f'Последняя ошибка: {last_error}'
    ) from last_error
