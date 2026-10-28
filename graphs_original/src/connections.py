"""Параметры подключений к БД (не бизнес-настройки пайплайна)."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Сначала .env рядом с graphs_original, затем cwd (Jupyter / сервер).
load_dotenv(_PROJECT_ROOT / '.env')
load_dotenv()

#: Типичные пути к libmsodbcsql на Jupyter / RHEL (как в monitoring / fin_effects).
_SO_GLOBS = (
    '/usr/lib64/libmsodbcsql*.so',
    '/usr/lib/libmsodbcsql*.so',
    '/opt/microsoft/msodbcsql*/lib64/libmsodbcsql*.so',
    '/opt/microsoft/msodbcsql*/lib64/libmsodbcsql-*.so*',
)


def _env(*keys: str, default: str = '') -> str:
    """Первое непустое значение из списка ключей окружения."""
    for key in keys:
        value = os.getenv(key)
        if value is not None and str(value).strip():
            return str(value).strip().strip('"').strip("'")
    return default


oisuu_conn = {
    'host': _env('OISUU_CONN_HOST', 'OISUU_SERVER', default='oisuu-db-biz.vsk.ru'),
    'username': _env('OISUU_CONN_USERNAME', 'OISUU_UID'),
    'password': _env('OISUU_CONN_PASSWORD', 'OISUU_PWD'),
    'database': _env('OISUU_CONN_DATABASE', 'OISUU_DATABASE', default='OISUU_report'),
    'driver': _env('OISUU_ODBC_DRIVER'),
}

actuar_conn = {
    'host': _env('ACTUAR_CONN_HOST', 'ORACLE_CONN_HOST', default='actuar2.vsk.ru'),
    'username': _env('ACTUAR_CONN_USERNAME', 'ORACLE_CONN_USERNAME'),
    'password': _env('ACTUAR_CONN_PASSWORD', 'ORACLE_CONN_PASSWORD'),
    'database': _env('ACTUAR_CONN_DATABASE', 'ORACLE_CONN_DATABASE', default='Motor'),
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


def list_available_odbc_drivers() -> list[str]:
    """Драйверы, зарегистрированные в unixODBC / Windows ODBC."""
    try:
        return list(_require_pyodbc().drivers())
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


def _odbc_driver_candidates(explicit: str | None = None) -> list[str]:
    """Кандидаты: явный → реальные ``.so`` → зарегистрированные SQL Server / FreeTDS.

    Именованные «ODBC Driver 18» часто прописаны в odbcinst с несуществующим
    путём — поэтому сначала предпочитаем найденные на диске библиотеки.
    """
    drivers: list[str] = []

    def _add(name: str | None) -> None:
        if name and name not in drivers:
            drivers.append(name)

    preferred = (explicit or oisuu_conn.get('driver') or '').strip()
    _add(preferred or None)
    for so_path in find_msodbcsql_libraries():
        _add(so_path)
    for registered in list_available_odbc_drivers():
        low = registered.lower()
        if any(token in low for token in ('sql server', 'msodbcsql', 'freetds', 'tds')):
            _add(registered)
    return drivers


def _connection_string(driver: str, database: str | None = None) -> str:
    """Собирает ODBC-строку для OISUU."""
    host = oisuu_conn.get('host') or ''
    user = oisuu_conn.get('username') or ''
    password = oisuu_conn.get('password') or ''
    db = database or oisuu_conn.get('database') or 'OISUU_report'
    if not host or not user:
        raise RuntimeError(
            'Не заданы OISUU_CONN_HOST / OISUU_CONN_USERNAME '
            '(или OISUU_SERVER / OISUU_UID). '
            'Скопируйте env_template → .env и заполните.'
        )
    conn = (
        f'DRIVER={{{driver}}};'
        f'SERVER={host};'
        f'DATABASE={db};'
        f'UID={user};'
        f'PWD={password};'
        'Trusted_Connection=no;'
    )
    low = driver.lower()
    if 'freetds' in low or low.endswith('tds'):
        conn += 'TDS_Version=7.4;'
    # Driver 18 по умолчанию Encrypt=yes; корпоративный SQL часто с self-signed.
    if '18' in low or 'sql server' in low or 'msodbcsql' in low:
        conn += 'Encrypt=yes;TrustServerCertificate=yes;'
    else:
        conn += 'TrustServerCertificate=yes;'
    return conn


def connect_oisuu(database: str | None = None) -> Any:
    """Открывает pyodbc-соединение с OISUU_report.

    Returns:
        Живое соединение (вызывающий закрывает сам).
    """
    # Перечитываем .env на случай, если его положили после импорта модуля.
    load_dotenv(_PROJECT_ROOT / '.env')
    load_dotenv()
    oisuu_conn['host'] = _env('OISUU_CONN_HOST', 'OISUU_SERVER', default='oisuu-db-biz.vsk.ru')
    oisuu_conn['username'] = _env('OISUU_CONN_USERNAME', 'OISUU_UID')
    oisuu_conn['password'] = _env('OISUU_CONN_PASSWORD', 'OISUU_PWD')
    oisuu_conn['database'] = _env(
        'OISUU_CONN_DATABASE', 'OISUU_DATABASE', default='OISUU_report'
    )
    oisuu_conn['driver'] = _env('OISUU_ODBC_DRIVER')

    pyodbc = _require_pyodbc()
    available = list_available_odbc_drivers()
    libs = find_msodbcsql_libraries()
    candidates = _odbc_driver_candidates()

    if not candidates:
        raise RuntimeError(
            'Не найден ODBC-драйвер SQL Server.\n'
            f'pyodbc.drivers() = {available}\n'
            f'libmsodbcsql*.so = {libs}\n'
            'На Jupyter найдите .so и пропишите в .env:\n'
            '  find /usr /opt -name \'libmsodbcsql*\' 2>/dev/null\n'
            '  OISUU_ODBC_DRIVER=/путь/к/libmsodbcsql-….so'
        )

    last_error: Exception | None = None
    tried: list[str] = []
    for driver in candidates:
        tried.append(driver)
        try:
            return pyodbc.connect(_connection_string(driver, database=database))
        except Exception as error:  # pragma: no cover - зависит от окружения
            last_error = error

    raise RuntimeError(
        'Не удалось подключиться к OISUU.\n'
        f'pyodbc.drivers() = {available}\n'
        f'libmsodbcsql on disk = {libs}\n'
        f'Пробовали: {tried}\n'
        'Если «file not found» — в odbcinst битый путь. Укажите реальный .so:\n'
        '  find /usr /opt -name \'libmsodbcsql*\' 2>/dev/null\n'
        '  OISUU_ODBC_DRIVER=/путь/к/найденному.so\n'
        f'Последняя ошибка: {last_error}'
    ) from last_error
