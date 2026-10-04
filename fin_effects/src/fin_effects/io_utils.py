"""Утилиты ввода-вывода: промежуточные артефакты, Excel-отчёты, печать таблиц.

Исходная тетрадь сохраняла и перечитывала промежуточные ``.pkl`` вперемешку с
расчётами (около десяти пар «dump/load»), причём повторное чтение уже
объявленных переменных создавало иллюзию работы с «новыми» данными. Здесь:
артефакты пишутся явно, читаются явно, форматы — parquet/pickle с проверкой схемы.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from .config import INTERIM_DIR, REPORT_DIR


def read_pickle(path: str | Path) -> object:
    """Читает объект из pickle-файла.

    Args:
        path: путь к файлу.

    Returns:
        Десериализованный объект.

    Raises:
        FileNotFoundError: если файл не найден.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Не найден артефакт: {resolved}")
    with resolved.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - доверенные локальные артефакты


def write_pickle(obj: object, path: str | Path) -> Path:
    """Сохраняет объект в pickle-файл, создавая каталоги.

    Args:
        obj: сериализуемый объект.
        path: путь к файлу.

    Returns:
        Путь к сохранённому файлу.
    """
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return resolved


def cache_frame(frame: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Сохраняет ``DataFrame`` в parquet.

    Args:
        frame: таблица.
        path: путь к файлу.
        index: сохранять ли индекс.

    Returns:
        Путь к сохранённому файлу.
    """
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(resolved, index=index)
    return resolved


def write_excel_report(
    sheets: Mapping[str, pd.DataFrame],
    path: str | Path,
    *,
    formats: Mapping[str, str] | None = None,
) -> Path:
    """Пишет многолистовой Excel-отчёт.

    Args:
        sheets: соответствие «имя листа -> таблица».
        path: путь к файлу ``.xlsx``.
        formats: опциональные форматы чисел по имени листа (например ``'#,##0'``).

    Returns:
        Путь к сохранённому файлу.
    """
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    formats = formats or {}

    with pd.ExcelWriter(resolved, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            safe_name = sheet_name[:31]
            frame.to_excel(writer, sheet_name=safe_name)
            number_format = formats.get(sheet_name)
            if number_format:
                worksheet = writer.sheets[safe_name]
                for row in worksheet.iter_rows(
                    min_row=2,
                    max_col=max(1, frame.shape[1]),
                    max_row=frame.shape[0] + 1,
                ):
                    for cell in row:
                        if isinstance(cell.value, (int, float)):
                            cell.number_format = number_format
    return resolved


def interim_path(name: str, suffix: str = ".parquet") -> Path:
    """Путь к промежуточному артефакту в ``data/interim``.

    Args:
        name: имя артефакта без расширения.
        suffix: расширение.

    Returns:
        Абсолютный путь.
    """
    return INTERIM_DIR / f"{name}{suffix}"


def report_path(name: str) -> Path:
    """Путь к файлу отчёта в ``outputs/reports``.

    Args:
        name: имя файла.

    Returns:
        Абсолютный путь.
    """
    return REPORT_DIR / name


def format_money(value: float, unit: str = "млн ₽") -> str:
    """Форматирует сумму в удобочитаемую строку.

    Args:
        value: сумма в рублях.
        unit: подпись единицы измерения (сумма переводится в миллионы).

    Returns:
        Строка вида ``"1 234.5 млн ₽"``.

    Examples:
        >>> format_money(1_234_500_000)
        '1 234.5 млн ₽'
    """
    millions = value / 1_000_000
    return f"{millions:,.1f} {unit}".replace(",", " ")


def describe_frame(frame: pd.DataFrame, name: str = "df") -> str:
    """Короткое текстовое описание таблицы для логов и markdown.

    Args:
        frame: таблица.
        name: имя таблицы.

    Returns:
        Строка вида ``"df: 12 345 строк x 16 колонок"``.
    """
    return f"{name}: {frame.shape[0]:,} строк x {frame.shape[1]} колонок".replace(",", " ")


def subset_columns(frame: pd.DataFrame, columns: Iterable[str], *, strict: bool = False) -> pd.DataFrame:
    """Возвращает подмножество колонок, сообщая об отсутствующих.

    Args:
        frame: таблица.
        columns: требуемые колонки.
        strict: если ``True`` — падать при отсутствии колонки.

    Returns:
        Таблица с выбранными колонками.

    Raises:
        KeyError: если ``strict`` и колонки отсутствуют.
    """
    required = list(columns)
    missing = [column for column in required if column not in frame.columns]
    if missing and strict:
        raise KeyError(f"Отсутствуют обязательные колонки: {missing}")
    present = [column for column in required if column in frame.columns]
    return frame.loc[:, present]
