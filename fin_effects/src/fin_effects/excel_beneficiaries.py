"""Загрузка выгрузок сервиса графовой аналитики (xlsx с участниками групп).

Каждый файл — одна или несколько групп; каждый лист — список участников
(ФИО, дата рождения). Колонка с номером группы повторяет номер по всем строкам
листа, поэтому из неё и берётся идентификатор группы.

Оптимизации относительно исходной тетради:

1. Файл открывается **один раз** через ``pd.ExcelFile`` (в тетради было два
   полных чтения: сначала ``ExcelFile`` ради проверки видимости листов, потом
   ``pd.read_excel(file, sheet_name=...)`` по каждому листу).
2. Листы читаются с ``header=None`` и позиционными колонками — иначе
   ``pd.read_excel`` съедал первую строку данных как заголовок, и обращение
   ``df[0]``/``df[1]``/``df[2]`` давало ``KeyError``.
3. Нечитаемые листы логируются, а не «игнорируются» молча (в тетради был
   ``try/except: pass`` вокруг проверки видимости).
4. Файлы можно обрабатывать в пуле процессов — чтение Excel это I/O + CPU.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import pandas as pd

from .config import EXCEL_HEADER_TOKENS, EXCEL_WORKERS
from .textnorm import format_person_key, normalize_series, normalize_text

LOGGER = logging.getLogger(__name__)

#: Позиции колонок в листе участников (после ``header=None``).
COLUMN_NAME = 0
COLUMN_BIRTH_DATE = 1
COLUMN_GROUP = 2

#: Имена итоговых колонок.
OUTPUT_COLUMNS = ("full_name", "birth_date", "group_id")


@dataclass(frozen=True)
class BeneficiaryLoadStats:
    """Статистика загрузки: сколько файлов/листов прочитано и сколько пропущено."""

    files: int = 0
    sheets_total: int = 0
    sheets_ok: int = 0
    sheets_skipped: int = 0
    rows: int = 0
    errors: tuple[str, ...] = ()

    def as_text(self) -> str:
        """Человекочитаемая сводка.

        Returns:
            Строка для логов ноутбука.
        """
        return (
            f"файлов: {self.files}, листов: {self.sheets_ok}/{self.sheets_total} "
            f"(пропущено {self.sheets_skipped}), строк: {self.rows}, "
            f"ошибок: {len(self.errors)}"
        )


def discover_workbooks(root: str | Path, years: Iterable[str | int] = ()) -> list[Path]:
    """Ищет xlsx/xls-файлы выгрузки сервиса.

    Args:
        root: корневой каталог выгрузки.
        years: подкаталоги-годы; если пусто — поиск по всему ``root``.

    Returns:
        Отсортированный список путей к файлам.

    Note:
        В тетради использовался ``glob.glob`` в цикле по годам и расширениям.
        Здесь то же самое делает ``Path.glob`` без лишнего промежуточного списка.
    """
    root_path = Path(root)
    if not root_path.exists():
        LOGGER.warning("Каталог выгрузок не найден: %s", root_path)
        return []

    year_dirs = [root_path / str(year) for year in years] or [root_path]
    found: list[Path] = []
    for directory in year_dirs:
        if not directory.exists():
            LOGGER.warning("Нет каталога за год: %s", directory)
            continue
        for pattern in ("**/*.xlsx", "**/*.xls"):
            found.extend(directory.glob(pattern))

    unique = sorted(
        {path for path in found if not path.name.startswith("~$") and path.is_file()}
    )
    LOGGER.info("Найдено файлов выгрузки: %d", len(unique))
    return unique


def visible_sheet_names(workbook: pd.ExcelFile) -> list[str]:
    """Возвращает имена видимых листов.

    Args:
        workbook: открытый ``pd.ExcelFile``.

    Returns:
        Список имён листов, у которых ``sheet_state != 'hidden'``.
    """
    book = getattr(workbook, "book", None)
    if book is None:  # pragma: no cover - зависит от движка чтения
        return list(workbook.sheet_names)

    visible: list[str] = []
    for sheet_name in workbook.sheet_names:
        try:
            sheet = book[sheet_name]
            state = getattr(sheet, "sheet_state", None)
        except (KeyError, AttributeError):  # pragma: no cover - защитный путь
            LOGGER.debug("Не удалось определить видимость листа %s", sheet_name)
            visible.append(sheet_name)
            continue
        if state is None or state == "visible":
            visible.append(sheet_name)
    return visible


def _looks_like_header(row: Sequence[object]) -> bool:
    """Является ли строка шапкой таблицы участников.

    Args:
        row: значения строки листа.

    Returns:
        ``True``, если строку нельзя считать участником группы.

    Note:
        Признаки шапки — служебные токены (``фио``, ``др``, ``group_number``),
        пустое ФИО или пустой номер группы. **Нераспознаваемая дата признаком
        шапки не является**: в исходной тетради такие строки отбрасывались
        («фильтр по ``др``»), из-за чего участники с нестандартной датой молча
        исчезали из расчёта.
    """
    if len(row) < 3 or pd.isna(row[COLUMN_GROUP]):
        return True

    name = normalize_text(row[COLUMN_NAME])
    if not name:
        return True

    tokens = {normalize_text(value) for value in row}
    return bool(tokens & set(EXCEL_HEADER_TOKENS))


def parse_sheet(
    raw_sheet: pd.DataFrame,
    group_id: str,
) -> pd.DataFrame:
    """Превращает «сырой» лист участников в нормализованную таблицу.

    Args:
        raw_sheet: лист, прочитанный с ``header=None``.
        group_id: идентификатор группы (имя листа или номер файла).

    Returns:
        Таблица с колонками :data:`OUTPUT_COLUMNS`, где ``person_key`` — ключ
        сопоставления «ФИО + дата рождения».
    """
    if raw_sheet is None or raw_sheet.empty:
        return _empty_beneficiaries()

    working = raw_sheet.iloc[:, :3].copy()
    working.columns = [COLUMN_NAME, COLUMN_BIRTH_DATE, COLUMN_GROUP]
    header_mask = working.apply(_looks_like_header, axis=1)
    working = working.loc[~header_mask]

    if working.empty:
        return _empty_beneficiaries()

    working["full_name"] = normalize_series(working[COLUMN_NAME])
    working["birth_date"] = pd.to_datetime(
        working[COLUMN_BIRTH_DATE], errors="coerce"
    ).dt.normalize()
    working["group_id"] = str(group_id)

    working = working.loc[working["full_name"] != ""]
    working["person_key"] = [
        format_person_key(name, birth_date)
        for name, birth_date in zip(working["full_name"], working["birth_date"], strict=True)
    ]

    return working.loc[:, (*OUTPUT_COLUMNS, "person_key")].reset_index(drop=True)


def _empty_beneficiaries() -> pd.DataFrame:
    """Пустая таблица участников с правильной схемой.

    Returns:
        Пустой ``DataFrame`` с колонками :data:`OUTPUT_COLUMNS` + ``person_key``.
    """
    return pd.DataFrame(columns=(*OUTPUT_COLUMNS, "person_key"))


def read_beneficiary_file(
    path: str | Path,
    engine: str = "openpyxl",
) -> tuple[pd.DataFrame, BeneficiaryLoadStats]:
    """Читает один файл выгрузки сервиса.

    Args:
        path: путь к xlsx/xls.
        engine: движок чтения (``openpyxl``/``calamine``).

    Returns:
        Кортеж «таблица участников, статистика». Ошибки чтения не приводят к
        падению: файл попадает в ``stats.errors``.
    """
    path = Path(path)
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    sheets_total = 0
    sheets_skipped = 0

    try:
        with pd.ExcelFile(path, engine=engine) as workbook:
            for sheet_name in visible_sheet_names(workbook):
                sheets_total += 1
                try:
                    raw_sheet = workbook.parse(sheet_name=sheet_name, header=None)
                    parsed = parse_sheet(raw_sheet, group_id=str(path.stem))
                    if parsed.empty:
                        sheets_skipped += 1
                        LOGGER.debug("Лист без участников: %s / %s", path.name, sheet_name)
                        continue
                    parsed["source_file"] = path.name
                    parsed["source_sheet"] = sheet_name
                    frames.append(parsed)
                except Exception as error:  # noqa: BLE001 - данные бывают любые
                    sheets_skipped += 1
                    message = f"{path.name} / {sheet_name}: {error}"
                    errors.append(message)
                    LOGGER.warning("Не удалось прочитать лист %s", message)
    except Exception as error:  # noqa: BLE001
        message = f"{path.name}: {error}"
        errors.append(message)
        LOGGER.error("Не удалось открыть файл %s", message)

    frame = pd.concat(frames, ignore_index=True) if frames else _empty_beneficiaries()
    stats = BeneficiaryLoadStats(
        files=1,
        sheets_total=sheets_total,
        sheets_ok=len(frames),
        sheets_skipped=sheets_skipped,
        rows=len(frame),
        errors=tuple(errors),
    )
    return frame, stats


def load_beneficiaries(
    paths: Iterable[str | Path],
    *,
    engine: str = "openpyxl",
    workers: int = 1,
    group_by_file: bool = True,
) -> tuple[pd.DataFrame, BeneficiaryLoadStats]:
    """Читает все выгрузки сервиса в одну таблицу.

    Args:
        paths: пути к файлам (см. :func:`discover_workbooks`).
        engine: движок чтения Excel.
        workers: число процессов; ``1`` — последовательно.
        group_by_file: ``True`` — группа определяется файлом (листы одного файла —
            одна группа, как в исходной выгрузке «один файл = одна группа»);
            ``False`` — группа определяется листом.

    Returns:
        Кортеж «таблица участников, агрегированная статистика».
    """
    path_list = [Path(path) for path in paths]
    if not path_list:
        return _empty_beneficiaries(), BeneficiaryLoadStats()

    results = _read_all(path_list, engine=engine, workers=workers)

    frames = [frame for frame, _ in results if not frame.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else _empty_beneficiaries()

    if group_by_file and not combined.empty:
        combined["group_id"] = combined["source_file"].map(_file_group_id)

    stats = BeneficiaryLoadStats(
        files=sum(stat.files for _, stat in results),
        sheets_total=sum(stat.sheets_total for _, stat in results),
        sheets_ok=sum(stat.sheets_ok for _, stat in results),
        sheets_skipped=sum(stat.sheets_skipped for _, stat in results),
        rows=len(combined),
        errors=tuple(error for _, stat in results for error in stat.errors),
    )
    LOGGER.info("Загрузка выгрузок сервиса: %s", stats.as_text())
    return combined, stats


def _read_all(
    path_list: list[Path],
    *,
    engine: str,
    workers: int,
) -> list[tuple[pd.DataFrame, BeneficiaryLoadStats]]:
    """Читает файлы последовательно или в пуле процессов.

    Args:
        path_list: пути к файлам.
        engine: движок чтения Excel.
        workers: желаемое число процессов.

    Returns:
        Список результатов :func:`read_beneficiary_file`.

    Note:
        Пул процессов — оптимизация, а не требование: в средах, где нельзя
        создать именованные каналы (жёсткая песочница, некоторые CI), чтение
        автоматически откатывается к последовательному, а не падает.
    """
    if workers > 1 and len(path_list) > 1:
        try:
            with ProcessPoolExecutor(max_workers=min(workers, len(path_list))) as pool:
                return list(
                    pool.map(_read_file_worker, [str(path) for path in path_list], chunksize=1)
                )
        except (OSError, PermissionError, ImportError) as error:  # pragma: no cover - среда
            LOGGER.warning(
                "Пул процессов недоступен (%s), читаем последовательно", error
            )

    return [read_beneficiary_file(path, engine=engine) for path in path_list]


def _read_file_worker(path: str) -> tuple[pd.DataFrame, BeneficiaryLoadStats]:
    """Обёртка для пула процессов (должна быть на уровне модуля, чтобы пиклилась).

    Args:
        path: путь к файлу.

    Returns:
        Результат :func:`read_beneficiary_file`.
    """
    return read_beneficiary_file(path)


def _file_group_id(source_file: str) -> str:
    """Идентификатор группы по имени файла.

    Args:
        source_file: имя файла выгрузки.

    Returns:
        Имя файла без расширения.
    """
    return Path(str(source_file)).stem


def summarize_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Сводка по группам: сколько участников и листов в каждой.

    Args:
        frame: таблица участников.

    Returns:
        Таблица «group_id, участников, листов, источники».
    """
    if frame.empty:
        return pd.DataFrame(columns=["group_id", "members", "sheets", "sources"])

    summary = (
        frame.groupby("group_id", as_index=False)
        .agg(
            members=("person_key", "size"),
            unique_members=("person_key", "nunique"),
            sheets=("source_sheet", "nunique"),
            sources=("source_file", lambda values: ", ".join(sorted(set(values)))),
        )
        .sort_values("members", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def load_beneficiaries_from_dir(
    root: str | Path,
    years: Iterable[str | int] = (),
    **kwargs,
) -> tuple[pd.DataFrame, BeneficiaryLoadStats]:
    """Находит и читает выгрузки в каталоге.

    Args:
        root: корневой каталог выгрузки.
        years: подкаталоги-годы.
        **kwargs: пробрасывается в :func:`load_beneficiaries`.

    Returns:
        Кортеж «таблица участников, статистика».
    """
    return load_beneficiaries(discover_workbooks(root, years), **kwargs)


def _glob_legacy(root: str | Path, years: Iterable[str | int] = ()) -> list[str]:
    """Исходный (медленный) способ поиска файлов — оставлен для бенчмарков.

    Args:
        root: корневой каталог.
        years: подкаталоги-годы.

    Returns:
        Список путей.
    """
    import os

    result: list[str] = []
    for year in years:
        for ext in ("*.xlsx", "*.xls"):
            result.extend(glob(os.path.join(str(root), str(year), "**", ext), recursive=True))
    return result
