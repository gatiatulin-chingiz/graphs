"""Тесты чтения выгрузок сервиса (xlsx) и парсинга листов участников."""

from __future__ import annotations

import pandas as pd
import pytest

from fin_effects.excel_beneficiaries import (
    discover_workbooks,
    load_beneficiaries,
    load_beneficiaries_from_dir,
    parse_sheet,
    summarize_groups,
)


def test_discover_workbooks_finds_generated_files(synthetic) -> None:
    """Поиск находит все сгенерированные файлы групп."""
    root = synthetic.output_dir / "raw"
    found = discover_workbooks(root, years=[2025])
    assert len(found) == len(synthetic.beneficiary_files)
    assert all(path.suffix == ".xlsx" for path in found)


def test_discover_workbooks_returns_empty_for_missing_dir(tmp_path) -> None:
    """Отсутствующий каталог не приводит к исключению."""
    assert discover_workbooks(tmp_path / "nope") == []


def test_parse_sheet_skips_header_row() -> None:
    """Строка-шапка отбрасывается, данные сохраняются."""
    raw = pd.DataFrame(
        [
            ["ФИО", "Дата рождения", "Номер группы"],
            ["Иванов Иван Иванович", pd.Timestamp("1980-01-01"), 5],
            ["Петров Пётр Петрович", pd.Timestamp("1990-02-02"), 5],
        ]
    )
    parsed = parse_sheet(raw, group_id="grp_1")
    assert parsed["full_name"].tolist() == ["иванов иван иванович", "петров пётр петрович"]
    assert parsed["group_id"].unique().tolist() == ["grp_1"]
    assert parsed["person_key"].iloc[0] == "иванов иван иванович 1980-01-01"


def test_parse_sheet_handles_repeated_header_and_blanks() -> None:
    """Повторная шапка и пустые строки не ломают парсинг."""
    raw = pd.DataFrame(
        [
            ["ФИО", "Дата рождения", "Номер группы"],
            ["Иванов Иван", pd.Timestamp("1980-01-01"), 1],
            [None, None, None],
            ["ФИО", "Дата рождения", "Номер группы"],
            ["Петров Пётр", pd.Timestamp("1990-02-02"), 1],
        ]
    )
    parsed = parse_sheet(raw, group_id="grp_1")
    assert len(parsed) == 2


def test_parse_sheet_accepts_string_dates() -> None:
    """Даты в виде строк разбираются."""
    raw = pd.DataFrame([["Иванов Иван", "1980-01-01", "1"]])
    parsed = parse_sheet(raw, group_id="grp_1")
    assert parsed["person_key"].iloc[0] == "иванов иван 1980-01-01"


def test_parse_sheet_empty_input() -> None:
    """Пустой лист даёт пустую таблицу правильной схемы."""
    assert parse_sheet(pd.DataFrame(), group_id="g").empty
    assert parse_sheet(None, group_id="g").empty


def test_parse_sheet_unparsable_dates_are_kept_by_name() -> None:
    """Участник без разбираемой даты всё равно попадает в выгрузку."""
    raw = pd.DataFrame([["Иванов Иван", "не дата", "1"]])
    parsed = parse_sheet(raw, group_id="grp_1")
    assert parsed["person_key"].tolist() == ["иванов иван"]


def test_load_beneficiaries_round_trip(synthetic) -> None:
    """Прочитанные xlsx совпадают с «эталонной» таблицей участников."""
    loaded, stats = load_beneficiaries(synthetic.beneficiary_files, workers=1)
    assert stats.sheets_skipped == 0
    assert stats.errors == ()
    expected = set(synthetic.beneficiaries["person_key"])
    assert set(loaded["person_key"]) == expected
    assert loaded["group_id"].nunique() == synthetic.groups["group_id"].nunique()


def test_load_beneficiaries_parallel_matches_sequential(synthetic) -> None:
    """Параллельное чтение даёт тот же результат, что последовательное."""
    sequential, _ = load_beneficiaries(synthetic.beneficiary_files[:6], workers=1)
    parallel, _ = load_beneficiaries(synthetic.beneficiary_files[:6], workers=3)
    assert sorted(sequential["person_key"]) == sorted(parallel["person_key"])


def test_load_beneficiaries_from_dir(synthetic) -> None:
    """Загрузка из каталога находит группы."""
    loaded, stats = load_beneficiaries_from_dir(synthetic.output_dir / "raw", years=[2025])
    assert not loaded.empty
    assert stats.files == len(synthetic.beneficiary_files)


def test_load_beneficiaries_empty_input() -> None:
    """Пустой список файлов даёт пустую таблицу."""
    loaded, stats = load_beneficiaries([])
    assert loaded.empty
    assert stats.files == 0


def test_summarize_groups(synthetic) -> None:
    """Сводка по группам содержит число участников."""
    loaded, _ = load_beneficiaries(synthetic.beneficiary_files, workers=1)
    summary = summarize_groups(loaded)
    assert {"group_id", "members", "sheets", "sources"} <= set(summary.columns)
    assert summary["members"].sum() == len(loaded)
    assert summary["members"].is_monotonic_decreasing


def test_summarize_groups_empty() -> None:
    """Пустая таблица участников даёт пустую сводку."""
    assert summarize_groups(pd.DataFrame()).empty


@pytest.mark.parametrize("workers", [1, 2])
def test_group_id_from_file_name(synthetic, workers: int) -> None:
    """Группа определяется файлом выгрузки, а не листом."""
    loaded, _ = load_beneficiaries(synthetic.beneficiary_files[:5], workers=workers)
    assert set(loaded["group_id"]) == {
        path.stem for path in synthetic.beneficiary_files[:5]
    }
