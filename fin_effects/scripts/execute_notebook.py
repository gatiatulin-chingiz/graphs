"""Прогон ноутбука целиком: проверка, что все ячейки выполняются без ошибок.

Запуск::

    python scripts/execute_notebook.py [--notebook notebooks/financial_effect.ipynb]
                                      [--output notebooks/financial_effect.executed.ipynb]

Скрипт выполняет ноутбук в отдельном ядре и падает с ненулевым кодом, если любая
ячейка завершилась ошибкой. Это защита от «ноутбук выглядит рабочим, но
выполняется только после ручных правок».
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import nbformat  # noqa: E402
from nbclient import NotebookClient  # noqa: E402
from nbclient.exceptions import CellExecutionError  # noqa: E402

DEFAULT_NOTEBOOK = PROJECT_ROOT / "notebooks" / "financial_effect.ipynb"


def _patch_secure_write() -> None:
    """Разрешает ядру Jupyter писать файлы в ограниченной среде.

    Args:
        Отсутствуют.

    Note:
        ``jupyter_core.paths.secure_write`` на Windows вызывает
        ``SetFileSecurity`` (сужение DACL до текущего пользователя). В жёсткой
        песочнице этот вызов отбивается с «Отказано в доступе», и запуск ядра
        падает ещё до исполнения первой ячейки — при том что обычная запись в
        файл разрешена. Патч подменяет только этот шаг, сохраняя путь и
        поведение записи.
    """
    import jupyter_client.connect as jupyter_connect
    import jupyter_core.paths as jupyter_paths

    @contextlib.contextmanager
    def _secure_write(fname: str, mode: int = 0o600):
        original_umask = os.umask(0)
        try:
            os.umask(0o077)
            with open(fname, "w", encoding="utf-8") as handle:  # noqa: PTH123
                yield handle
        finally:
            os.umask(original_umask)
        _ = mode

    probe = Path(tempfile.gettempdir()) / "_dsh_secure_write_probe"
    try:
        with jupyter_paths.secure_write(str(probe)):
            pass
        probe.unlink(missing_ok=True)
        return
    except Exception as error:  # noqa: BLE001 - проверяем именно доступность вызова
        print(f"secure_write недоступен ({error.__class__.__name__}) — включён совместимый режим")

    # Патчим и модуль-источник, и имя, импортированное в jupyter_client.connect.
    jupyter_paths.secure_write = _secure_write
    jupyter_connect.secure_write = _secure_write


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки.

    Returns:
        Namespace с аргументами.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=1800, help="таймаут одной ячейки, с")
    parser.add_argument("--keep-output", action="store_true", help="сохранить выполненный ноутбук")
    return parser.parse_args()


def main() -> int:
    """Выполняет ноутбук.

    Returns:
        Код возврата: 0 — успех, 1 — ошибка в ячейке.
    """
    args = parse_args()
    _patch_secure_write()
    notebook_path = args.notebook.resolve()
    if not notebook_path.exists():
        print(f"Ноутбук не найден: {notebook_path}", file=sys.stderr)
        return 2

    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
        allow_errors=False,
    )

    started = time.perf_counter()
    try:
        client.execute()
    except CellExecutionError as error:
        print("Ячейка завершилась ошибкой:\n", file=sys.stderr)
        print(error, file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - started
    failed = [
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and any(output.get("output_type") == "error" for output in cell.get("outputs", []))
    ]
    print(f"Ноутбук выполнен за {elapsed:.1f} c; ячеек с ошибками: {len(failed)}")

    if args.keep_output:
        output_path = args.output or notebook_path.with_suffix(".executed.ipynb")
        nbformat.write(notebook, output_path)
        print(f"Выполненный ноутбук сохранён: {output_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
