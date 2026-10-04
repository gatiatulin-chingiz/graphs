"""Настройки приложения из переменных окружения."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Загрузка .env из корня проекта (не читаем содержимое в коде — только путь).
load_dotenv()


DEFAULT_WRONG_PEOPLE = [
    "НЕИЗВЕСТНО",
    "НЕ УСТАНОВЛЕН",
    "НЕ УСТАНОВЛЕНО",
    "ОТСУТСТВУЕТ",
    "-",
    "",
]

DEFAULT_WRONG_AUTO = {
    "ОТСУТСТВУЕТ": None,
    "НЕИЗВЕСТНО": None,
    "-": None,
    "": None,
}


@dataclass(frozen=True)
class Settings:
    """Параметры подключения и обработки данных."""

    db_server: str
    db_port: int
    db_user: str
    db_password: str
    db_name: str
    group_size_bound: int
    wrong_people_values: list[str]
    wrong_auto_values: dict[str, str | None]
    input_dir: Path
    output_dir: Path

    @classmethod
    def from_env(cls) -> Settings:
        """Собрать настройки из переменных окружения."""
        wrong_people = os.getenv("WRONG_PEOPLE_VALUES")
        wrong_auto = os.getenv("WRONG_AUTO_VALUES")

        return cls(
            db_server=os.getenv("DB_SERVER", ""),
            db_port=int(os.getenv("DB_PORT", "1433")),
            db_user=os.getenv("DB_USER", ""),
            db_password=os.getenv("DB_PASSWORD", ""),
            db_name=os.getenv("DB_NAME", ""),
            group_size_bound=int(os.getenv("GROUP_SIZE_BOUND", "3")),
            wrong_people_values=(
                json.loads(wrong_people) if wrong_people else list(DEFAULT_WRONG_PEOPLE)
            ),
            wrong_auto_values=(
                json.loads(wrong_auto) if wrong_auto else dict(DEFAULT_WRONG_AUTO)
            ),
            input_dir=Path(os.getenv("INPUT_DIR", ".")),
            output_dir=Path(os.getenv("OUTPUT_DIR", "./output")),
        )

    def validate_db(self) -> None:
        """Проверить, что заданы параметры SQL Server."""
        missing = [
            name
            for name, value in (
                ("DB_SERVER", self.db_server),
                ("DB_USER", self.db_user),
                ("DB_PASSWORD", self.db_password),
                ("DB_NAME", self.db_name),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"Для загрузки из SQL задайте переменные окружения: {', '.join(missing)}"
            )
