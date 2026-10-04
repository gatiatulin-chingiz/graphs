"""Конфигурация пайплайна: пути, имена колонок, параметры методики.

Все «магические» константы исходной тетради собраны здесь, чтобы их можно было
менять в одном месте (и переопределять из ноутбука через :func:`with_overrides`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SYNTH_DIR = DATA_DIR / "synthetic"
INTERIM_DIR = DATA_DIR / "interim"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"

for _directory in (DATA_DIR, RAW_DIR, SYNTH_DIR, INTERIM_DIR, OUTPUT_DIR, REPORT_DIR, FIGURE_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

#: Имена промежуточных артефактов (pkl/parquet).
ARTIFACT_BENEFICIARIES = INTERIM_DIR / "df_beneficiaries.pkl"
ARTIFACT_LOSSES = INTERIM_DIR / "losses.pkl"
ARTIFACT_LOSS_FACTS = INTERIM_DIR / "loss_facts.parquet"
ARTIFACT_GROUP_SUMMARY = INTERIM_DIR / "group_summary.parquet"
ARTIFACT_EFFECT = INTERIM_DIR / "financial_effect.parquet"


# ---------------------------------------------------------------------------
# Параметры бизнес-логики
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EffectParams:
    """Параметры методики расчёта финансового эффекта.

    Attributes:
        service_share: доля вклада (заслуги) графового сервиса в предотвращённый
            убыток. В исходной тетради это коэффициент ``0.35``.
        min_losses_total: минимальное число убытков у группы, чтобы считать её
            «устойчивой» (в исходной тетради задумано ``3``, но фильтр закомментирован).
        min_losses_per_year: минимальное число убытков группы в году, чтобы этот год
            считался «активным». ``1`` = любой год с убытками.
        first_year: нижняя граница периода наблюдения по дате убытка.
        last_year: верхняя граница периода наблюдения / горизонт прогноза эффекта.
        project_years: на сколько лет после года обнаружения проецируется эффект
            (1 = только год обнаружения).
        baseline_window: окно расчёта «среднего годового убытка» группы:
            ``"full"`` — вся наблюдаемая история, ``"before_detection"`` — только
            годы строго до года обнаружения.
        min_years_for_baseline: минимальное число активных лет, необходимое для
            попадания группы в расчёт (защита от единичных выбросов).
    """

    service_share: float = 0.35
    min_losses_total: int = 3
    min_losses_per_year: int = 3
    first_year: int = 2020
    last_year: int = 2025
    project_years: int = 2
    baseline_window: str = "full"
    min_years_for_baseline: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.service_share <= 1.0:
            raise ValueError("service_share должен быть в диапазоне [0, 1]")
        if self.baseline_window not in {"full", "before_detection"}:
            raise ValueError("baseline_window: 'full' или 'before_detection'")
        if self.last_year < self.first_year:
            raise ValueError("last_year должен быть >= first_year")
        if self.project_years < 1:
            raise ValueError("project_years должен быть >= 1")

    @property
    def observation_years(self) -> range:
        """Годы периода наблюдения (включительно)."""
        return range(self.first_year, self.last_year + 1)


DEFAULT_PARAMS = EffectParams()

#: Метки классов релевантности группы (используются и в отчётах).
RELEVANCE_ALIVE = "Продолжает действовать"
RELEVANCE_DIED_AFTER = "Распалась после обнаружения"
RELEVANCE_DIED_BEFORE = "Распалась самостоятельно"

RELEVANCE_LABELS = (RELEVANCE_ALIVE, RELEVANCE_DIED_AFTER, RELEVANCE_DIED_BEFORE)

#: Классы, по которым сервис даёт эффект (группа «жива» на момент обнаружения).
RELEVANT_CLASSES = (RELEVANCE_ALIVE, RELEVANCE_DIED_AFTER)


# ---------------------------------------------------------------------------
# Палитра САО ВСК (брендбук: синий / светло-синий, логотип #2D2D81)
# ---------------------------------------------------------------------------

VSK_BLUE = "#2D2D81"
VSK_BLUE_DARK = "#1E1E5A"
VSK_BLUE_MID = "#3D4FA8"
VSK_SKY = "#6BA3D9"
VSK_SKY_SOFT = "#E8F0FA"
VSK_ACCENT = "#E0922F"
VSK_MUTED = "#8A93A8"
VSK_INK = "#1A1A2E"

#: Палитра для серий графиков (порядок — основной → акцентный → вспомогательные).
VSK_PALETTE = (
    VSK_BLUE,
    VSK_ACCENT,
    VSK_SKY,
    VSK_BLUE_MID,
    VSK_BLUE_DARK,
    VSK_MUTED,
)

#: Цвета классов релевантности в графиках/отчётах.
VSK_RELEVANCE_COLORS = {
    RELEVANCE_ALIVE: VSK_BLUE,
    RELEVANCE_DIED_AFTER: VSK_ACCENT,
    RELEVANCE_DIED_BEFORE: VSK_MUTED,
}


# ---------------------------------------------------------------------------
# Схема данных
# ---------------------------------------------------------------------------

#: Колонки-идентификаторы физлиц в выгрузке Losses: ФИО+ДР.
PERSON_COLUMNS: tuple[str, ...] = (
    "Убыток виновник полннаименование",
    "Потерпевший.Полн наименование",
    "Страхователь.Полн наименование",
    "Виновник.Полн наименование",
    "Потерпевший объект владелец.Полн наименование",
    "Страхователь объект владелец.Полн наименование",
    "Лицо управлявшее ТС.Полн наименование",
    "Получатель возмещения.Полн наименование",
    "Потерпевший страхователь.Полн наименование",
)

#: Колонки-идентификаторы объектов (VIN).
OBJECT_COLUMNS: tuple[str, ...] = (
    "Объект виновника.Идентификатор",
    "Объект потерпевшего.Идентификатор",
    "Убыток.Объект страхователя.Идентификатор",
)

#: Соответствие «ФИО -> дата рождения» в выгрузке Losses.
PERSON_DATE_PAIRS: dict[str, str] = {
    "Убыток виновник полннаименование": "Убыток виновник датарождения",
    "Потерпевший.Полн наименование": "Потерпевший.Дата рождения",
    "Страхователь.Полн наименование": "Страхователь.Дата рождения",
    "Виновник.Полн наименование": "Виновник.Дата рождения",
    "Потерпевший объект владелец.Полн наименование": "Потерпевший объект владелец.Дата рождения",
    "Страхователь объект владелец.Полн наименование": (
        "Страхователь объект владелец.Дата рождения"
    ),
    "Лицо управлявшее ТС.Полн наименование": "Лицо управлявшее ТС.Дата рождения",
    "Получатель возмещения.Полн наименование": "Получатель возмещения.Дата рождения",
    "Потерпевший страхователь.Полн наименование": "Потерпевший страхователь.Дата рождения",
}

#: Колонки-ключи участника: только идентификаторы физлиц (ФИО+ДР).
#:
#: VIN-колонки сюда сознательно **не** входят: они уникальны для каждого убытка,
#: поэтому любой «ключ группы», собранный с их участием, вырождается в «одна
#: группа = один убыток» и ломает агрегацию по группам.
BENEFICIARY_KEY_COLUMNS: tuple[str, ...] = tuple(f"_{name}" for name in PERSON_COLUMNS)

#: Денежные колонки убытка.
MONEY_COLUMNS: tuple[str, ...] = (
    "ValueRub",
    "Сумма_выплат_по_претензиям",
    "Сумма_взыскано_по_ФУ",
    "Суммы_взыскано_по_иску",
)

#: Листы выгрузки сервиса, которые не являются группами.
EXCEL_HEADER_TOKENS: tuple[str, ...] = ("фио", "др", "group_number", "дата рождения", "группа")

#: Число листов, читаемых параллельно (``None`` = все ядра).
EXCEL_WORKERS = min(8, (os.cpu_count() or 4))


# ---------------------------------------------------------------------------
# Подключение к БД
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DbConfig:
    """Параметры подключения к OISUU (только для обновления сырых данных)."""

    driver: str = "ODBC Driver 17 for SQL Server"
    server: str = "oisuu-db-biz.vsk.ru"
    database: str = "OISUU_report"
    uid: str = field(default_factory=lambda: os.getenv("OISUU_UID", ""))
    pwd: str = field(default_factory=lambda: os.getenv("OISUU_PWD", ""))
    trusted_connection: bool = False

    def connection_string(self, database: str | None = None) -> str:
        """Собирает строку подключения pyodbc.

        Args:
            database: переопределение БД по умолчанию.

        Returns:
            ODBC-строку подключения.

        Note:
            В исходной тетради каталог задавался несуществующим ключом
            ``incoming_claim_new_logicbase=...`` — корректный ключ ``DATABASE``.
        """
        parts = [
            f"DRIVER={{{self.driver}}}",
            f"SERVER={self.server}",
            f"DATABASE={database or self.database}",
        ]
        if self.trusted_connection:
            parts.append("Trusted_Connection=yes")
        else:
            parts.append("Trusted_Connection=no")
            parts.append(f"UID={self.uid}")
            parts.append(f"PWD={self.pwd}")
        return ";".join(parts) + ";"


def with_overrides(params: EffectParams | None = None, **changes) -> EffectParams:
    """Возвращает копию параметров с переопределёнными полями.

    Args:
        params: базовые параметры (по умолчанию :data:`DEFAULT_PARAMS`).
        **changes: поля для замены.

    Returns:
        Новый :class:`EffectParams`.

    Examples:
        >>> with_overrides(service_share=0.5).service_share
        0.5
    """
    return replace(params or DEFAULT_PARAMS, **changes)
