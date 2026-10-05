"""Генератор синтетических данных для прогонки пайплайна.

Модуль создаёт полный комплект сырья, повторяющий структуру продовых выгрузок:

* ``data/synthetic/raw/<last_year>/groups/*.xlsx`` — выгрузка сервиса: один файл =
  одна группа, на листе участники (``ФИО | Дата рождения | Номер группы``);
* ``data/synthetic/raw/losses.pkl`` — выгрузка убытков КАСКО со всеми
  идентификаторами физлиц/объектов и денежными колонками;
* ``data/synthetic/raw/cards.pkl`` — карточки УПС (дата обнаружения группы);
* ``data/synthetic/raw/payments.pkl`` — платежи по убыткам (для проверки эффекта
  размножения строк при ``JOIN``);
* ``data/synthetic/groups_truth.csv`` — «эталон»: сценарий группы, её годы,
  год обнаружения и ожидаемый класс релевантности (по нему тестируется методика).

Сценарии заложены так, чтобы в данных встречались все ветки методики:

======================  ==========================================================
Сценарий                Что проверяет
======================  ==========================================================
``alive``               группа работает после обнаружения -> эффект в горизонте
``died_after_detection`` группа «накрыта» сервисом в последний год жизни -> 1 год
``died_before_detection`` группа исчезла до обнаружения -> эффект 0
``no_card``             нет карточки УПС -> fallback даты обнаружения
``single_loss``         группа из одного убытка -> отсечение порогом `min_losses_total`
``noise``               убытки вне групп
``duplicate_payments``  несколько платежей/ПСР на убыток -> проверка дедупликации
======================  ==========================================================
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    PERSON_DATE_PAIRS,
    RELEVANCE_CLOSED,
    RELEVANCE_IN_PROGRESS,
    SYNTH_DIR,
)
from .io_utils import write_pickle
from .loaders import CARD_LOSS_KEY
from .preprocess import LOSS_DATE, LOSS_KEY
from .textnorm import format_person_key

LOGGER = logging.getLogger(__name__)

#: Сценарии поведения синтетической группы.
SCENARIOS = (
    "alive",
    "died_after_detection",
    "died_before_detection",
    "no_card",
    "single_loss",
)

_SURNAMES = (
    "Иванов", "Петров", "Сидоров", "Кузнецов", "Смирнов", "Попов", "Волков", "Морозов",
    "Новиков", "Фёдоров", "Егоров", "Павлов", "Козлов", "Степанов", "Николаев", "Орлов",
    "Андреев", "Макаров", "Никитин", "Захаров", "Зайцев", "Соловьёв", "Борисов", "Яковлев",
    "Григорьев", "Романов", "Воробьёв", "Сергеев", "Кузьмин", "Фролов", "Александров",
    "Дмитриев", "Королёв", "Гусев", "Киселёв", "Ильин", "Максимов", "Поляков", "Сорокин",
)
_NAMES_MALE = (
    "Александр", "Сергей", "Дмитрий", "Андрей", "Алексей", "Максим", "Иван", "Артём",
    "Николай", "Михаил", "Роман", "Егор", "Павел", "Владимир", "Денис", "Тимур",
)
_NAMES_FEMALE = (
    "Анна", "Мария", "Елена", "Ольга", "Наталья", "Ирина", "Татьяна", "Светлана",
    "Екатерина", "Юлия", "Ксения", "Дарья", "Людмила", "Галина", "Вера", "Полина",
)
_PATRONYMICS = (
    "Иванович", "Петрович", "Сергеевич", "Андреевич", "Алексеевич", "Николаевич",
    "Михайлович", "Владимирович", "Дмитриевич", "Юрьевич", "Ивановна", "Петровна",
    "Сергеевна", "Андреевна", "Алексеевна", "Николаевна", "Михайловна", "Владимировна",
)
_VIN_PREFIXES = ("XTA", "XWB", "WVW", "ZFA", "JHM", "KMH", "LJD", "SJN")
_VIN_ALPHABET = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"

_OPTIONAL_PERSON_ROLES = (
    "Страхователь.Полн наименование",
    "Потерпевший объект владелец.Полн наименование",
    "Страхователь объект владелец.Полн наименование",
    "Лицо управлявшее ТС.Полн наименование",
    "Получатель возмещения.Полн наименование",
    "Потерпевший страхователь.Полн наименование",
)

#: Пары «колонка ФИО -> колонка даты рождения» (все роли из выгрузки).
_PERSON_ROLE_PAIRS: tuple[tuple[str, str], ...] = tuple(PERSON_DATE_PAIRS.items())


@dataclass
class SynthConfig:
    """Параметры генерации синтетических данных.

    Attributes:
        seed: зерно генератора.
        groups: число «мошеннических» групп.
        members_min: минимум участников в группе.
        members_max: максимум участников в группе.
        losses_per_group_year: диапазон числа убытков группы в активный год.
        active_years: диапазон длительности жизни группы, лет.
        noise_losses: число убытков вне групп.
        year_range: диапазон годов убытков (включительно).
        payments_per_loss: сколько платежей на убыток вешать в «грязных» убытках.
        duplicate_losses: доля убытков с дублирующими платежами/ПСР.
        payout_range: диапазон выплаты по убытку, ₽.
        recovery_probability: вероятность взыскания по убытку.
        output_dir: каталог вывода.
        write_workbooks: писать xlsx групп (медленно); ``False`` — только таблицы
            в памяти (достаточно для пайплайна/отчёта).
        scenarios: распределение сценариев (имя -> вес).
    """

    seed: int = 20240517
    groups: int = 48
    members_min: int = 4
    members_max: int = 9
    losses_per_group_year: tuple[int, int] = (4, 11)
    active_years: tuple[int, int] = (1, 3)
    noise_losses: int = 400
    year_range: tuple[int, int] = (2020, 2025)
    payments_per_loss: tuple[int, int] = (1, 3)
    duplicate_losses: float = 0.25
    payout_range: tuple[int, int] = (45_000, 480_000)
    recovery_probability: float = 0.18
    output_dir: Path = field(default_factory=lambda: SYNTH_DIR)
    write_workbooks: bool = True
    scenarios: dict[str, float] = field(
        default_factory=lambda: {
            "alive": 0.45,
            "died_after_detection": 0.20,
            "died_before_detection": 0.15,
            "no_card": 0.10,
            "single_loss": 0.10,
        }
    )


@dataclass(frozen=True)
class Member:
    """Участник группы.

    Attributes:
        full_name: ФИО.
        birth_date: дата рождения.
    """

    full_name: str
    birth_date: pd.Timestamp

    @property
    def person_key(self) -> str:
        """Ключ сопоставления «ФИО + ДР».

        Returns:
            Нормализованный ключ.
        """
        return format_person_key(self.full_name, self.birth_date)


@dataclass
class GroupPlan:
    """План одной синтетической группы.

    Attributes:
        group_id: идентификатор группы (= имя файла выгрузки).
        group_number: номер группы для выгрузки сервиса.
        scenario: сценарий поведения.
        members: участники группы.
        loss_years: годы, в которых группа совершала убытки.
        has_card: есть ли карточка УПС.
        card_year: год создания карточки (``None`` для ``has_card=False``).
    """

    group_id: str
    group_number: int
    scenario: str
    members: list[Member]
    loss_years: list[int]
    has_card: bool
    card_year: int | None

    @property
    def observed_detection_year(self) -> int | None:
        """Год обнаружения по карточке УПС; ``None``, если карточки нет."""
        if self.has_card and self.card_year is not None:
            return int(self.card_year)
        return None

    @property
    def card_stage(self) -> str | None:
        """Стадия карточки УПС для эталона классификации."""
        if not self.has_card:
            return None
        # «в работе» — сценарии с продолжающейся жизнью группы; остальное — закрыта
        if self.scenario in {"alive", "no_card"}:
            # no_card не имеет стадии; alive — в работе
            return RELEVANCE_IN_PROGRESS if self.scenario == "alive" else None
        return RELEVANCE_CLOSED

    @property
    def expected_relevance(self) -> str | None:
        """Ожидаемый статус по карточке УПС; ``None`` без карточки."""
        return self.card_stage


@dataclass
class SynthResult:
    """Результат генерации синтетических данных."""

    beneficiaries: pd.DataFrame
    losses: pd.DataFrame
    cards: pd.DataFrame
    payments: pd.DataFrame
    groups: pd.DataFrame
    beneficiary_files: list[Path]
    output_dir: Path

    def summary(self) -> str:
        """Текстовая сводка по сгенерированным данным.

        Returns:
            Многострочная строка для логов.
        """
        duplicate_rows = int(self.losses[LOSS_KEY].duplicated().sum())
        return "\n".join(
            [
                f"Групп: {self.groups['group_id'].nunique()}",
                f"Участников в выгрузке сервиса: {len(self.beneficiaries)}",
                f"Строк убытков в выгрузке: {len(self.losses)} "
                f"(уникальных убытков: {self.losses[LOSS_KEY].nunique()}, "
                f"строк-дублей: {duplicate_rows})",
                f"Платежей: {len(self.payments)}",
                f"Карточек УПС: {len(self.cards)}",
                f"Файлов выгрузки сервиса: {len(self.beneficiary_files)}",
                f"Каталог: {self.output_dir}",
            ]
        )


# ---------------------------------------------------------------------------
# Случайные сущности
# ---------------------------------------------------------------------------


def _random_full_name(rng: random.Random, sex: str) -> str:
    """Собирает случайное ФИО.

    Args:
        rng: генератор случайных чисел.
        sex: ``"m"`` или ``"f"``.

    Returns:
        ФИО строкой.
    """
    surname = rng.choice(_SURNAMES) + ("а" if sex == "f" else "")
    name = rng.choice(_NAMES_FEMALE if sex == "f" else _NAMES_MALE)
    patronymic = rng.choice(_PATRONYMICS)
    if sex == "f" and patronymic.endswith("вич"):
        patronymic = patronymic[:-1] + "на"
    return f"{surname} {name} {patronymic}"


def _random_vin(rng: random.Random) -> str:
    """Собирает случайный VIN.

    Args:
        rng: генератор случайных чисел.

    Returns:
        Строка VIN (17 символов).
    """
    body = "".join(rng.choice(_VIN_ALPHABET) for _ in range(14))
    return rng.choice(_VIN_PREFIXES) + body


def _random_birth_date(rng: random.Random) -> pd.Timestamp:
    """Случайная дата рождения (1955-2003).

    Args:
        rng: генератор случайных чисел.

    Returns:
        Дата рождения.
    """
    return pd.Timestamp(year=rng.randint(1955, 2003), month=rng.randint(1, 12), day=rng.randint(1, 28))


def _weighted_scenario(rng: random.Random, scenarios: dict[str, float]) -> str:
    """Выбирает сценарий группы по весам.

    Args:
        rng: генератор случайных чисел.
        scenarios: распределение сценариев.

    Returns:
        Имя сценария.
    """
    names = list(scenarios)
    weights = np.array([scenarios[name] for name in names], dtype=float)
    weights = weights / weights.sum()
    return str(rng.choices(names, weights=weights, k=1)[0])


# ---------------------------------------------------------------------------
# Планирование
# ---------------------------------------------------------------------------


def build_group_plan(config: SynthConfig, rng: random.Random, group_index: int) -> GroupPlan:
    """Планирует одну группу: участников, годы убытков, карточку УПС.

    Args:
        config: параметры генерации.
        rng: генератор случайных чисел.
        group_index: порядковый номер группы.

    Returns:
        :class:`GroupPlan`.

    Note:
        Год обнаружения задаётся так, чтобы получить все ветки методики:

        * ``alive`` — карточка в год **первого** убытка, убытки продолжаются;
        * ``died_after_detection`` — карточка в год **последнего** убытка;
        * ``died_before_detection`` — карточка через 1-2 года после последнего
          убытка (группа распалась сама);
        * ``no_card`` — карточки нет, дата обнаружения берётся как год первого
          убытка (fallback), группа «жива» до конца периода.
    """
    scenario = _weighted_scenario(rng, config.scenarios)

    members_count = rng.randint(config.members_min, config.members_max)
    if scenario == "single_loss":
        members_count = max(2, config.members_min - 1)
    members = _generate_members(rng, members_count)

    min_year, max_year = config.year_range
    first_year = rng.randint(min_year, max_year - 2)
    lifetime = rng.randint(*config.active_years)
    last_year = first_year + lifetime - 1
    if last_year > max_year:
        last_year = max_year
        first_year = max(min_year, last_year - lifetime + 1)
    loss_years = list(range(first_year, last_year + 1))

    if scenario == "single_loss":
        loss_years = [last_year]
        card_year = last_year
    elif scenario == "died_before_detection":
        # Карточка создаётся через год-два после последнего убытка: группа
        # распалась сама, эффект должен быть нулевым. Год обнаружения не выходит
        # за пределы периода наблюдения.
        card_year = min(max_year, last_year + rng.randint(1, 2))
    elif scenario == "no_card":
        card_year = None
    elif scenario == "alive":
        # Карточка в первый год жизни группы, убытки продолжаются после неё.
        card_year = first_year
    else:  # died_after_detection
        card_year = last_year

    return GroupPlan(
        group_id=f"grp_{group_index:04d}",
        group_number=group_index + 1,
        scenario=scenario,
        members=members,
        loss_years=loss_years,
        has_card=card_year is not None,
        card_year=card_year,
    )


def _generate_members(rng: random.Random, count: int) -> list[Member]:
    """Генерирует уникальных участников группы.

    Args:
        rng: генератор случайных чисел.
        count: сколько участников нужно.

    Returns:
        Список участников (уникальных по ФИО+ДР).
    """
    members: dict[str, Member] = {}
    attempts = 0
    while len(members) < count and attempts < count * 20:
        attempts += 1
        member = Member(
            full_name=_random_full_name(rng, rng.choice("mf")),
            birth_date=_random_birth_date(rng),
        )
        members.setdefault(member.person_key, member)
    return list(members.values())


# ---------------------------------------------------------------------------
# Генерация убытков
# ---------------------------------------------------------------------------


def _random_vin_pool(rng: random.Random, size: int = 3) -> list[str]:
    """Пул VIN для одного убытка.

    Args:
        rng: генератор случайных чисел.
        size: сколько VIN нужно.

    Returns:
        Список VIN.
    """
    return [_random_vin(rng) for _ in range(size)]


def _loss_dates_in_year(rng: random.Random, year: int, count: int) -> list[pd.Timestamp]:
    """Раскидывает убытки по дням года.

    Args:
        rng: генератор случайных чисел.
        year: год.
        count: число убытков.

    Returns:
        Отсортированный список дат.
    """
    days = sorted(rng.sample(range(0, 360), k=min(count, 360)))
    start = pd.Timestamp(year=year, month=1, day=1)
    return [start + pd.Timedelta(days=int(day)) for day in days]


def _payout(config: SynthConfig, rng: random.Random) -> float:
    """Сумма выплаты по убытку.

    Args:
        config: параметры генерации.
        rng: генератор случайных чисел.

    Returns:
        Сумма в рублях (округлённая до сотен).
    """
    low, high = config.payout_range
    return float(round(rng.uniform(low, high), -2))


def _build_loss_record(
    config: SynthConfig,
    rng: random.Random,
    loss_number: int,
    loss_date: pd.Timestamp,
    participants: list[Member],
    plan: GroupPlan | None,
) -> dict[str, object]:
    """Формирует строку выгрузки убытков.

    Args:
        config: параметры генерации.
        rng: генератор случайных чисел.
        loss_number: номер убытка.
        loss_date: дата убытка.
        participants: участники убытка.
        plan: план группы (``None`` для «обычного» убытка).

    Returns:
        Словарь-строка выгрузки (по одной строке на платёж).
    """
    guilty = participants[0] if participants else None
    victim = participants[1] if len(participants) > 1 else None
    vins = _random_vin_pool(rng, 3)

    record: dict[str, object] = {
        LOSS_KEY: f"L{loss_number:08d}",
        LOSS_DATE: loss_date,
        "IncidentNumber": f"INC{loss_number:08d}",
        "ValueRub": _payout(config, rng),
        "Сумма_выплат_по_претензиям": 0.0,
        "Сумма_взыскано_по_ФУ": 0.0,
        "Суммы_взыскано_по_иску": 0.0,
        "Объект виновника.Идентификатор": vins[0],
        "Объект потерпевшего.Идентификатор": vins[1],
        "Убыток.Объект страхователя.Идентификатор": vins[2],
        "RefundForm": rng.choice(["Прямое возмещение", "Возмещение", None]),
        "LossProcess": "Централизованное КАСКО",
        "_synthetic_group": plan.group_id if plan else None,
        "_synthetic_scenario": plan.scenario if plan else "noise",
    }

    for column, date_column in _PERSON_ROLE_PAIRS:
        record[column] = None
        record[date_column] = pd.NaT

    if guilty is not None:
        record["Виновник.Полн наименование"] = guilty.full_name
        record["Виновник.Дата рождения"] = guilty.birth_date
    if victim is not None:
        record["Потерпевший.Полн наименование"] = victim.full_name
        record["Потерпевший.Дата рождения"] = victim.birth_date

    # Часть ролей заполняется «как в жизни»: не всегда и не всегда тем же лицом.
    for role in _OPTIONAL_PERSON_ROLES:
        if participants and rng.random() < 0.35:
            member = rng.choice(participants)
            record[role] = member.full_name
            record[PERSON_DATE_PAIRS[role]] = member.birth_date

    if rng.random() < config.recovery_probability:
        record["Сумма_взыскано_по_ФУ"] = float(
            round(float(record["ValueRub"]) * rng.uniform(0.1, 0.4), -2)
        )
    if rng.random() < 0.12:
        record["Сумма_выплат_по_претензиям"] = float(
            round(float(record["ValueRub"]) * rng.uniform(0.05, 0.3), -2)
        )
    return record


def _split_amount(total: float, parts: int) -> list[float]:
    """Делит сумму на части (последняя забирает остаток).

    Args:
        total: сумма.
        parts: число частей.

    Returns:
        Список сумм, в точности дающий ``total``.
    """
    if parts <= 1:
        return [total]
    share = round(total / parts, 2)
    result = [share] * (parts - 1)
    result.append(round(total - sum(result), 2))
    return result


def expand_losses_to_payment_rows(
    losses: pd.DataFrame,
    payments: list[dict[str, object]],
    config: SynthConfig,
    rng: random.Random,
) -> pd.DataFrame:
    """Размножает часть убытков построчно и собирает платежи.

    Args:
        losses: сгенерированные убытки (строка на убыток).
        payments: наполняемый список платежей.
        config: параметры генерации.
        rng: генератор случайных чисел.

    Returns:
        Таблица убытков, в которой часть убытков встречается несколько раз —
        имитация ``LEFT JOIN`` к таблице платежей и ПСР.

    Note:
        Дубликаты нужны, чтобы поймать регрессию: если денежные колонки
        суммируются без свёртки до уровня убытка, суммы завышаются кратно числу
        платежей. Проверка — в тестах и в блоке «как было / как стало».
    """
    duplicated_frames: list[dict[str, object]] = []
    claims_mask: list[bool] = []

    for record in losses.to_dict("records"):
        payments_count = (
            rng.randint(*config.payments_per_loss) if rng.random() < config.duplicate_losses else 1
        )
        total_payout = float(record["ValueRub"] or 0.0)
        shares = _split_amount(total_payout, payments_count)

        for position, share in enumerate(shares):
            payments.append(
                {
                    LOSS_KEY: record[LOSS_KEY],
                    "ValueRub": share,
                    "PaymentNumber": f"{record[LOSS_KEY]}-{position + 1}",
                }
            )

        for position, share in enumerate(shares):
            row = dict(record)
            row["ValueRub"] = share
            row["_payment_index"] = position
            # Претензии приходят из отдельного джойна: только на первой строке.
            row["Сумма_выплат_по_претензиям"] = (
                float(record["Сумма_выплат_по_претензиям"] or 0.0) if position == 0 else 0.0
            )
            row["Сумма_взыскано_по_ФУ"] = (
                float(record["Сумма_взыскано_по_ФУ"] or 0.0) if position == 0 else 0.0
            )
            row["Суммы_взыскано_по_иску"] = (
                float(record["Суммы_взыскано_по_иску"] or 0.0) if position == 0 else 0.0
            )
            duplicated_frames.append(row)
            claims_mask.append(position == 0)

    expanded = pd.DataFrame(duplicated_frames)
    expanded["_is_claims_row"] = claims_mask
    return expanded


def generate_beneficiary_workbooks(config: SynthConfig, plans: list[GroupPlan]) -> list[Path]:
    """Пишет xlsx-файлы выгрузки сервиса (один файл = одна группа).

    Args:
        config: параметры генерации.
        plans: планы групп.

    Returns:
        Список путей к созданным файлам.
    """
    directory = config.output_dir / "raw" / str(config.year_range[1]) / "groups"
    directory.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for plan in plans:
        frame = pd.DataFrame(
            {
                "ФИО": [member.full_name for member in plan.members],
                "Дата рождения": [member.birth_date for member in plan.members],
                "Номер группы": [plan.group_number] * len(plan.members),
            }
        )
        path = directory / f"{plan.group_id}.xlsx"
        frame.to_excel(path, index=False, sheet_name=f"group_{plan.group_number}")
        paths.append(path)

    LOGGER.info("Создано файлов выгрузки сервиса: %d (%s)", len(paths), directory)
    return paths


def generate_synthetic_data(config: SynthConfig | None = None) -> SynthResult:
    """Генерирует полный синтетический набор данных.

    Args:
        config: параметры генерации.

    Returns:
        :class:`SynthResult` с таблицами и путями к файлам.
    """
    config = config or SynthConfig()
    rng = random.Random(config.seed)
    np.random.seed(config.seed)

    plans = [build_group_plan(config, rng, index) for index in range(config.groups)]

    loss_records: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    payments: list[dict[str, object]] = []
    cards: list[dict[str, object]] = []
    loss_number = 0

    for plan in plans:
        group_loss_numbers: list[str] = []
        for year in plan.loss_years:
            volume = 1 if plan.scenario == "single_loss" else rng.randint(*config.losses_per_group_year)
            for loss_date in _loss_dates_in_year(rng, year, volume):
                loss_number += 1
                participants_count = min(len(plan.members), rng.randint(1, 3))
                participants = rng.sample(plan.members, k=participants_count)
                record = _build_loss_record(
                    config, rng, loss_number, loss_date, participants, plan
                )
                loss_records.append(record)
                group_loss_numbers.append(str(record[LOSS_KEY]))

        if plan.has_card and group_loss_numbers and plan.card_year is not None:
            detection_date = pd.Timestamp(year=int(plan.card_year), month=3, day=15)
            stage = "В работе" if plan.card_stage == RELEVANCE_IN_PROGRESS else "Закрыта"
            cards.append(
                {
                    "Номер карточки УПС": f"UPS-{plan.group_number:06d}",
                    CARD_LOSS_KEY: group_loss_numbers[0],
                    "Дата создания карточки": detection_date,
                    "Дата закрытия карточки (дата заключения)": detection_date
                    + pd.Timedelta(days=rng.randint(10, 90)),
                    "Стадия": stage,
                    "Финансовый результат": rng.choice(["Мошенничество", "Отказ"]),
                    "Финансовый результат (денежный)": 0.0,
                    "Предотвращенный ущерб": 0.0,
                    "Пометка на удаление (карточка)": 0,
                }
            )

        group_rows.append(
            {
                "group_id": plan.group_id,
                "group_number": plan.group_number,
                "scenario": plan.scenario,
                "members": len(plan.members),
                "first_loss_year": min(plan.loss_years),
                "last_loss_year": max(plan.loss_years),
                "card_year": plan.card_year,
                "observed_detection_year": plan.observed_detection_year,
                "expected_relevance": plan.expected_relevance,
            }
        )

    # «Обычные» убытки вне групп: свой пул людей, чтобы не пересекаться с группами.
    for _ in range(config.noise_losses):
        loss_number += 1
        year = rng.randint(*config.year_range)
        loss_date = pd.Timestamp(year=year, month=rng.randint(1, 12), day=rng.randint(1, 28))
        participants = [
            Member(_random_full_name(rng, rng.choice("mf")), _random_birth_date(rng))
            for _ in range(rng.randint(1, 2))
        ]
        loss_records.append(_build_loss_record(config, rng, loss_number, loss_date, participants, None))

    raw_losses = pd.DataFrame(loss_records)
    losses = expand_losses_to_payment_rows(raw_losses, payments, config, rng)

    beneficiaries = _beneficiaries_frame(plans)
    cards_frame = pd.DataFrame(cards)
    payments_frame = pd.DataFrame(payments)
    groups_frame = pd.DataFrame(group_rows)

    raw_dir = config.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_pickle(losses, raw_dir / "losses.pkl")
    write_pickle(cards_frame, raw_dir / "cards.pkl")
    write_pickle(payments_frame, raw_dir / "payments.pkl")
    write_pickle(beneficiaries, raw_dir / "beneficiaries.pkl")
    groups_frame.to_csv(config.output_dir / "groups_truth.csv", index=False)

    files: list[Path] = []
    if config.write_workbooks:
        files = generate_beneficiary_workbooks(config, plans)

    result = SynthResult(
        beneficiaries=beneficiaries,
        losses=losses,
        cards=cards_frame,
        payments=payments_frame,
        groups=groups_frame,
        beneficiary_files=files,
        output_dir=config.output_dir,
    )
    LOGGER.info("Синтетические данные готовы:\n%s", result.summary())
    return result


def _beneficiaries_frame(plans: list[GroupPlan]) -> pd.DataFrame:
    """Собирает «эталонную» таблицу участников групп.

    Args:
        plans: планы групп.

    Returns:
        Таблица ``group_id``, ``full_name``, ``birth_date``, ``group_number``,
        ``person_key``.
    """
    rows: list[dict[str, object]] = []
    for plan in plans:
        for member in plan.members:
            rows.append(
                {
                    "group_id": plan.group_id,
                    "full_name": member.full_name,
                    "birth_date": member.birth_date,
                    "group_number": plan.group_number,
                    "person_key": member.person_key,
                }
            )
    return pd.DataFrame(rows)


def load_beneficiaries_truth(config: SynthConfig | None = None) -> pd.DataFrame:
    """Читает «эталонную» таблицу участников, сохранённую генератором.

    Args:
        config: параметры генерации.

    Returns:
        Таблица участников с колонкой ``person_key``.

    Raises:
        FileNotFoundError: если генератор ещё не запускался.
    """
    from .io_utils import read_pickle

    config = config or SynthConfig()
    path = config.output_dir / "raw" / "beneficiaries.pkl"
    return read_pickle(path)  # type: ignore[return-value]
