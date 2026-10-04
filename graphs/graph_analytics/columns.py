"""Имена столбцов в выгрузке убытков."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LossColumns:
    """Соответствие логических полей и столбцов источника."""

    loss_id: str = "Убыток"
    culprit_fio: str = "Убыток виновник полннаименование"
    culprit_bd: str = "Убыток виновник датарождения"
    victim_fio: str = "Потерпевший.Полн наименование"
    victim_bd: str = "Потерпевший.Дата рождения"
    policyholder_fio: str = "Страхователь.Полн наименование"
    policyholder_bd: str = "Страхователь.Дата рождения"
    culprit_owner_fio: str = "Виновник.Полн наименование"
    culprit_owner_bd: str = "Виновник.Дата рождения"
    victim_owner_fio: str = "Потерпевший объект владелец.Полн наименование"
    victim_owner_bd: str = "Потерпевший объект владелец.Дата рождения"
    policyholder_owner_fio: str = "Страхователь объект владелец.Полн наименование"
    policyholder_owner_bd: str = "Страхователь объект владелец.Дата рождения"
    driver_fio: str = "Лицо управлявшее ТС.Полн наименование"
    driver_bd: str = "Лицо управлявшее ТС.Дата рождения"
    recipient_fio: str = "Получатель возмещения.Полн наименование"
    recipient_bd: str = "Получатель возмещения.Дата рождения"
    culprit_vin: str = "Объект виновника.Идентификатор"
    victim_vin: str = "Объект потерпевшего.Идентификатор"
    branch: str = "Филиалурегулировщик"
    decision: str = "Решение"
    amount: str = "Сумма"
    victim_policyholder_fio: str = "Потерпевший страхователь.Полн наименование"
    victim_policyholder_bd: str = "Потерпевший страхователь.Дата рождения"

    def required_source_columns(self) -> list[str]:
        """Столбцы, обязательные в исходных данных."""
        return [
            self.loss_id,
            self.culprit_fio,
            self.culprit_bd,
            self.victim_fio,
            self.victim_bd,
            self.policyholder_fio,
            self.policyholder_bd,
            self.culprit_owner_fio,
            self.culprit_owner_bd,
            self.victim_owner_fio,
            self.victim_owner_bd,
            self.policyholder_owner_fio,
            self.policyholder_owner_bd,
            self.driver_fio,
            self.driver_bd,
            self.recipient_fio,
            self.recipient_bd,
            self.culprit_vin,
            self.victim_vin,
            self.branch,
            self.decision,
            self.amount,
            self.victim_policyholder_fio,
            self.victim_policyholder_bd,
        ]

    def person_pairs(self) -> list[tuple[str, str]]:
        """Пары (ФИО, дата рождения) для извлечения участников."""
        return [
            (self.culprit_fio, self.culprit_bd),
            (self.victim_fio, self.victim_bd),
            (self.policyholder_fio, self.policyholder_bd),
            (self.culprit_owner_fio, self.culprit_owner_bd),
            (self.victim_owner_fio, self.victim_owner_bd),
            (self.policyholder_owner_fio, self.policyholder_owner_bd),
            (self.driver_fio, self.driver_bd),
            (self.recipient_fio, self.recipient_bd),
            (self.victim_policyholder_fio, self.victim_policyholder_bd),
        ]

    def link_fields(self) -> list[str]:
        """Имена вычисляемых полей для построения связей."""
        return [
            "Victim",
            "Culprit",
            "CulpritPolicyholder",
            "VictimObjectOwner",
            "CulpritObjectOwner",
            "PolicyholderObjectOwner",
            "PolicyholderDriver",
            "PaymentRecipient",
            "VINv",
            "VINc",
            "VictimPolicyholder",
        ]
