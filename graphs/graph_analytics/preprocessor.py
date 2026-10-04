"""Предобработка таблицы убытков и формирование объектов графа."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from graph_analytics.columns import LossColumns
from graph_analytics.config import Settings
from graph_analytics.state import PipelineState


class DataPreprocessor:
    """Нормализация данных и подготовка узлов графа."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cols = LossColumns()

    def run(self, state: PipelineState, raw: pd.DataFrame) -> None:
        """Полный цикл предобработки."""
        data = raw.copy()
        self._strip_column_names(data)
        self._validate_columns(data)
        self._deduplicate(data)
        self._clean_values(data)
        self._cast_dates(data)
        self._add_link_columns(data)

        people = self._build_people(data)
        vin = self._build_vin(data)
        objects = pd.concat([people, vin], ignore_index=True)
        objects = objects[~objects.isnull()]
        objects = objects.drop_duplicates()
        objects = objects.reset_index(drop=True)

        state.columns = self._cols
        state.data = data
        state.people = people
        state.vin = vin
        state.objects = objects

    def _strip_column_names(self, data: pd.DataFrame) -> None:
        data.rename(columns={col: col.strip() for col in data.columns}, inplace=True)

    def _validate_columns(self, data: pd.DataFrame) -> None:
        missing = [c for c in self._cols.required_source_columns() if c not in data.columns]
        if missing:
            raise ValueError(f"В данных отсутствуют столбцы: {', '.join(missing)}")

    def _deduplicate(self, data: pd.DataFrame) -> None:
        data[self._cols.loss_id] = data[self._cols.loss_id].astype(int)
        if data[[self._cols.loss_id]].duplicated().any():
            data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)

    def _clean_values(self, data: pd.DataFrame) -> None:
        text_cols = [
            self._cols.culprit_fio,
            self._cols.victim_fio,
            self._cols.policyholder_fio,
            self._cols.culprit_owner_fio,
            self._cols.victim_owner_fio,
            self._cols.policyholder_owner_fio,
            self._cols.driver_fio,
            self._cols.recipient_fio,
            self._cols.victim_policyholder_fio,
            self._cols.culprit_vin,
            self._cols.victim_vin,
        ]
        for column in text_cols:
            data[column] = data[column].astype(str).str.upper()
            for bad in self._settings.wrong_people_values:
                data[column] = data[column].replace(bad, np.nan)

        data["Reject"] = (
            data[self._cols.decision]
            .fillna(0)
            .replace(
                {
                    "Отказ": 1,
                    "Отказать": 1,
                    "Выплата": 0,
                    "Выплатить": 0,
                    "Не урегулирован": 1,
                }
            )
        )

        invalid_amount = data[self._cols.amount].apply(type) == datetime.datetime
        data.drop(data.index[invalid_amount], inplace=True)
        data[self._cols.amount] = data[self._cols.amount].astype(float)
        data.reset_index(drop=True, inplace=True)

        for column in (self._cols.culprit_vin, self._cols.victim_vin):
            if data[column].isin(self._settings.wrong_auto_values.keys()).any():
                data[column] = data[column].replace(self._settings.wrong_auto_values)

    def _cast_dates(self, data: pd.DataFrame) -> None:
        date_cols = [bd for _, bd in self._cols.person_pairs()]
        for column in date_cols:
            data[column] = (
                pd.to_datetime(data[column], errors="coerce").astype(str).replace("NaT", np.nan)
            )

    def _add_link_columns(self, data: pd.DataFrame) -> None:
        c = self._cols
        data["Victim"] = data[c.victim_fio] + " " + data[c.victim_bd].astype(str)
        data["Culprit"] = data[c.culprit_fio] + " " + data[c.culprit_bd].astype(str)
        data["CulpritPolicyholder"] = data[c.policyholder_fio] + " " + data[c.policyholder_bd].astype(str)
        data["VictimObjectOwner"] = data[c.victim_owner_fio] + " " + data[c.victim_owner_bd].astype(str)
        data["CulpritObjectOwner"] = data[c.culprit_owner_fio] + " " + data[c.culprit_owner_bd].astype(str)
        data["PolicyholderObjectOwner"] = (
            data[c.policyholder_owner_fio] + " " + data[c.policyholder_owner_bd].astype(str)
        )
        data["PolicyholderDriver"] = data[c.driver_fio] + " " + data[c.driver_bd].astype(str)
        data["PaymentRecipient"] = data[c.recipient_fio] + " " + data[c.recipient_bd].astype(str)
        data["VINv"] = data[c.victim_vin]
        data["VINc"] = data[c.culprit_vin]
        data["VictimPolicyholder"] = (
            data[c.victim_policyholder_fio] + " " + data[c.victim_policyholder_bd].astype(str)
        )

    def _build_people(self, data: pd.DataFrame) -> pd.Series:
        parts: list[pd.Series] = []
        for fio_col, bd_col in self._cols.person_pairs():
            label = data[fio_col].astype(str).str.strip() + " " + data[bd_col].astype(str).str.strip()
            parts.append(label)
        people = pd.concat(parts, ignore_index=True)
        return people[~people.isnull() & (people.str.strip() != "")].drop_duplicates().reset_index(drop=True)

    def _build_vin(self, data: pd.DataFrame) -> pd.Series:
        c = self._cols
        vin = pd.concat([data[c.victim_vin], data[c.culprit_vin]], ignore_index=True)
        vin = vin.drop_duplicates()
        vin = vin[~vin.isnull() & (vin != "")]
        return vin.reset_index(drop=True)
