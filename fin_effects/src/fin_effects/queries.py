"""SQL-запросы к OISUU (SQL Server).

Запросы вынесены из тетради и приведены к состоянию, при котором результат не
зависит от плана выполнения:

* фильтр ``RefundForm NOT IN (...)`` перенесён в ``ON`` джойна платежей. В исходном
  виде он стоял в ``WHERE`` у ``LEFT JOIN`` и, если у убытка есть платёж с
  ``RefundForm IS NULL``, убыток целиком выпадал из выгрузки (``NULL NOT IN (...)
  -> UNKNOWN``);
* агрегаты по ПСР/претензиям берутся из подзапроса с ``GROUP BY``, а не джойнятся
  построчно (иначе суммы размножаются по числу строк платежей/претензий);
* платежи агрегируются до одного значения на убыток (``SUM(ValueRub)``), чтобы
  ``ValueRub`` не дублировался при нескольких выплатах по одному убытку.
"""

from __future__ import annotations

#: Фильтры, отсекающие судебные формы возмещения.
REFUND_FORM_EXCLUSIONS = ("Судебная ИСК", "Судебная ФУ")

LOSSES_QUERY = """
WITH payment_agg AS (
    SELECT
        pay.LossNumber,
        SUM(pay.ValueRub) AS ValueRubPayments
    FROM [OISUU_report].[dbo].[oisuu81_t_Payments] AS pay
    GROUP BY pay.LossNumber
),
psr_agg AS (
    SELECT
        psr.[Номер_убытка] AS LossNumber,
        SUM(psr.[Сумма_выплат_по_претензиям]) AS [Сумма_выплат_по_претензиям],
        SUM(psr.[Сумма_взыскано_по_ФУ]) AS [Сумма_взыскано_по_ФУ],
        SUM(psr.[Суммы_взыскано_по_иску]) AS [Суммы_взыскано_по_иску]
    FROM [OISUU_report].[Datamart].[oisuu81_t_ПСР] AS psr
    GROUP BY psr.[Номер_убытка]
)
SELECT
    ich.[IncidentNumber],
    l.[LossNumber],
    l.[LossDateTime],

    P.[PersonFullName] AS [Убыток виновник полннаименование],
    CAST(P.[PersonBirthDate] AS date) AS [Убыток виновник датарождения],

    P1.[PersonFullName] AS [Потерпевший.Полн наименование],
    CAST(P1.[PersonBirthDate] AS date) AS [Потерпевший.Дата рождения],

    P2.[PersonFullName] AS [Страхователь.Полн наименование],
    CAST(P2.[PersonBirthDate] AS date) AS [Страхователь.Дата рождения],

    P3.[PersonFullName] AS [Виновник.Полн наименование],
    CAST(P3.[PersonBirthDate] AS date) AS [Виновник.Дата рождения],

    P4.[PersonFullName] AS [Потерпевший объект владелец.Полн наименование],
    CAST(P4.[PersonBirthDate] AS date) AS [Потерпевший объект владелец.Дата рождения],

    P5.[PersonFullName] AS [Страхователь объект владелец.Полн наименование],
    CAST(P5.[PersonBirthDate] AS date) AS [Страхователь объект владелец.Дата рождения],

    P6.[PersonFullName] AS [Лицо управлявшее ТС.Полн наименование],
    CAST(P6.[PersonBirthDate] AS date) AS [Лицо управлявшее ТС.Дата рождения],

    P7.[PersonFullName] AS [Получатель возмещения.Полн наименование],
    CAST(P7.[PersonBirthDate] AS date) AS [Получатель возмещения.Дата рождения],

    P8.[PersonFullName] AS [Потерпевший страхователь.Полн наименование],
    CAST(P8.[PersonBirthDate] AS date) AS [Потерпевший страхователь.Дата рождения],

    l.[GuiltyObjectVIN] AS [Объект виновника.Идентификатор],
    l.[VictimObjectVIN] AS [Объект потерпевшего.Идентификатор],
    l.[VictimObjectVIN] AS [Убыток.Объект страхователя.Идентификатор],

    pay.ValueRubPayments AS ValueRub,

    psr.[Сумма_выплат_по_претензиям],
    psr.[Сумма_взыскано_по_ФУ],
    psr.[Суммы_взыскано_по_иску]
FROM [OISUU_report].[dbo].[oisuu81_t_Losses] AS l
LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_IncidentCharacteristics] AS ich
       ON ich.LossID = l.LossID
LEFT JOIN psr_agg AS psr
       ON psr.LossNumber = l.[LossNumber]
LEFT JOIN payment_agg AS pay
       ON pay.LossNumber = l.[LossNumber]
LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_Persons] AS P
       ON P.[PersonID] = l.[GuiltyPersonID]
LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_Persons] AS P1
       ON P1.[PersonID] = l.[VictimPersonID]
LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_Persons] AS P2
       ON P2.[PersonID] = l.[PolicyholderPersonID]
LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_Persons] AS P3
       ON P3.[PersonID] = l.[GuiltyObjectOwnerPersonID]
LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_Persons] AS P4
       ON P4.[PersonID] = l.[VictimObjectOwnerPersonID]
LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_Persons] AS P5
       ON P5.[PersonID] = l.[PolicyholderObjectOwnerPersonID]
LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_Persons] AS P6
       ON P6.[PersonID] = l.[DriverPersonID]
LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_Persons] AS P7
       ON P7.[PersonID] = l.[PaymentRecipientPersonID]
LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_Persons] AS P8
       ON P8.[PersonID] = l.[VictimPolicyholderPersonID]
WHERE l.[LossProcess] = 'Централизованное КАСКО'
  AND (l.[RefundForm] IS NULL OR l.[RefundForm] NOT IN ('Судебная ИСК', 'Судебная ФУ'))
"""


CARDS_QUERY = """
SELECT
    [КарточкаУПС]._Number AS [Номер карточки УПС],
    [Убыток]._Number AS [Номер Убытка],
    [КарточкаУПС]._Date_Time AS [Дата создания карточки],
    [ЗаключениеУУПС]._Date_Time AS [Дата закрытия карточки (дата заключения)],
    [ПеречислениеСтадия].EnumValueSynonym AS [Стадия],
    [ПеречислениеФинансовыйРезультат].EnumValueSynonym AS [Финансовый результат],
    [КарточкаУПС]._Fld12850 AS [Финансовый результат (денежный)],
    [КарточкаУПС]._Fld12851 AS [Предотвращенный ущерб],
    IIF([КарточкаУПС]._Marked = 0, 0, 1) AS [Пометка на удаление (карточка)]
FROM oisuu81.dbo._Document12802 AS [КарточкаУПС]
LEFT JOIN oisuu81.dbo._Document39 AS [Убыток]
       ON [КарточкаУПС]._Fld12859RRef = [Убыток]._IDRRef
LEFT JOIN OISUU_report.dbo.EnumList AS [ПеречислениеСтадия]
       ON [КарточкаУПС]._Fld12840RRef = [ПеречислениеСтадия].EnumId
LEFT JOIN OISUU_report.dbo.EnumList AS [ПеречислениеФинансовыйРезультат]
       ON [КарточкаУПС]._Fld12849RRef = [ПеречислениеФинансовыйРезультат].EnumId
LEFT JOIN oisuu81.dbo._Document12803 AS [ЗаключениеУУПС]
       ON [КарточкаУПС]._IDRRef = [ЗаключениеУУПС]._Fld12907RRef
"""
