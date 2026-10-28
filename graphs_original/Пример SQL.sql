/*
  Выгрузка убытков без полной нагрузки на OISUU_report.
  Только SELECT + CTE: ничего не создаёт и не меняет в БД.

  1. Сначала узкий срез убытков (фильтр до джойнов к Persons).
  2. Persons читается один раз по нужным PersonID.
  3. PaymentOrders не джойнится: в выборке не используется и размножает строки.
  4. Платежи сворачиваются в одну сумму на убыток.
  NOLOCK — чтобы не держать shared-блокировки на отчётной базе.
*/

WITH losses AS (
    SELECT
           L.[LossID],
           CAST(L.[LossNumber] AS int) AS [Убыток],
           L.[InsuranceTypeName] AS [ВидУбытка],
           L.[LossUnit] AS [Филиал],
           L.[Filial] AS [Филиалурегулировщик],
           CAST(L.[EventDate] AS date) AS [ДатаСобытия],
           L.[EventLocationDescription] AS [МестоСобытия],
           L.[GuiltyPersonID],
           L.[VictimPersonID],
           L.[PolicyholderPersonID],
           L.[GuiltyObjectOwnerPersonID],
           L.[VictimObjectOwnerPersonID],
           L.[PolicyholderObjectOwnerPersonID],
           L.[DriverPersonID],
           L.[PaymentRecipientPersonID],
           L.[VIctimPolicyholderPersonID],
           L.[GuiltyObjectRegNum] AS [Объект виновника.Гос номер],
           L.[GuiltyObjectVIN] AS [Объект виновника.Идентификатор],
           L.[VictimObjectRegNum] AS [Объект потерпевшего.Гос номер],
           L.[VictimObjectVIN] AS [Объект потерпевшего.Идентификатор],
           L.[PolicyholderObjectRegNum] AS [Убыток.Объект страхователя.Гос номер],
           L.[VictimObjectVIN] AS [Убыток.Объект страхователя.Идентификатор],
           L.[LossStateByIA] AS [Решение],
           L.[RefundFormByPaymentOrder] AS [ВыплатыФормаВозмещения],
           L.[LossProcess] AS [СтадияПроцесс]
    FROM [OISUU_report].[dbo].[oisuu81_t_Losses] AS L WITH (NOLOCK)
    WHERE L.[Filial] IN ('ВСК-Москва', 'Головная')
      AND L.[EventDate] >= '20200101'
      AND L.[RefundForm] = 'Денежная'
      AND L.[LossProcess] IN (
            'Прямое ОСАГО (с 1 марта 2009)',
            'Традиционное ОСАГО',
            'Централизованное КАСКО',
            'Авто. Финансовые риски',
            'Прочие процессы урегулирования'
          )
      AND L.[EventLocationRegionName] IN ('г. Москва', 'обл. Московская')
),
pay AS (
    SELECT
           p.[LossID],
           SUM(CAST(p.[PaymentValue] AS float)) AS [Сумма]
    FROM [OISUU_report].[dbo].[oisuu81_t_Payments] AS p WITH (NOLOCK)
    INNER JOIN losses AS l ON l.[LossID] = p.[LossID]
    GROUP BY p.[LossID]
),
person_ids AS (
    SELECT PersonID
    FROM (
        SELECT [GuiltyPersonID] AS PersonID FROM losses
        UNION
        SELECT [VictimPersonID] FROM losses
        UNION
        SELECT [PolicyholderPersonID] FROM losses
        UNION
        SELECT [GuiltyObjectOwnerPersonID] FROM losses
        UNION
        SELECT [VictimObjectOwnerPersonID] FROM losses
        UNION
        SELECT [PolicyholderObjectOwnerPersonID] FROM losses
        UNION
        SELECT [DriverPersonID] FROM losses
        UNION
        SELECT [PaymentRecipientPersonID] FROM losses
        UNION
        SELECT [VIctimPolicyholderPersonID] FROM losses
    ) AS ids
    WHERE PersonID IS NOT NULL
),
persons AS (
    SELECT
           p.[PersonID],
           p.[PersonFullName],
           p.[PersonType],
           p.[PersonINN],
           CAST(p.[PersonBirthDate] AS date) AS [PersonBirthDate],
           p.[PersoneCode],
           p.[PhoneNumber],
           p.[PersonCity],
           p.[PersonSettlement]
    FROM [OISUU_report].[dbo].[oisuu81_t_Persons] AS p WITH (NOLOCK)
    INNER JOIN person_ids AS i ON i.PersonID = p.[PersonID]
)
SELECT
       l.[Убыток],
       l.[ВидУбытка],
       l.[Филиал],
       l.[Филиалурегулировщик],
       l.[ДатаСобытия],
       l.[МестоСобытия],

       P.[PersonFullName] AS [Убыток виновник полннаименование],
       P.[PersonType] AS [ВиновникТипЛица],
       P.[PersonINN] AS [ВиновникИНН],
       P.[PersonBirthDate] AS [Убыток виновник датарождения],
       P.[PersoneCode] AS [ВиновникКод],
       P.[PhoneNumber] AS [ВиновникНомерТелефона],
       P.[PersonCity] AS [ВиновникГород],
       P.[PersonSettlement] AS [ВиновникПоселок],

       P1.[PersonFullName] AS [Потерпевший.Полн наименование],
       P1.[PersonType] AS [ПотерпевшийТипЛица],
       P1.[PersonINN] AS [ПотерпевшийИНН],
       P1.[PersonBirthDate] AS [Потерпевший.Дата рождения],
       P1.[PersoneCode] AS [ПотерпевшийКод],
       P1.[PhoneNumber] AS [ПотерпевшийНомерТелефона],
       P1.[PersonCity] AS [ПотерпевшийГород],
       P1.[PersonSettlement] AS [ПотерпевшийПоселок],

       P2.[PersonFullName] AS [Страхователь.Полн наименование],
       P2.[PersonType] AS [СтраховательТипЛица],
       P2.[PersonINN] AS [СтраховательИНН],
       P2.[PersonBirthDate] AS [Страхователь.Дата рождения],
       P2.[PersoneCode] AS [СтраховательКод],
       P2.[PhoneNumber] AS [СтраховательНомерТелефона],
       P2.[PersonCity] AS [СтраховательГород],
       P2.[PersonSettlement] AS [СтраховательПоселок],

       P3.[PersonFullName] AS [Виновник.Полн наименование],
       P3.[PersonType] AS [ВиновникОбъектВладелецТипЛица],
       P3.[PersonINN] AS [ВиновникОбъектВладелецИНН],
       P3.[PersonBirthDate] AS [Виновник.Дата рождения],
       P3.[PersoneCode] AS [ВиновникОбъектВладелецКод],
       P3.[PhoneNumber] AS [ВиновникОбъектВладелецНомерТелефона],
       P3.[PersonCity] AS [ВиновникОбъектВладелецГород],
       P3.[PersonSettlement] AS [ВиновникОбъектВладелецПоселок],

       P4.[PersonFullName] AS [Потерпевший объект владелец.Полн наименование],
       P4.[PersonType] AS [ПотерпевшийОбъектВладелецТипЛица],
       P4.[PersonINN] AS [ПотерпевшийОбъектВладелецИНН],
       P4.[PersonBirthDate] AS [Потерпевший объект владелец.Дата рождения],
       P4.[PersoneCode] AS [ПотерпевшийОбъектВладелецКод],
       P4.[PhoneNumber] AS [ПотерпевшийОбъектВладелецНомерТелефона],
       P4.[PersonCity] AS [ПотерпевшийОбъектВладелецГород],
       P4.[PersonSettlement] AS [ПотерпевшийОбъектВладелецПоселок],

       P5.[PersonFullName] AS [Страхователь объект владелец.Полн наименование],
       P5.[PersonType] AS [СтраховательВладелецТипЛица],
       P5.[PersonINN] AS [СтраховательВладелецИНН],
       P5.[PersonBirthDate] AS [Страхователь объект владелец.Дата рождения],
       P5.[PersoneCode] AS [СтраховательВладелецКод],
       P5.[PhoneNumber] AS [СтраховательВладелецНомерТелефона],
       P5.[PersonCity] AS [СтраховательВладелецГород],
       P5.[PersonSettlement] AS [СтраховательВладелецПоселок],

       P6.[PersonFullName] AS [Лицо управлявшее ТС.Полн наименование],
       P6.[PersonType] AS [ВодительСтрахователяТипЛица],
       P6.[PersonINN] AS [ВодительСтрахователяИНН],
       P6.[PersonBirthDate] AS [Лицо управлявшее ТС.Дата рождения],
       P6.[PersoneCode] AS [ВодительСтрахователяКод],
       P6.[PhoneNumber] AS [ВодительСтрахователяНомерТелефона],
       P6.[PersonCity] AS [ВодительСтрахователяГород],
       P6.[PersonSettlement] AS [ВодительСтрахователяПоселок],

       P7.[PersonFullName] AS [Получатель возмещения.Полн наименование],
       P7.[PersonType] AS [ПолучательВозмещенияТипЛица],
       P7.[PersonINN] AS [ПолучательВозмещенияИНН],
       P7.[PersonBirthDate] AS [Получатель возмещения.Дата рождения],
       P7.[PersoneCode] AS [ПолучательВозмещенияКод],
       P7.[PhoneNumber] AS [ПолучательВозмещенияНомерТелефона],
       P7.[PersonCity] AS [ПолучательВозмещенияГород],
       P7.[PersonSettlement] AS [ПолучательВозмещенияПоселок],

       P8.[PersonFullName] AS [Потерпевший страхователь.Полн наименование],
       P8.[PersonType] AS [ПотерпевшийСтраховательТипЛица],
       P8.[PersonINN] AS [ПотерпевшийСтраховательИНН],
       P8.[PersonBirthDate] AS [Потерпевший страхователь.Дата рождения],
       P8.[PersoneCode] AS [ПотерпевшийСтраховательКод],
       P8.[PhoneNumber] AS [ПотерпевшийСтраховательНомерТелефона],
       P8.[PersonCity] AS [ПотерпевшийСтраховательГород],
       P8.[PersonSettlement] AS [ПотерпевшийСтраховательПоселок],

       l.[Объект виновника.Гос номер],
       l.[Объект виновника.Идентификатор],
       l.[Объект потерпевшего.Гос номер],
       l.[Объект потерпевшего.Идентификатор],
       l.[Убыток.Объект страхователя.Гос номер],
       l.[Убыток.Объект страхователя.Идентификатор],
       l.[Решение],
       pay.[Сумма],
       l.[ВыплатыФормаВозмещения],
       l.[СтадияПроцесс]
FROM losses AS l
LEFT JOIN persons AS P  ON P.[PersonID]  = l.[GuiltyPersonID]
LEFT JOIN persons AS P1 ON P1.[PersonID] = l.[VictimPersonID]
LEFT JOIN persons AS P2 ON P2.[PersonID] = l.[PolicyholderPersonID]
LEFT JOIN persons AS P3 ON P3.[PersonID] = l.[GuiltyObjectOwnerPersonID]
LEFT JOIN persons AS P4 ON P4.[PersonID] = l.[VictimObjectOwnerPersonID]
LEFT JOIN persons AS P5 ON P5.[PersonID] = l.[PolicyholderObjectOwnerPersonID]
LEFT JOIN persons AS P6 ON P6.[PersonID] = l.[DriverPersonID]
LEFT JOIN persons AS P7 ON P7.[PersonID] = l.[PaymentRecipientPersonID]
LEFT JOIN persons AS P8 ON P8.[PersonID] = l.[VIctimPolicyholderPersonID]
LEFT JOIN pay ON pay.[LossID] = l.[LossID]
OPTION (MAXDOP 2);
