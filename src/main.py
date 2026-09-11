import os
import json
import pickle
import shutil
import datetime
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import networkx as nx

from . import config

wrong_people_val = config.wrong_people_val
wrong_auto_val = config.wrong_auto_val
bound = config.bound


def _viz_top_n() -> int:
    """Сколько крупнейших групп рисовать (из config, при каждом вызове)."""
    return max(1, int(getattr(config, 'viz_top_n', 200)))


VARS_DIR = './vars'
OUTPUT_DIR = './output'
GEPHI_DIR = os.path.join(OUTPUT_DIR, 'gephi')
HTML_DIR = os.path.join(OUTPUT_DIR, 'html')
FINGERPRINT_PATH = os.path.join(VARS_DIR, 'input_fingerprint.json')
# Ниже этого порога — подробные title на рёбрах; выше — быстрая стилизация
DETAILED_STYLE_MAX_EDGES = 50_000
MAX_RELABEL_NODES = 10_000
# Physics / spring только на малых компонентах (иначе снова зависания)
PHYSICS_MAX_NODES = 400
PHYSICS_MAX_EDGES = 2_000
REQUIRED_ARTIFACTS = (
    'data', 'people', 'VIN', 'objects', 'columns',
    'links', 'G', 'groups', 'stat',
)


def clear_graph_outputs():
    """Удалить старые HTML и Gephi перед новым расчётом/визуализацией."""
    for path in (HTML_DIR, GEPHI_DIR):
        if os.path.isdir(path):
            shutil.rmtree(path)
    if os.path.isdir(OUTPUT_DIR):
        for name in os.listdir(OUTPUT_DIR):
            full = os.path.join(OUTPUT_DIR, name)
            if os.path.isfile(full) and name.lower().endswith(('.html', '.gexf')):
                os.remove(full)


def ensure_artifact_dirs():
    """Создать папки для pickle/Excel/HTML/Gephi, если их ещё нет."""
    for path in (VARS_DIR, OUTPUT_DIR, HTML_DIR, GEPHI_DIR):
        os.makedirs(path, exist_ok=True)


def log(msg: str = '', *, level: str = 'info') -> None:
    """Единый формат логов пайплайна."""
    prefixes = {
        'header': '',
        'step': '▸ ',
        'ok': '✓ ',
        'info': '  · ',
        'warn': '⚠ ',
        'err': '✗ ',
    }
    prefix = prefixes.get(level, '  · ')
    if level == 'header':
        line = '─' * 44
        print(f'\n{line}\n  {msg}\n{line}', flush=True)
        return
    if level == 'step':
        print(f'\n{prefix}{msg}', flush=True)
        return
    print(f'{prefix}{msg}', flush=True)


def _list_input_files():
    """Excel и SQL в рабочей папке — источники для отпечатка входа."""
    files = []
    for name in os.listdir(os.getcwd()):
        full = os.path.join(os.getcwd(), name)
        if not os.path.isfile(full):
            continue
        lower = name.lower()
        if lower.endswith(('.xlsx', '.xls', '.sql')):
            files.append(full)
    return sorted(files)


def _fingerprint_inputs(files):
    items = []
    for full in files:
        st = os.stat(full)
        mtime_ns = getattr(st, 'st_mtime_ns', int(st.st_mtime * 1_000_000_000))
        items.append({
            'name': os.path.basename(full),
            'size': st.st_size,
            'mtime_ns': mtime_ns,
        })
    return {'bound': bound, 'files': items}


def _artifacts_ready():
    return all(
        os.path.isfile(os.path.join(VARS_DIR, name))
        for name in REQUIRED_ARTIFACTS
    )


def _save_fingerprint(fingerprint):
    os.makedirs(VARS_DIR, exist_ok=True)
    with open(FINGERPRINT_PATH, 'w', encoding='utf-8') as fh:
        json.dump(fingerprint, fh, ensure_ascii=False, indent=2)


def _load_fingerprint():
    if not os.path.isfile(FINGERPRINT_PATH):
        return None
    with open(FINGERPRINT_PATH, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def try_load_cached_artifacts():
    """
    Если Excel/SQL (и bound) не менялись и в ./vars есть артефакты — загрузить их.
    Возвращает (успех, fingerprint).
    """
    files = _list_input_files()
    fingerprint = _fingerprint_inputs(files)
    if not _artifacts_ready():
        return False, fingerprint
    old = _load_fingerprint()
    if old != fingerprint:
        return False, fingerprint
    log('вход не изменился — загружаю ./vars', level='ok')
    load_objects()
    load_links()
    load_statistics()
    return True, fingerprint


def run_pipeline():
    """Пайплайн: кэш или пересчёт → статистика/Gephi/HTML. Без GUI в ноутбуке.

    В Jupyter вызывайте ``run_pipeline()``, не ``run()`` (конфликт с ``%run``).
    Число групп для отрисовки: ``config.viz_top_n``.
    """
    top_n = _viz_top_n()
    log('Пайплайн графов', level='header')
    log(f'viz_top_n = {top_n}', level='info')
    ensure_artifact_dirs()
    clear_graph_outputs()
    ensure_artifact_dirs()
    cached, fingerprint = try_load_cached_artifacts()
    if not cached:
        log('1/4  Загрузка и предобработка Excel', level='step')
        load()
        log('2/4  Построение связей', level='step')
        create_links()
        log('3/4  Статистика → ./output/statistics.xlsx', level='step')
        create_statistics()
        _save_fingerprint(fingerprint)
    else:
        log('1–3/4  Пересчёт пропущен (кэш)', level='ok')
    log(f'4/4  Gephi + HTML (топ-{top_n} групп)', level='step')
    visualize()
    log('Готово', level='ok')


# Для скриптов; в ноутбуке используйте run_pipeline()
run = run_pipeline


def ensure_runtime_state():
    """Поднять data/links/big_groups из памяти или из ./vars."""
    global links, big_groups, objects, people
    need = (
        'links' not in globals()
        or links is None
        or 'big_groups' not in globals()
        or big_groups is None
        or len(big_groups) == 0
    )
    if need:
        ok, _ = try_load_cached_artifacts()
        if not ok:
            raise RuntimeError(
                'Нет данных в памяти и кэш ./vars не подходит. Сначала run_pipeline().'
            )
    if not isinstance(objects, pd.DataFrame):
        objects = pd.DataFrame(objects)


def _subgraph_for_group(group_index: int = 0) -> nx.Graph:
    ensure_runtime_state()
    if group_index < 0 or group_index >= len(big_groups):
        raise IndexError(f'group_index должен быть 0…{len(big_groups) - 1}')
    group_nodes = big_groups[group_index]
    sub_links = links[
        links['obj1'].isin(group_nodes) & links['obj2'].isin(group_nodes)
    ]
    graph = nx.from_pandas_edgelist(sub_links, 'obj1', 'obj2', create_using=nx.Graph())
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def degree_report(group_index: int = 0, top_n: int = 40):
    """
    Таблицы для выбора порога хабов (degree > N).

    Возвращает (summary, top_hubs, cut_scenarios).
    Скорость: O(E) — на сотнях тысяч рёбер обычно секунды.
    """
    ensure_runtime_state()
    graph = _subgraph_for_group(group_index)
    people_count = len(people)
    objects_df = pd.DataFrame(objects)

    deg = pd.Series(dict(graph.degree()), name='degree')
    if deg.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    def _pct(p):
        return float(np.percentile(deg.to_numpy(), p))

    p90, p95, p99 = _pct(90), _pct(95), _pct(99)
    mean_d = float(deg.mean())
    std_d = float(deg.std(ddof=0)) if len(deg) > 1 else 0.0
    mean_2std = mean_d + 2 * std_d

    summary = pd.DataFrame(
        [
            ('group_index', group_index),
            ('узлов', int(graph.number_of_nodes())),
            ('рёбер', int(graph.number_of_edges())),
            ('degree min', int(deg.min())),
            ('degree median', float(deg.median())),
            ('degree mean', round(mean_d, 2)),
            ('degree std', round(std_d, 2)),
            ('degree p90', round(p90, 2)),
            ('degree p95', round(p95, 2)),
            ('degree p99', round(p99, 2)),
            ('degree max', int(deg.max())),
            ('mean+2std', round(mean_2std, 2)),
        ],
        columns=['метрика', 'значение'],
    )

    top_hubs = (
        deg.sort_values(ascending=False)
        .head(top_n)
        .rename_axis('obj_idx')
        .reset_index()
    )
    top_hubs['тип'] = np.where(top_hubs['obj_idx'] > people_count, 'VIN', 'человек')
    top_hubs['метка'] = top_hubs['obj_idx'].map(
        lambda i: str(objects_df.loc[i, 0]) if i in objects_df.index else str(i)
    )

    scenarios = []
    for name, thr in [
        ('p90', p90),
        ('p95', p95),
        ('p99', p99),
        ('mean+2std', mean_2std),
        ('degree>20', 20),
        ('degree>50', 50),
        ('degree>100', 100),
    ]:
        thr_i = int(np.floor(thr))
        hubs = set(deg[deg > thr_i].index)
        if not hubs:
            scenarios.append(
                {
                    'сценарий': name,
                    'порог N (degree > N)': thr_i,
                    'хабов снято': 0,
                    'компонент после': graph.number_connected_components(),
                    'размер топ-1': max((len(c) for c in nx.connected_components(graph)), default=0),
                    'размер топ-5 (сумма)': 0,
                }
            )
            continue
        H = graph.copy()
        H.remove_nodes_from(hubs)
        comps = sorted(nx.connected_components(H), key=len, reverse=True)
        top5 = sum(len(c) for c in comps[:5])
        scenarios.append(
            {
                'сценарий': name,
                'порог N (degree > N)': thr_i,
                'хабов снято': len(hubs),
                'компонент после': len(comps),
                'размер топ-1': len(comps[0]) if comps else 0,
                'размер топ-5 (сумма)': top5,
            }
        )
    cut_scenarios = pd.DataFrame(scenarios)
    return summary, top_hubs, cut_scenarios


def show_degree_report(group_index: int = 0, top_n: int = 40):
    """Показать отчёт по степеням в Jupyter."""
    from IPython.display import display, Markdown

    summary, top_hubs, cut_scenarios = degree_report(group_index, top_n=top_n)
    display(Markdown(f'### Степени группы {group_index} (для выбора N: режем degree **> N**)'))
    display(summary)
    display(Markdown('### Сценарии среза хабов'))
    display(cut_scenarios)
    display(Markdown(f'### Топ-{top_n} хабов'))
    display(top_hubs)
    return summary, top_hubs, cut_scenarios


def load():
    choice_query()


def choice_query():
    global data
    onlyfiles = [
        f for f in os.listdir(os.getcwd())
        if os.path.isfile(os.path.join(os.getcwd(), f)) and f.lower().endswith(('.xlsx', '.xls'))
    ]
    if not onlyfiles:
        raise FileNotFoundError('В рабочей папке нет xlsx/xls-файлов')
    log(f'Excel: {", ".join(onlyfiles)}', level='info')
    max_workers = min(4, len(onlyfiles))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        frames = list(pool.map(pd.read_excel, onlyfiles))
    data = pd.concat(frames, axis=0, ignore_index=True)
    log(f'строк: {len(data):,}', level='info')
    preprocessing()

# Препроцессинг
def preprocessing():
    global data, people, VIN, objects
    global ID_col, FIO_cul_col, BD_cul_col, FIO_vic_col, BD_vic_col, FIO_pol_col, BD_pol_col
    global FIO_cul_obj_own_col, BD_cul_obj_own_col, FIO_vic_obj_own_col, BD_vic_obj_own_col, FIO_pol_obj_own_col, BD_pol_obj_own_col
    global FIO_pol_driv_col, BD_pol_driv_col, FIO_ref_rec, BD_ref_rec
    global VIN_cul_col, VIN_vic_col, Filial_col, Reject_col, Sum_col, FIO_vic_pol_col, BD_vic_pol_col
    global wrong_people_val, wrong_auto_val
    global persons
    
    # Запись длинных колонок в переменные, чтобы сократить код
    ID_col, FIO_cul_col, BD_cul_col, \
    FIO_vic_col, BD_vic_col, \
    FIO_pol_col, BD_pol_col, \
    FIO_cul_obj_own_col, BD_cul_obj_own_col, \
    FIO_vic_obj_own_col, BD_vic_obj_own_col, \
    FIO_pol_obj_own_col, BD_pol_obj_own_col, \
    FIO_pol_driv_col, BD_pol_driv_col, \
    FIO_ref_rec, BD_ref_rec, \
    VIN_cul_col, VIN_vic_col, Filial_col, Reject_col, Sum_col, FIO_vic_pol_col, BD_vic_pol_col = \
    'Убыток', 'Убыток виновник полннаименование', 'Убыток виновник датарождения', 'Потерпевший.Полн наименование', \
    'Потерпевший.Дата рождения', 'Страхователь.Полн наименование', 'Страхователь.Дата рождения', \
    'Виновник.Полн наименование', 'Виновник.Дата рождения', \
    'Потерпевший объект владелец.Полн наименование', 'Потерпевший объект владелец.Дата рождения', \
    'Страхователь объект владелец.Полн наименование', 'Страхователь объект владелец.Дата рождения', \
    'Лицо управлявшее ТС.Полн наименование', 'Лицо управлявшее ТС.Дата рождения', \
    'Получатель возмещения.Полн наименование', 'Получатель возмещения.Дата рождения', \
    'Объект виновника.Идентификатор', 'Объект потерпевшего.Идентификатор', \
    'Филиалурегулировщик', 'Решение', 'Сумма', 'Потерпевший страхователь.Полн наименование', \
    'Потерпевший страхователь.Дата рождения'
    
    for i in data.columns:
        j = i.strip()
        data.rename({i: j}, axis=1, inplace=True)
        
    error = 0
    for i in [ID_col, \
              FIO_cul_col, BD_cul_col, FIO_vic_col, BD_vic_col, FIO_pol_col, BD_pol_col, \
              FIO_cul_obj_own_col, BD_cul_obj_own_col, FIO_vic_obj_own_col, BD_vic_obj_own_col, FIO_pol_obj_own_col, BD_pol_obj_own_col, \
              FIO_pol_driv_col, BD_pol_driv_col, \
              FIO_ref_rec, BD_ref_rec, \
              VIN_cul_col, VIN_vic_col, \
              Filial_col, Reject_col, Sum_col, FIO_vic_pol_col, BD_vic_pol_col]:
        if i not in data.columns:
            log(f'нет столбца: {i}', level='err')
            error = 1
    if error == 1:
        return -1
    
    # Удаление дубликатов
    data[ID_col] = data[ID_col].astype(int)
    temp = data[[ID_col]].duplicated().mean()
    if temp > 0:
        data.drop_duplicates(inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    # Удаление аномалий в людях
    for i in [FIO_cul_col, FIO_vic_col, FIO_pol_col, FIO_cul_obj_own_col, FIO_vic_obj_own_col, FIO_pol_obj_own_col, FIO_pol_driv_col,
              FIO_ref_rec, VIN_cul_col, VIN_vic_col, FIO_vic_pol_col]:
        data[i] = data[i].str.upper()
        for j in wrong_people_val:
            data[i] = data[i].replace(j, np.nan)
    
    # Замена категорий по столбцу "решения по акту" на численные значения
    data['Reject'] = data[Reject_col].fillna(0).replace({'Отказ': 1, 'Отказать': 1, 'Выплата': 0, 'Выплатить': 0, 'Не урегулирован': 1})
    
    #Замена аномалий в сумме платежа
    data = data[~(data['Сумма'].apply(type) == datetime.datetime)]
    data['Сумма'] = data['Сумма'].astype(float)
    
    data.reset_index(drop=True, inplace=True)
        
    #Проверка на мусор
    for i in [VIN_cul_col, VIN_vic_col]: 
        temp = data[i].isin(wrong_auto_val.keys()).mean()
        if temp > 0:
            data[i] = data[i].replace(wrong_auto_val)
    
    # Изменение типа в полях "ден
    for i in [BD_vic_col, BD_cul_col, BD_pol_col, BD_cul_obj_own_col,
              BD_vic_obj_own_col, BD_pol_obj_own_col, BD_pol_driv_col, BD_ref_rec, BD_vic_pol_col]:
        data[i] = pd.to_datetime(data[i], errors='coerce').astype(str).replace('NaT', np.nan)
        
        
    data['ВиновникФИОДР'] = data['Убыток виновник полннаименование'] + ' ' + data['Убыток виновник датарождения']
    data['ПотерпевшийФИОДР'] = data['Потерпевший.Полн наименование'] + ' ' + data['Потерпевший.Дата рождения']
    data['СтраховательФИОДР'] = data['Страхователь.Полн наименование'] + ' ' + data['Страхователь.Дата рождения']
    data['ВиновникОбъектВладелецФИОДР'] = data['Виновник.Полн наименование'] + ' ' + data['Виновник.Дата рождения']
    data['ПотерпевшийОбъектВладелецФИОДР'] = data['Потерпевший объект владелец.Полн наименование'] + ' ' + data['Потерпевший объект владелец.Дата рождения']
    data['СтраховательВладелецФИОДР'] = data['Страхователь объект владелец.Полн наименование'] + ' ' + data['Страхователь объект владелец.Дата рождения']
    data['ВодительСтрахователяФИОДР'] = data['Лицо управлявшее ТС.Полн наименование'] + ' ' + data['Лицо управлявшее ТС.Дата рождения']
    data['ПолучательВозмещенияФИОДР'] = data['Получатель возмещения.Полн наименование'] + ' ' + data['Получатель возмещения.Дата рождения']
    data['ПотерпевшийСтраховательФИОДР'] = data['Потерпевший страхователь.Полн наименование'] + ' ' + data['Потерпевший страхователь.Дата рождения']
    
    try:
        persons = pd.DataFrame(columns=['FIO',
                                        'BD',
                                        'PersoneCode',
                                        'PhoneNumber',
                                        'PersonCity',
                                        'PersonSettlement',
                                        'PersonNameFIO',
    #                                    'Person_Email'
                                       ])
        persons = pd.concat([persons, data[['Убыток виновник полннаименование',
                                            'Убыток виновник датарождения',
                                            'ВиновникКод',
                                            'ВиновникНомерТелефона',
                                            'ВиновникГород',
                                            'ВиновникПоселок',
                                            'ВиновникФИОДР',
    #                                        'ВиновникEmail'
                                           ]].rename(columns={'Убыток виновник полннаименование': 'FIO',
                                                                              'Убыток виновник датарождения': 'BD',
                                                                              'ВиновникКод': 'PersoneCode',
                                                                              'ВиновникНомерТелефона': 'PhoneNumber',
                                                                              'ВиновникГород': 'PersonCity',
                                                                              'ВиновникПоселок': 'PersonSettlement',
                                                                              'ВиновникФИОДР': 'PersonNameFIO',
     #                                                                         'ВиновникEmail': 'Person_Email'
                                                             })], ignore_index=True)
        persons = pd.concat([persons, data[['Потерпевший.Полн наименование',
'Потерпевший.Дата рождения',
                                            'ПотерпевшийКод',
                                            'ПотерпевшийНомерТелефона',
                                            'ПотерпевшийГород',
                                            'ПотерпевшийПоселок',
                                            'ПотерпевшийФИОДР',
    #                                        'ПотерпевшийEmail'
                                           ]].rename(columns={'Потерпевший.Полн наименование': 'FIO',
                                                                                 'Потерпевший.Дата рождения': 'BD',
                                                                                 'ПотерпевшийКод': 'PersoneCode',
                                                                                 'ПотерпевшийНомерТелефона': 'PhoneNumber',
                                                                                 'ПотерпевшийГород': 'PersonCity',
                                                                                 'ПотерпевшийПоселок': 'PersonSettlement',
                                                                                 'ПотерпевшийФИОДР': 'PersonNameFIO',
    #                                                                             'ПотерпевшийEmail': 'Person_Email'
                                                             })], ignore_index=True)
        persons = pd.concat([persons, data[['Страхователь.Полн наименование',
                                            'Страхователь.Дата рождения',
                                            'СтраховательКод',
                                            'СтраховательНомерТелефона',
                                            'СтраховательГород',
                                            'СтраховательПоселок',
                                            'СтраховательФИОДР',
    #                                        'СтраховательEmail'
                                           ]].rename(columns={'Страхователь.Полн наименование': 'FIO',
                                                                                  'Страхователь.Дата рождения': 'BD',
                                                                                  'СтраховательКод': 'PersoneCode',
                                                                                  'СтраховательНомерТелефона': 'PhoneNumber',
                                                                                  'СтраховательГород': 'PersonCity',
                                                                                  'СтраховательПоселок': 'PersonSettlement',
                                                                                  'СтраховательФИОДР': 'PersonNameFIO',
    #                                                                              'СтраховательEmail': 'Person_Email'
                                                             })], ignore_index=True)
        persons = pd.concat([persons, data[['Виновник.Полн наименование',
                                            'Виновник.Дата рождения',
                                            'ВиновникОбъектВладелецКод',
                                            'ВиновникОбъектВладелецНомерТелефона',
                                            'ВиновникОбъектВладелецГород',
                                            'ВиновникОбъектВладелецПоселок',
                                            'ВиновникОбъектВладелецФИОДР',
    #                                        'ВиновникОбъектВладелецEmail'
                                           ]].rename(columns={'Виновник.Полн наименование': 'FIO',
                                                                                            'Виновник.Дата рождения': 'BD',
'ВиновникОбъектВладелецКод': 'PersoneCode',
                                                                                            'ВиновникОбъектВладелецНомерТелефона': 'PhoneNumber',
                                                                                            'ВиновникОбъектВладелецГород': 'PersonCity',
                                                                                            'ВиновникОбъектВладелецПоселок': 'PersonSettlement',
                                                                                            'ВиновникОбъектВладелецФИОДР': 'PersonNameFIO',
                                                                                            #'ВиновникОбъектВладелецEmail': 'Person_Email'
                                                             })], ignore_index=True)               
        persons = pd.concat([persons, data[['Потерпевший объект владелец.Полн наименование',
                                            'Потерпевший объект владелец.Дата рождения',
                                            'ПотерпевшийОбъектВладелецКод',
                                            'ПотерпевшийОбъектВладелецНомерТелефона',
                                            'ПотерпевшийОбъектВладелецГород',
                                            'ПотерпевшийОбъектВладелецПоселок',
                                            'ПотерпевшийОбъектВладелецФИОДР',
                                            #'ПотерпевшийОбъектВладелецEmail'
                                           ]].rename(columns={'Потерпевший объект владелец.Полн наименование': 'FIO',
                                                                                               'Потерпевший объект владелец.Дата рождения': 'BD',
                                                                                               'ПотерпевшийОбъектВладелецКод': 'PersoneCode',
                                                                                               'ПотерпевшийОбъектВладелецНомерТелефона': 'PhoneNumber',
                                                                                               'ПотерпевшийОбъектВладелецГород': 'PersonCity',
                                                                                               'ПотерпевшийОбъектВладелецПоселок': 'PersonSettlement',
                                                                                               'ПотерпевшийОбъектВладелецФИОДР': 'PersonNameFIO',
                                                                                               #'ПотерпевшийОбъектВладелецEmail': 'Person_Email'
                                                             })], ignore_index=True)
        persons = pd.concat([persons, data[['Страхователь объект владелец.Полн наименование',
                                            'Страхователь объект владелец.Дата рождения',
                                            'СтраховательВладелецКод',
                                            'СтраховательВладелецНомерТелефона',
                                            'СтраховательВладелецГород',
                                            'СтраховательВладелецПоселок',
                                            'СтраховательВладелецФИОДР',
                                            #'СтраховательВладелецEmail'
                                           ]].rename(columns={'Страхователь объект владелец.Полн наименование': 'FIO',
                                                                                          'Страхователь объект владелец.Дата рождения': 'BD',
                                                                                          'СтраховательВладелецКод': 'PersoneCode',
                                                                                          'СтраховательВладелецНомерТелефона': 'PhoneNumber',
'СтраховательВладелецГород':'PersonCity',
                                                                                          'СтраховательВладелецПоселок': 'PersonSettlement',
                                                                                          'СтраховательВладелецФИОДР': 'PersonNameFIO',
                                                                                          #'СтраховательВладелецEmail': 'Person_Email'
                                                             })], ignore_index=True)
        persons = pd.concat([persons, data[['Лицо управлявшее ТС.Полн наименование',
                                            'Лицо управлявшее ТС.Дата рождения',
                                            'ВодительСтрахователяКод',
                                            'ВодительСтрахователяНомерТелефона',
                                            'ВодительСтрахователяГород',
                                            'ВодительСтрахователяПоселок',
                                            'ВодительСтрахователяФИОДР',
                                            #'ВодительСтрахователяEmail'
                                           ]].rename(columns = {'Лицо управлявшее ТС.Полн наименование': 'FIO',
                                                                                            'Лицо управлявшее ТС.Дата рождения': 'BD',
                                                                                            'ВодительСтрахователяКод': 'PersoneCode',
                                                                                            'ВодительСтрахователяНомерТелефона': 'PhoneNumber',
                                                                                            'ВодительСтрахователяГород': 'PersonCity',
                                                                                            'ВодительСтрахователяПоселок': 'PersonSettlement',
                                                                                            'ВодительСтрахователяФИОДР': 'PersonNameFIO',
                                                                                            #'ВодительСтрахователяEmail': 'Person_Email'
                                                               })], ignore_index=True)
        persons = pd.concat([persons, data[['Получатель возмещения.Полн наименование',
                                            'Получатель возмещения.Дата рождения',
                                            'ПолучательВозмещенияКод',
                                            'ПолучательВозмещенияНомерТелефона',
                                            'ПолучательВозмещенияГород',
                                            'ПолучательВозмещенияПоселок',
                                            'ПолучательВозмещенияФИОДР',
                                            #'ПолучательВозмещенияEmail'
                                           ]].rename(columns={'Получатель возмещения.Полн наименование': 'FIO',
                                                                                          'Получатель возмещения.Дата рождения': 'BD',
                                                                                          'ПолучательВозмещенияКод': 'PersoneCode',
                                                                                          'ПолучательВозмещенияНомерТелефона': 'PhoneNumber',
                                                                                          'ПолучательВозмещенияГород': 'PersonCity',
                                                                                          'ПолучательВозмещенияПоселок': 'PersonSettlement',
                                                                                          'ПолучательВозмещенияФИОДР': 'PersonNameFIO',
#'ПолучательВозмещенияEmail': 'Person_Email'
                                                             })], ignore_index=True)
        persons = pd.concat([persons, data[['Потерпевший страхователь.Полн наименование',
                                            'Потерпевший страхователь.Дата рождения',
                                            'ПотерпевшийСтраховательКод',
                                            'ПотерпевшийСтраховательНомерТелефона',
                                            'ПотерпевшийСтраховательГород',
                                            'ПотерпевшийСтраховательПоселок',
                                            'ПотерпевшийСтраховательФИОДР',
                                            #'ПотерпевшийСтраховательEmail'
                                           ]].rename(columns={'Потерпевший страхователь.Полн наименование': 'FIO',
                                                                                             'Потерпевший страхователь.Дата рождения': 'BD',
                                                                                             'ПотерпевшийСтраховательКод': 'PersoneCode',
                                                                                             'ПотерпевшийСтраховательНомерТелефона': 'PhoneNumber',
                                                                                             'ПотерпевшийСтраховательГород': 'PersonCity',
                                                                                             'ПотерпевшийСтраховательПоселок': 'PersonSettlement',
                                                                                             'ПотерпевшийСтраховательФИОДР': 'PersonNameFIO',
                                                                                             #'ПотерпевшийСтраховательEmail': 'Person_Email'
                                                             })], ignore_index=True)
        persons = persons.drop_duplicates(subset=['FIO', 'BD'])
        persons['BD'] = persons['BD'].astype(str)
        persons.reset_index(drop=True, inplace=True)    
        people = persons[['FIO', 'BD']]
        people = people['FIO'].str.strip() + ' ' + people['BD'].str.strip()
        people = people[~people.isnull()]
        people.reset_index(drop=True, inplace=True)
    except:
        people = pd.DataFrame(columns=['FIO', 'BD'])
        people = pd.concat([people, data[[FIO_vic_col, BD_vic_col]].rename(columns={FIO_vic_col: 'FIO', BD_vic_col: 'BD'})], axis=0, ignore_index=True)
        people = pd.concat([people, data[[FIO_cul_col, BD_cul_col]].rename(columns={FIO_cul_col: 'FIO', BD_cul_col: 'BD'})], axis=0, ignore_index=True)
        people = pd.concat([people, data[[FIO_pol_col, BD_pol_col]].rename(columns={FIO_pol_col: 'FIO', BD_pol_col: 'BD'})], axis=0, ignore_index=True)
        people = pd.concat([people, data[[FIO_vic_obj_own_col, BD_vic_obj_own_col]].rename(columns={FIO_vic_obj_own_col: 'FIO',
                                                                                               BD_vic_obj_own_col: 'BD'})], axis=0, ignore_index=True)
        people = pd.concat([people, data[[FIO_cul_obj_own_col, BD_cul_obj_own_col]].rename(columns={FIO_cul_obj_own_col: 'FIO',
                                                                                               BD_cul_obj_own_col: 'BD'})], axis=0, ignore_index=True)
        people = pd.concat([people, data[[FIO_pol_obj_own_col, BD_pol_obj_own_col]].rename(columns={FIO_pol_obj_own_col: 'FIO',
                                                                                               BD_pol_obj_own_col: 'BD'})], axis=0, ignore_index=True)
        people = pd.concat([people, data[[FIO_pol_driv_col, BD_pol_driv_col]].rename(columns={FIO_pol_driv_col:'FIO', BD_pol_driv_col: 'BD'})], axis=0, ignore_index=True)
        people = pd.concat([people, data[[FIO_ref_rec,BD_ref_rec]].rename(columns={FIO_ref_rec: 'FIO', BD_ref_rec: 'BD'})], axis=0, ignore_index=True)
        people = people.drop_duplicates(subset=['FIO', 'BD'])
        people['BD'] = people['BD'].astype(str)
        people = people['FIO'].str.strip() + ' ' + people['BD'].str.strip()
        people = people[~people.isnull()]
        people.reset_index(drop=True, inplace=True) 


    VIN = pd.Series(name='VIN')
    VIN = pd.concat([VIN, data[VIN_vic_col]], ignore_index=True)
    VIN = pd.concat([VIN, data[VIN_cul_col]], ignore_index=True)
    VIN = VIN.drop_duplicates()
    VIN = VIN[~VIN.isnull()]
    VIN = VIN[~(VIN == '')]
    VIN.reset_index(drop=True, inplace=True)
    
    data['Victim'] = data[FIO_vic_col] + ' ' + data[BD_vic_col].astype(str)
    data['Culprit'] = data[FIO_cul_col] + ' ' + data[BD_cul_col].astype(str)
    data['CulpritPolicyholder'] = data[FIO_pol_col] + ' ' + data[BD_pol_col].astype(str)
    data['VictimObjectOwner'] = data[FIO_vic_obj_own_col] + ' ' + data[BD_vic_obj_own_col].astype(str)
    data['CulpritObjectOwner'] = data[FIO_cul_obj_own_col] + ' ' + data[BD_cul_obj_own_col].astype(str)
    data['PolicyholderObjectOwner'] = data[FIO_pol_obj_own_col] + ' ' + data[BD_pol_obj_own_col].astype(str)
    data['PolicyholderDriver'] = data[FIO_pol_driv_col] + ' ' + data[BD_pol_driv_col].astype(str)
    data['PaymentRecipient'] = data[FIO_ref_rec] + ' ' + data[BD_ref_rec].astype(str)
    data['VINv'] = data[VIN_vic_col]
    data['VINc'] = data[VIN_cul_col]
    data['VictimPolicyholder'] = data[FIO_vic_pol_col] + ' ' + data[BD_vic_pol_col].astype(str)
    
    objects = people.copy()
    objects = pd.concat([objects, VIN], ignore_index=True)
    objects = objects[~objects.isnull()]
    objects.drop_duplicates(inplace=True)
#    objects.reset_index(drop=True, inplace=True)
    
    ensure_artifact_dirs()
    pickle.dump(data, open('./vars/data', 'wb'))  
    pickle.dump(people, open('./vars/people', 'wb'))  
    pickle.dump(VIN, open('./vars/VIN','wb'))  
    pickle.dump(objects, open('./vars/objects', 'wb'))  
    pickle.dump(persons, open('./vars/persons', 'wb'))
    pickle.dump([ID_col, \
                 FIO_cul_col, BD_cul_col, \
                 FIO_vic_col, BD_vic_col, \
                 FIO_pol_col, BD_pol_col, \
                 FIO_cul_obj_own_col, BD_cul_obj_own_col, \
                 FIO_vic_obj_own_col, BD_vic_obj_own_col, \
                 FIO_pol_obj_own_col, BD_pol_obj_own_col, \
                 FIO_pol_driv_col, BD_pol_driv_col, \
                 FIO_ref_rec, BD_ref_rec, \
                 VIN_cul_col, VIN_vic_col, Filial_col, Reject_col, Sum_col, FIO_vic_pol_col, BD_vic_pol_col], open("./vars/columns",'wb'))  
    log(f'объекты: {len(objects):,}  |  people: {len(people):,}', level='info')

def load_objects():
    global data, people, VIN, objects
    global ID_col, FIO_cul_col, BD_cul_col, FIO_vic_col, BD_vic_col, FIO_pol_col, BD_pol_col
    global FIO_cul_obj_own_col, BD_cul_obj_own_col, FIO_vic_obj_own_col, BD_vic_obj_own_col, FIO_pol_obj_own_col, BD_pol_obj_own_col
    global FIO_pol_driv_col, BD_pol_driv_col, FIO_ref_rec, BD_ref_rec
    global VIN_cul_col, VIN_vic_col, Filial_col, Reject_col, Sum_col, FIO_vic_pol_col, BD_vic_pol_col
    global persons

    data = pickle.load(open('./vars/data', 'rb'))
    people = pickle.load(open('./vars/people', 'rb'))
    VIN = pickle.load(open('./vars/VIN', 'rb'))
    objects = pickle.load(open('./vars/objects', 'rb'))
    temp = pickle.load(open('./vars/columns', 'rb'))
    ID_col = temp[0]
    FIO_cul_col, BD_cul_col = temp[1], temp[2]
    FIO_vic_col, BD_vic_col = temp[3], temp[4]
    FIO_pol_col, BD_pol_col = temp[5], temp[6]
    FIO_cul_obj_own_col, BD_cul_obj_own_col = temp[7], temp[8]
    FIO_vic_obj_own_col, BD_vic_obj_own_col = temp[9], temp[10]
    FIO_pol_obj_own_col, BD_pol_obj_own_col = temp[11], temp[12]
    FIO_pol_driv_col, BD_pol_driv_col = temp[13], temp[14]
    FIO_ref_rec, BD_ref_rec = temp[15], temp[16]
    VIN_cul_col, VIN_vic_col, Filial_col, Reject_col, Sum_col, FIO_vic_pol_col, BD_vic_pol_col = temp[17], temp[18], temp[19], temp[20], temp[21], temp[22], temp[23]
    log(f'из vars: data={len(data):,}', level='info')
    
def create_links():
    global data, people, VIN, objects, links
    global ID_col, FIO_cul_col, BD_cul_col, FIO_vic_col, BD_vic_col, FIO_pol_col, BD_pol_col
    global FIO_cul_obj_own_col, BD_cul_obj_own_col, FIO_vic_obj_own_col, BD_vic_obj_own_col, FIO_pol_obj_own_col, BD_pol_obj_own_col
    global FIO_pol_driv_col, BD_pol_driv_col, FIO_ref_rec, BD_ref_rec
    global VIN_cul_col, VIN_vic_col, Filial_col, Reject_col, Sum_col, FIO_vic_pol_col, BD_vic_pol_col
    global persons
    
    main_cols = ['Victim', 'Culprit', 'CulpritPolicyholder',
                 'VictimObjectOwner', 'CulpritObjectOwner', 'PolicyholderObjectOwner',
                 'PolicyholderDriver', 'PaymentRecipient', 'VINv', 'VINc', 'VictimPolicyholder']
    object_idxes = pd.Series(data=objects.index, index=objects.values)
    loss_idx = data.index.to_numpy()
    parts = []
    col_pairs = list(combinations(main_cols, 2))
    for col, link_col in col_pairs:
        obj1 = data[col].map(object_idxes)
        obj2 = data[link_col].map(object_idxes)
        mask = obj1.notna() & obj2.notna()
        if not mask.any():
            continue
        mask_np = mask.to_numpy()
        parts.append(pd.DataFrame({
            'obj1': obj1.to_numpy()[mask_np].astype(np.int64),
            'obj2': obj2.to_numpy()[mask_np].astype(np.int64),
            'Loss_idx': loss_idx[mask_np],
            'link_type': f'{col}_{link_col}',
        }))
    links = (pd.concat(parts, ignore_index=True)
             if parts else
             pd.DataFrame(columns=['obj1', 'obj2', 'Loss_idx', 'link_type']))
    ensure_artifact_dirs()
    pickle.dump(links, open('./vars/links', 'wb'))
    log(f'связей: {len(links):,}', level='info')
    
def load_links():
    global links
    
    links = pickle.load(open('./vars/links', 'rb'))
    log(f'связей из vars: {len(links):,}', level='info')

def create_statistics():
    global links, data, big_groups, stat, objects, bound
    
    G = nx.from_pandas_edgelist(links, 'obj1', 'obj2', create_using=nx.Graph())
    groups = sorted(nx.connected_components(G), key=len, reverse=True)
    big_groups = [i for i in groups if len(i) > bound]

    main_cols = ['Victim', 'Culprit', 'CulpritPolicyholder',
                 'VictimObjectOwner', 'CulpritObjectOwner', 'PolicyholderObjectOwner',
                 'PolicyholderDriver', 'PaymentRecipient', 'VINv', 'VINc', 'VictimPolicyholder']
    objects = pd.DataFrame(objects)

    # Один проход: объект/метка -> номер группы, без повторных isin по всему data
    label_to_group = {}
    obj_idx_to_group = {}
    for group_id, group in enumerate(big_groups, start=1):
        for obj_idx in group:
            obj_idx_to_group[obj_idx] = group_id
            label_to_group[objects.loc[obj_idx, 0]] = group_id

    data['Группа'] = np.nan
    for col in main_cols:
        data['Группа'] = data['Группа'].fillna(data[col].map(label_to_group))
    data['Группа'] = data['Группа'].astype('Int64')

    link_counts = links['obj1'].map(obj_idx_to_group).value_counts()
    grouped = data.groupby('Группа', dropna=True)
    loss_counts = grouped.size()
    mean_sum = grouped[Sum_col].mean()
    mean_reject = grouped['Reject'].mean()
    loss_numbers = grouped[ID_col].agg(
        lambda s: '; '.join(str(x) for x in pd.unique(s.dropna()))
    )

    people_count = len(people)
    rows = []
    for group_id, group in enumerate(big_groups, start=1):
        fio_labels = [
            str(objects.loc[obj_idx, 0])
            for obj_idx in group
            if obj_idx < people_count and pd.notna(objects.loc[obj_idx, 0])
        ]
        fio_str = '; '.join(dict.fromkeys(fio_labels))
        rows.append([
            group_id,
            int(loss_counts.get(group_id, 0)),
            len(group),
            int(link_counts.get(group_id, 0)),
            fio_str,
            loss_numbers.get(group_id, ''),
            mean_sum.get(group_id, np.nan),
            mean_reject.get(group_id, np.nan),
        ])
    stat = pd.DataFrame(rows, columns=[
        'Группа', 'Число убытков', 'Число объектов', 'Число связей',
        'ФИО участников группы', 'Номера убытков',
        'Средняя выплата', 'Средняя доля отказов',
    ])
    stat['Плотность связей'] = stat['Число связей'] / stat['Число убытков']
    stat['Плотность убытков'] = stat['Число убытков'] / stat['Число объектов']
    
    ensure_artifact_dirs()
    pickle.dump(G, open('./vars/G', 'wb'))
    pickle.dump(groups, open('./vars/groups', 'wb'))
    pickle.dump(stat, open('./vars/stat', 'wb'))
    pickle.dump(data, open('./vars/data', 'wb'))

    with pd.ExcelWriter('./output/statistics.xlsx', engine='xlsxwriter') as writer:
        stat.to_excel(writer, index=False, sheet_name='statistics')
    log(f'big_groups: {len(big_groups)}  →  ./output/statistics.xlsx', level='info')
    
    
def load_statistics():
    global links, data, big_groups, stat, objects, bound
    
    G = pickle.load(open('./vars/G', 'rb'))
    groups = pickle.load(open('./vars/groups', 'rb'))
    data = pickle.load(open('./vars/data', 'rb'))
    stat = pickle.load(open('./vars/stat', 'rb'))
    big_groups = [i for i in groups if len(i) > bound]
    objects = pd.DataFrame(objects)
    log(f'big_groups из vars: {len(big_groups)}', level='info')
        
def visualize():
    global links, data, big_groups, objects, people, VIN, ID_col

    def loss_title(loss_ids):
        if not loss_ids:
            return ''
        unique_ids = list(pd.unique(data.loc[loss_ids, ID_col]))[:3]
        return ', '.join(str(v) for v in unique_ids)

    def relabel_graph(graph):
        temp = objects.loc[list(objects.index < len(people)), 0]
        temp = temp[list(set(people.index) & set(graph.nodes))]
        temp = temp.str.split(' ').apply(lambda x: x[0].lower().capitalize()).to_dict()
        temp = {
            **temp,
            **(objects.loc[list((objects.index >= len(people))), 0][
                list(set(graph.nodes) - set(people.index))
            ].to_dict()),
        }
        df = pd.DataFrame(pd.Series(temp))
        df[1] = df[0]
        df.loc[df[0].duplicated(keep=False), 1] = (
            df.loc[df[0].duplicated(keep=False), 0]
            + ' '
            + df.groupby(0).cumcount().add(1).astype(str)
        )
        mapping = pd.Series(df[1].values, index=temp.keys()).to_dict()
        return nx.relabel_nodes(graph, mapping)

    def _sanitize_for_gexf(graph):
        H = graph.copy()
        for node in H.nodes:
            for key, value in list(H.nodes[node].items()):
                if key in ('x', 'y', 'physics'):
                    H.nodes[node].pop(key, None)
                    continue
                H.nodes[node][key] = '' if value is None else str(value)
        for u, v, attrs in H.edges(data=True):
            for key, value in list(attrs.items()):
                attrs[key] = '' if value is None else str(value)
        return H

    def _node_hover_text(node_id, attrs):
        text = attrs.get('title') or attrs.get('label') or str(node_id)
        return str(text).strip().replace('\n', ' ')

    def save_html(graph, html_path):
        """HTML (vis-network): кружки, ФИО в title (hover), physics на малых графах."""
        if graph.number_of_nodes() == 0:
            return
        parent = os.path.dirname(html_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        use_physics = (
            n_nodes <= PHYSICS_MAX_NODES and n_edges <= PHYSICS_MAX_EDGES
        )
        side = max(1, int(np.ceil(np.sqrt(n_nodes))))

        with open(html_path, 'w', encoding='utf-8') as fh:
            fh.write(
                '<!DOCTYPE html><html><head><meta charset="utf-8">'
                '<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js">'
                '</script>'
                '<style>html,body,#m{margin:0;height:100%;width:100%;}</style>'
                '</head><body><div id="m"></div><script>\n'
                'const nodes=new vis.DataSet([\n'
            )
            first = True
            for i, node in enumerate(graph.nodes):
                attrs = graph.nodes[node]
                item = {
                    'id': str(node),
                    'label': '',  # пустой label → кружок, не овал от длинного ФИО
                    'title': _node_hover_text(node, attrs),
                    'shape': 'dot',
                    'size': 10,
                    'color': attrs.get('color', 'grey'),
                }
                if use_physics:
                    item['physics'] = True
                else:
                    item['physics'] = False
                    item['x'] = float(i % side) * 80.0
                    item['y'] = float(i // side) * 80.0
                if not first:
                    fh.write(',\n')
                first = False
                json.dump(item, fh, ensure_ascii=False)
            fh.write('\n]);\nconst edges=new vis.DataSet([\n')
            first = True
            for u, v, attrs in graph.edges(data=True):
                item = {
                    'from': str(u),
                    'to': str(v),
                    'color': attrs.get('color', '#999'),
                    'width': int(attrs.get('width', 1) or 1),
                }
                title = attrs.get('title')
                if title:
                    item['title'] = str(title)
                if not first:
                    fh.write(',\n')
                first = False
                json.dump(item, fh, ensure_ascii=False)
            phys_js = (
                'physics:{enabled:true,stabilization:{iterations:200}},'
                if use_physics
                else 'physics:{enabled:false},'
            )
            fh.write(
                '\n]);\n'
                'new vis.Network(document.getElementById("m"),{nodes,edges},{'
                f'{phys_js}'
                'nodes:{shape:"dot",size:10,font:{size:0}},'
                'interaction:{dragNodes:true,dragView:true,zoomView:true,hover:true}'
                '});\n</script></body></html>\n'
            )

    def save_gephi(graph, stem):
        if graph.number_of_nodes() == 0:
            return
        os.makedirs(GEPHI_DIR, exist_ok=True)
        gexf_path = os.path.join(GEPHI_DIR, f'{stem}.gexf')
        nx.write_gexf(_sanitize_for_gexf(graph), gexf_path)

    def style_graph(G, sub_links, people_len):
        """Раскраска узлов/рёбер. На больших графах — упрощённо."""
        n_edges = G.number_of_edges()
        if n_edges > DETAILED_STYLE_MAX_EDGES:
            red = sub_links[
                sub_links['link_type'].isin(['Victim_Culprit', 'Culprit_Victim'])
            ]
            red_nodes = set(red['obj1']).union(set(red['obj2'])) if len(red) else set()
            for i in G.nodes:
                if i in red_nodes:
                    G.nodes[i]['color'] = 'red'
                elif i > people_len:
                    G.nodes[i]['color'] = 'black'
                else:
                    G.nodes[i]['color'] = 'grey'
                try:
                    G.nodes[i]['label'] = str(objects.loc[i, 0])
                except Exception:
                    G.nodes[i]['label'] = str(i)
            return

        node_types_df = pd.concat([
            sub_links[['obj1', 'link_type']].rename(columns={'obj1': 'obj'}),
            sub_links[['obj2', 'link_type']].rename(columns={'obj2': 'obj'}),
        ], ignore_index=True)
        node_types = node_types_df.groupby('obj')['link_type'].agg(lambda s: set(s.unique())).to_dict()
        a = np.minimum(sub_links['obj1'].to_numpy(), sub_links['obj2'].to_numpy())
        b = np.maximum(sub_links['obj1'].to_numpy(), sub_links['obj2'].to_numpy())
        edge_meta = sub_links.assign(_a=a, _b=b)
        edge_types = edge_meta.groupby(['_a', '_b'])['link_type'].agg(lambda s: set(s.unique())).to_dict()
        edge_loss_idx = (
            edge_meta[edge_meta['Loss_idx'].notna()]
            .groupby(['_a', '_b'])['Loss_idx']
            .agg(lambda s: list(pd.unique(s)))
            .to_dict()
        )
        for i in G.nodes:
            temp = node_types.get(i, set())
            if 'Victim_Culprit' in temp or 'Culprit_Victim' in temp:
                G.nodes[i]['color'] = 'red'
            elif i > people_len:
                G.nodes[i]['color'] = 'black'
            else:
                G.nodes[i]['color'] = 'grey'
            G.nodes[i]['label'] = str(objects.loc[i, 0])
        for edge in G.edges:
            left, right = sorted(edge)
            key = (left, right)
            temp = edge_types.get(key, set())
            title = loss_title(edge_loss_idx.get(key, []))
            if 'Culprit_VINv' in temp or 'Victim_VINc' in temp:
                G[left][right].update(color='black', weight=0, title=title)
            elif 'VINv_VINc' in temp or 'VINc_VINv' in temp:
                G[left][right].update(color='black', weight=2, width=2, title=title)
            elif 'Victim_VINv' in temp or 'Culprit_VINc' in temp:
                G[left][right].update(color='black', weight=3, width=3, title=title)
            elif 'Victim_Culprit' in temp or 'Culprit_Victim' in temp:
                G[left][right].update(color='red', weight=4, width=4, title=title)

    objects = pd.DataFrame(objects)
    n_groups = len(big_groups[:_viz_top_n()])
    people_len = len(people)

    for group in range(n_groups):
        group_nodes = big_groups[group]
        sub_links = links[
            links['obj1'].isin(group_nodes) & links['obj2'].isin(group_nodes)
        ]
        G = nx.from_pandas_edgelist(sub_links, 'obj1', 'obj2', create_using=nx.Graph())
        G.remove_edges_from(nx.selfloop_edges(G))
        n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
        log(
            f'группа {group + 1}/{n_groups}:  {n_nodes:,} узлов, {n_edges:,} рёбер',
            level='info',
        )

        style_graph(G, sub_links, people_len)

        huge = n_edges > DETAILED_STYLE_MAX_EDGES
        if huge:
            save_gephi(G, f'Group_visualisation{group}')
        else:
            save_gephi(relabel_graph(G.copy()), f'Group_visualisation{group}')

        components = sorted(nx.connected_components(G), key=len, reverse=True)
        # Папка только если реально несколько компонент
        if len(components) > 1:
            html_dir = os.path.join(HTML_DIR, f'group_{group}')
            os.makedirs(html_dir, exist_ok=True)
        else:
            html_dir = HTML_DIR
            os.makedirs(html_dir, exist_ok=True)

        for comp_i, comp in enumerate(components):
            sub = G.subgraph(comp).copy()
            if sub.number_of_nodes() <= MAX_RELABEL_NODES:
                sub = relabel_graph(sub)
            if len(components) > 1:
                html_name = f'{group}_{comp_i}.html'
            else:
                html_name = f'Group_visualisation{group}.html'
            save_html(sub, os.path.join(html_dir, html_name))
        log(f'HTML-компонент: {len(components)}', level='info')

    log(f'Gephi → {GEPHI_DIR}', level='ok')
    log(f'HTML  → {HTML_DIR}', level='ok')
