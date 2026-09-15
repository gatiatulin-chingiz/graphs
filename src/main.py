import os
import re
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

_WORD_CHAR = r'0-9A-Za-zА-Яа-яЁё'

# Кэш скомпилированных regex для hub_keywords: tuple(kws) -> list[Pattern]
_KEYWORD_PATTERN_CACHE: dict[tuple, list] = {}


def _normalize_keyword(kw: str) -> str:
    """Strip кавычек/дефисов с краёв, схлопнуть пробелы, casefold."""
    text = str(kw).strip()
    text = text.strip('\'"`«»')
    text = text.strip('-–—')
    text = re.sub(r'\s+', ' ', text).strip()
    return text.casefold()


def _hub_keywords():
    """Ключевые слова whitelist (нормализованные, без пустых/дублей)."""
    raw = getattr(config, 'hub_keywords', None) or []
    seen = set()
    out = []
    for item in raw:
        norm = _normalize_keyword(item)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _keyword_patterns(keywords=None):
    """Скомпилированные границы целого слова/фразы (с кэшем)."""
    kws = tuple(keywords if keywords is not None else _hub_keywords())
    cached = _KEYWORD_PATTERN_CACHE.get(kws)
    if cached is not None:
        return cached
    patterns = []
    for phrase in kws:
        if not phrase:
            continue
        patterns.append(re.compile(
            rf'(?<![{_WORD_CHAR}])'
            + re.escape(phrase)
            + rf'(?![{_WORD_CHAR}])'
        ))
    _KEYWORD_PATTERN_CACHE[kws] = patterns
    return patterns


def _clean_entity_label(label) -> str:
    text = str(label).strip()
    for bad in (' nan', ' NaT', ' None', ' NAN'):
        if text.endswith(bad):
            text = text[: -len(bad)].strip()
    if text.lower() in ('nan', 'none', 'nat'):
        return ''
    return text


def label_matches_keywords(label, keywords=None) -> bool:
    """
    Регистронезависимое совпадение целого слова/фразы из keywords.

    'ооо' → ООО "Ромашка" да; ПОООРФЕНОВ — нет.
    keywords=None → берётся hub_keywords из config.
    """
    patterns = _keyword_patterns(keywords)
    if not patterns:
        return False
    text = _clean_entity_label(label).casefold()
    if not text:
        return False
    return any(p.search(text) for p in patterns)


def _node_label(objects_df: pd.DataFrame, node) -> str:
    try:
        return _clean_entity_label(objects_df.loc[node, 0])
    except Exception:
        return _clean_entity_label(node)


VARS_DIR = './vars'
OUTPUT_DIR = './output'
GEPHI_DIR = os.path.join(OUTPUT_DIR, 'gephi')
HTML_DIR = os.path.join(OUTPUT_DIR, 'html')
FINGERPRINT_PATH = os.path.join(VARS_DIR, 'input_fingerprint.json')
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


def run_pipeline(*, draw: bool = False):
    """Пайплайн: кэш или пересчёт → статистика. Отрисовка — отдельно.

    По умолчанию ``draw=False``. Дальше в ноутбуке:
    ``show_hub_audit()`` → ``show_fraud_candidates()`` → ``visualize_fraud()``.

    В Jupyter: ``run_pipeline()``, не ``run()`` (конфликт с ``%run``).
    """
    keywords = _hub_keywords()
    log('Пайплайн графов', level='header')
    log(f'suspect_degree_n = {int(getattr(config, "suspect_degree_n", 20))}', level='info')
    log(
        f'fraud_top_n_per_type = {getattr(config, "fraud_top_n_per_type", 15)}',
        level='info',
    )
    log(
        f'hub_keywords (whitelist) = {len(keywords)} шт.'
        + (
            f' ({", ".join(keywords[:6])}…)'
            if len(keywords) > 6
            else (f' ({", ".join(keywords)})' if keywords else '')
        ),
        level='info',
    )
    ensure_artifact_dirs()
    cached, fingerprint = try_load_cached_artifacts()
    if not cached:
        log('1/3  Загрузка и предобработка Excel', level='step')
        load()
        log('2/3  Построение связей', level='step')
        create_links()
        log('3/3  Статистика → ./output/statistics.xlsx', level='step')
        create_statistics()
        _save_fingerprint(fingerprint)
    else:
        log('1–3/3  Пересчёт пропущен (кэш)', level='ok')
    try:
        from .fraud import clear_fraud_cache
        clear_fraud_cache()
    except Exception:
        pass
    if draw:
        log('отрисовка fraud HTML', level='step')
        visualize()
        log('Готово', level='ok')
    else:
        log(
            'дальше: show_hub_audit() → show_fraud_candidates() → visualize_fraud()',
            level='ok',
        )


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
    """Алиас: отрисовка fraud-кандидатов (см. visualize_fraud)."""
    from .fraud import visualize_fraud
    return visualize_fraud(group_index=0)


# Fraud-пайплайн (удобные реэкспорты для ноутбука)
def show_hub_audit(*args, **kwargs):
    from .fraud import show_hub_audit as _f
    return _f(*args, **kwargs)


def show_fraud_candidates(*args, **kwargs):
    from .fraud import show_fraud_candidates as _f
    return _f(*args, **kwargs)


def visualize_fraud(*args, **kwargs):
    from .fraud import visualize_fraud as _f
    return _f(*args, **kwargs)


def hub_audit_report(*args, **kwargs):
    from .fraud import hub_audit_report as _f
    return _f(*args, **kwargs)


def show_quick_slices(*args, **kwargs):
    from .fraud import show_quick_slices as _f
    return _f(*args, **kwargs)
