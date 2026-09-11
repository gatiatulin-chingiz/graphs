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
from tqdm import tqdm_notebook
from pyvis.network import Network

from . import config

wrong_people_val = config.wrong_people_val
wrong_auto_val = config.wrong_auto_val
bound = config.bound

VARS_DIR = './vars'
OUTPUT_DIR = './output'
GEPHI_DIR = os.path.join(OUTPUT_DIR, 'gephi')
HTML_DIR = os.path.join(OUTPUT_DIR, 'html')
FINGERPRINT_PATH = os.path.join(VARS_DIR, 'input_fingerprint.json')
VIZ_TOP_N = 20
REQUIRED_ARTIFACTS = (
    'data', 'people', 'VIN', 'objects', 'columns',
    'links', 'G', 'groups', 'stat',
)


def clear_graph_outputs():
    """Удалить старые HTML и Gephi перед новым расчётом/визуализацией."""
    print('Удаляю старые HTML/Gephi графы...')
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
    print('Входные Excel/SQL не изменились — загружаю сохранённые артефакты из ./vars')
    load_objects()
    load_links()
    load_statistics()
    return True, fingerprint


def run_pipeline():
    """Пайплайн без GUI: кэш или полный пересчёт, затем визуализация и Gephi.

    В Jupyter вызывайте ``run_pipeline()``, не ``run()``:
    имя ``run`` пересекается с magic ``%run``.
    """
    ensure_artifact_dirs()
    clear_graph_outputs()
    # clear_graph_outputs сносит html/gephi — восстановить пустые каталоги
    ensure_artifact_dirs()
    cached, fingerprint = try_load_cached_artifacts()
    if not cached:
        print('Вход изменился или нет кэша — полный пересчёт...')
        load()
        create_links()
        create_statistics()
        _save_fingerprint(fingerprint)
        print('Отпечаток входа сохранён в ./vars/input_fingerprint.json')
    visualize()


# Для скриптов; в ноутбуке используйте run_pipeline()
run = run_pipeline


def load():
    print('Подгрузка данных из Excel...')
    choice_query()


def choice_query():
    global data
    onlyfiles = [
        f for f in os.listdir(os.getcwd())
        if os.path.isfile(os.path.join(os.getcwd(), f)) and f.lower().endswith(('.xlsx', '.xls'))
    ]
    print('Вижу следующие excel-файлы:')
    for name in onlyfiles:
        print('\t' + name)
    if not onlyfiles:
        raise FileNotFoundError('В рабочей папке нет xlsx/xls-файлов')
    print('Подгружаю данные...')
    max_workers = min(4, len(onlyfiles))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        frames = list(tqdm_notebook(pool.map(pd.read_excel, onlyfiles), total=len(onlyfiles)))
    data = pd.concat(frames, axis=0, ignore_index=True)
    print('Данные подгружены ✅')
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
            print(f'Столбца {i} нет в данных ❌')
            error = 1
    if error == 1:
        return -1
    
    # Удаление дубликатов
    print('1. Предобработка данных...')
    data[ID_col] = data[ID_col].astype(int)
    temp = data[[ID_col]].duplicated().mean()
    if temp > 0:
        print('\tУдаление дубликатов...')
        data.drop_duplicates(inplace=True)
    data.reset_index(drop=True, inplace=True)
    print('\t\tДубликаты удалены.')
    
    # Удаление аномалий в людях
    print('\tУдаление аномалий...')
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
    print('\t\tАномалии удалены.')
    
    # Изменение типа в полях "ден
    print('   Формирование объектов...')
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
    print('\tОбъекты сфомированы.')
    
    print('   Сохранение объектов...')
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
    print('\tОбъекты сохранены.')
    print('Готово ✅')

def load_objects():
    global data, people, VIN, objects
    global ID_col, FIO_cul_col, BD_cul_col, FIO_vic_col, BD_vic_col, FIO_pol_col, BD_pol_col
    global FIO_cul_obj_own_col, BD_cul_obj_own_col, FIO_vic_obj_own_col, BD_vic_obj_own_col, FIO_pol_obj_own_col, BD_pol_obj_own_col
    global FIO_pol_driv_col, BD_pol_driv_col, FIO_ref_rec, BD_ref_rec
    global VIN_cul_col, VIN_vic_col, Filial_col, Reject_col, Sum_col, FIO_vic_pol_col, BD_vic_pol_col
    global persons

    print('2. Подгрузка объектов...')
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
    print('\tОбъекты подгружены.')
    print('Готово ✅')
    
def create_links():
    global data, people, VIN, objects, links
    global ID_col, FIO_cul_col, BD_cul_col, FIO_vic_col, BD_vic_col, FIO_pol_col, BD_pol_col
    global FIO_cul_obj_own_col, BD_cul_obj_own_col, FIO_vic_obj_own_col, BD_vic_obj_own_col, FIO_pol_obj_own_col, BD_pol_obj_own_col
    global FIO_pol_driv_col, BD_pol_driv_col, FIO_ref_rec, BD_ref_rec
    global VIN_cul_col, VIN_vic_col, Filial_col, Reject_col, Sum_col, FIO_vic_pol_col, BD_vic_pol_col
    global persons
    
    print('3. Создание связей...')
    main_cols = ['Victim', 'Culprit', 'CulpritPolicyholder',
                 'VictimObjectOwner', 'CulpritObjectOwner', 'PolicyholderObjectOwner',
                 'PolicyholderDriver', 'PaymentRecipient', 'VINv', 'VINc', 'VictimPolicyholder']
    # Векторизация по парам колонок вместо тройного Python-цикла по строкам
    object_idxes = pd.Series(data=objects.index, index=objects.values)
    loss_idx = data.index.to_numpy()
    parts = []
    col_pairs = list(combinations(main_cols, 2))
    for col, link_col in tqdm_notebook(col_pairs):
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
    print('\tСвязи созданы.')
    print('Готово ✅')
    
def load_links():
    global links
    
    print('4. Подгрузка связей...')
    links = pickle.load(open('./vars/links', 'rb'))
    print('\tСвязи подгружены')
    print('Готово ✅')

def create_statistics():
    global links, data, big_groups, stat, objects, bound
    
    print('5. Формирование статистики...')
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
    for group_id, group in enumerate(tqdm_notebook(big_groups), start=1):
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
    print('./output/statistics.xlsx')
    print('\tСтатистика сформирована.')
    print('Готово ✅')
    
    
def load_statistics():
    global links, data, big_groups, stat, objects, bound
    
    print('5. Подгрузка статистики...')
    G = pickle.load(open('./vars/G', 'rb'))
    groups = pickle.load(open('./vars/groups', 'rb'))
    data = pickle.load(open('./vars/data', 'rb'))
    stat = pickle.load(open('./vars/stat', 'rb'))
    
    big_groups = [(i) for i in groups if len(i) > bound]
    objects = pd.DataFrame(objects)
    
    print('Готово ✅')
        
def visualize():
    global links, data, big_groups, objects, people, VIN, ID_col

    def loss_title(loss_ids):  # noqa: E306 — nested helper
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
        """Gephi/GEXF плохо переваривает нестроковые атрибуты."""
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

    def save_html(graph, html_path):
        if graph.number_of_nodes() == 0:
            return
        pos = nx.spring_layout(graph, seed=42)
        scale = 1000.0
        for node, (x, y) in pos.items():
            graph.nodes[node]['x'] = float(x) * scale
            graph.nodes[node]['y'] = float(y) * scale
            graph.nodes[node]['physics'] = False

        os.makedirs(os.path.dirname(html_path) or '.', exist_ok=True)
        nt = Network(
            '100vh',
            '100%',
            bgcolor='#ffffff',
            font_color='black',
            select_menu=False,
            filter_menu=False,
            cdn_resources='remote',
        )
        nt.from_nx(graph)
        nt.toggle_physics(False)
        nt.set_options(
            """{
              "physics": {"enabled": false},
              "interaction": {
                "dragNodes": true,
                "dragView": true,
                "zoomView": true
              }
            }"""
        )
        nt.show(html_path, notebook=False)
        print(f'HTML сохранён: {html_path}')

    def save_gephi(graph, stem):
        """Полный граф группы без разбиения на компоненты."""
        if graph.number_of_nodes() == 0:
            return
        os.makedirs(GEPHI_DIR, exist_ok=True)
        gexf_path = os.path.join(GEPHI_DIR, f'{stem}.gexf')
        nx.write_gexf(_sanitize_for_gexf(graph), gexf_path)
        print(f'Gephi GEXF сохранён: {gexf_path}')

    print('6. Визуализация данных (HTML + Gephi)...')
    objects = pd.DataFrame(objects)

    for group in range(len(big_groups[:VIZ_TOP_N])):
        group_nodes = big_groups[group]
        sub_links = links[links['obj1'].isin(group_nodes)]
        G = nx.from_pandas_edgelist(sub_links, 'obj1', 'obj2', create_using=nx.Graph())
        G.remove_edges_from(nx.selfloop_edges(G))

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

        people_len = len(people)
        for i in tqdm_notebook(G.nodes):
            temp = node_types.get(i, set())
            if 'Victim_Culprit' in temp or 'Culprit_Victim' in temp:
                G.nodes[i]['color'] = 'red'
                G.nodes[i]['title'] = 'ФИО+ДР: ' + str(objects.loc[i, 0]) + '\n'
            elif i > people_len:
                G.nodes[i]['color'] = 'black'
                G.nodes[i]['title'] = 'VIN: ' + str(objects.loc[i, 0]) + '\n'
            else:
                G.nodes[i]['color'] = 'grey'
                G.nodes[i]['title'] = 'ФИО+ДР: ' + str(objects.loc[i, 0]) + '\n'

        for edge in tqdm_notebook(G.edges):
            left, right = sorted(edge)
            key = (left, right)
            temp = edge_types.get(key, set())
            title = loss_title(edge_loss_idx.get(key, []))

            if 'Culprit_VINv' in temp or 'Victim_VINc' in temp:
                G[left][right]['color'] = 'black'
                G[left][right]['weight'] = 0
                G[left][right]['title'] = title
            elif 'VINv_VINc' in temp or 'VINc_VINv' in temp:
                G[left][right]['color'] = 'black'
                G[left][right]['weight'] = 2
                G[left][right]['width'] = 2
                G[left][right]['title'] = title
            elif 'Victim_VINv' in temp or 'Culprit_VINc' in temp:
                G[left][right]['color'] = 'black'
                G[left][right]['weight'] = 3
                G[left][right]['width'] = 3
                G[left][right]['title'] = title
            elif 'Victim_Culprit' in temp or 'Culprit_Victim' in temp:
                G[left][right]['color'] = 'red'
                G[left][right]['weight'] = 4
                G[left][right]['width'] = 4
                G[left][right]['title'] = title

        # Gephi — полный граф группы (без разбиения на компоненты)
        full = relabel_graph(G.copy())
        save_gephi(full, f'Group_visualisation{group}')

        # HTML — все группы бьём на компоненты, каждая в своей папке
        group_html_dir = os.path.join(HTML_DIR, f'group_{group}')
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        for comp_i, comp in enumerate(components):
            H = relabel_graph(G.subgraph(comp).copy())
            save_html(
                H,
                os.path.join(group_html_dir, f'{group}_{comp_i}.html'),
            )

    print('Визуализация завершена ✅')
