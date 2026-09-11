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


def _hub_degree_n() -> int:
    """Порог degree > N; 0 = фильтр по степени выключен."""
    return max(0, int(getattr(config, 'hub_degree_n', 0)))


def _hub_keywords():
    """Ключевые слова среза (case-insensitive). Пустой = только degree-срез."""
    raw = getattr(config, 'hub_keywords', None)
    if raw is None:
        # старое имя списка
        raw = getattr(config, 'legal_forms_remove', None) or []
    return [str(x).strip() for x in raw if str(x).strip()]


def _clean_entity_label(label) -> str:
    text = str(label).strip()
    for bad in (' nan', ' NaT', ' None', ' NAN'):
        if text.endswith(bad):
            text = text[: -len(bad)].strip()
    if text.lower() in ('nan', 'none', 'nat'):
        return ''
    return text


def label_matches_keywords(label, keywords) -> bool:
    """Регистронезависимое вхождение любой подстроки из keywords в метку."""
    if not keywords:
        return False
    text = _clean_entity_label(label).casefold()
    if not text:
        return False
    return any(str(kw).strip().casefold() in text for kw in keywords if str(kw).strip())


def _node_label(objects_df: pd.DataFrame, node) -> str:
    try:
        return _clean_entity_label(objects_df.loc[node, 0])
    except Exception:
        return _clean_entity_label(node)


def collect_cut_nodes(graph: nx.Graph, objects_df: pd.DataFrame,
                      degree_n=None, keywords=None):
    """
    Узлы к удалению по правилам среза.

    - топ-хабы: degree > degree_n (если degree_n > 0), иначе все узлы;
    - если keywords непусты: seed = топ-хабы с совпадением по ключу;
      удаляем seed + всех соседей (рёбра уходят вместе с вершинами);
    - если keywords пусты: удаляем только сами топ-хабы (без соседей).
    """
    if degree_n is None:
        degree_n = _hub_degree_n()
    if keywords is None:
        keywords = _hub_keywords()
    if graph.number_of_nodes() == 0:
        return set(), set()

    deg = dict(graph.degree())
    if degree_n > 0:
        tops = {n for n, d in deg.items() if d > degree_n}
    else:
        tops = set(graph.nodes)

    if keywords:
        seeds = {
            n for n in tops
            if label_matches_keywords(_node_label(objects_df, n), keywords)
        }
        remove = set(seeds)
        for seed in seeds:
            remove.update(graph.neighbors(seed))
        return remove, seeds

    # без ключевых слов — классический срез только хабов по степени
    return set(tops), set(tops)


VARS_DIR = './vars'
OUTPUT_DIR = './output'
GEPHI_DIR = os.path.join(OUTPUT_DIR, 'gephi')
HTML_DIR = os.path.join(OUTPUT_DIR, 'html')
FINGERPRINT_PATH = os.path.join(VARS_DIR, 'input_fingerprint.json')
# Ниже этого порога — подробные title на рёбрах; выше — быстрая стилизация
DETAILED_STYLE_MAX_EDGES = 50_000
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


def run_pipeline(*, draw: bool = False):
    """Пайплайн: кэш или пересчёт → статистика. Отрисовка — отдельно.

    По умолчанию ``draw=False``: сначала смотрите ``show_degree_report()``,
    затем вызывайте ``visualize()`` (иначе на диск улетит куча HTML).

    В Jupyter: ``run_pipeline()``, не ``run()`` (конфликт с ``%run``).
    """
    top_n = _viz_top_n()
    keywords = _hub_keywords()
    log('Пайплайн графов', level='header')
    log(f'viz_top_n = {top_n}', level='info')
    log(f'hub_degree_n = {_hub_degree_n()}', level='info')
    log(
        f'hub_keywords = {len(keywords)} шт.'
        + (
            f' ({", ".join(keywords[:6])}…)'
            if len(keywords) > 6
            else (f' ({", ".join(keywords)})' if keywords else ' (выкл. → только degree)')
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
    log_cut_preview(group_index=0)
    if draw:
        log(f'отрисовка Gephi + HTML (топ-{top_n})', level='step')
        visualize()
        log('Готово', level='ok')
    else:
        log(
            'дальше: show_degree_report() → подберите N/keywords → visualize()',
            level='ok',
        )


# Для скриптов; в ноутбуке используйте run_pipeline()
run = run_pipeline


def log_cut_preview(group_index: int = 0) -> None:
    """Краткий лог: сколько компонент будет после среза (без записи на диск)."""
    try:
        ensure_runtime_state()
        graph = _subgraph_for_group(group_index)
        objects_df = pd.DataFrame(objects)
        before_n = graph.number_of_nodes()
        before_cc = nx.number_connected_components(graph)
        cut, info = apply_group_cuts(graph, objects_df)
        log(
            f'превью среза группы {group_index}: '
            f'{before_n:,} узлов / {before_cc} комп. → '
            f'{info["nodes_left"]:,} узлов / {info["components"]:,} комп. '
            f'(seed-хабов−{info["removed_seeds"]}, с соседями−{info["removed_total"]})',
            level='info',
        )
        if info['components'] > 500:
            log(
                'компонент очень много — сначала отчёт и пороги, '
                'visualize() пишет только компоненты размером ≥ bound',
                level='warn',
            )
    except Exception as exc:
        log(f'превью среза недоступно: {exc}', level='warn')


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
    Таблицы для подбора hub_degree_n и hub_keywords.

    Топ-хабов — уже ПОСЛЕ среза (кто остался).
    Возвращает (summary, top_hubs, cut_scenarios, config_result).
    """
    ensure_runtime_state()
    graph = _subgraph_for_group(group_index)
    people_count = len(people)
    objects_df = pd.DataFrame(objects)
    keywords = _hub_keywords()
    degree_n = _hub_degree_n()

    def _label(i):
        return _node_label(objects_df, i)

    def _node_kind(i, label: str) -> str:
        if i >= people_count:
            return 'VIN'
        if label_matches_keywords(label, keywords):
            return 'keyword-hit'
        return 'человек'

    deg = pd.Series(dict(graph.degree()), name='degree')
    if deg.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    labels = {i: _label(i) for i in deg.index}
    remove_set, seeds = collect_cut_nodes(graph, objects_df)

    def _pct(p):
        return float(np.percentile(deg.to_numpy(), p))

    p90, p95, p99 = _pct(90), _pct(95), _pct(99)
    mean_d = float(deg.mean())
    std_d = float(deg.std(ddof=0)) if len(deg) > 1 else 0.0
    mean_2std = mean_d + 2 * std_d
    n_cc_now = nx.number_connected_components(graph)

    summary = pd.DataFrame(
        [
            ('group_index', int(group_index)),
            ('узлов', int(graph.number_of_nodes())),
            ('рёбер', int(graph.number_of_edges())),
            ('компонент сейчас', int(n_cc_now)),
            ('hub_keywords (шт.)', int(len(keywords))),
            ('seed-хабов к срезу', int(len(seeds))),
            ('узлов к срезу (хаб+соседи)', int(len(remove_set))),
            ('degree min', int(deg.min())),
            ('degree median', round(float(deg.median()), 2)),
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

    def _apply_remove(base_graph, nodes_to_drop):
        H = base_graph.copy()
        drop = [n for n in nodes_to_drop if n in H]
        H.remove_nodes_from(drop)
        removed = len(drop)
        if H.number_of_nodes() == 0:
            return {
                'снято узлов': removed,
                'компонент после': 0,
                'размер топ-1': 0,
                'размер топ-5 (сумма)': 0,
            }, H
        comps = sorted(nx.connected_components(H), key=len, reverse=True)
        return {
            'снято узлов': removed,
            'компонент после': len(comps),
            'размер топ-1': len(comps[0]),
            'размер топ-5 (сумма)': sum(len(c) for c in comps[:5]),
        }, H

    cfg_stats, graph_after = _apply_remove(graph, remove_set)
    config_result = pd.DataFrame(
        [
            ('hub_degree_n (config)', degree_n),
            ('hub_keywords (шт.)', len(keywords)),
            ('seed-хабов', len(seeds)),
            ('снято узлов (хаб+соседи)', cfg_stats['снято узлов']),
            ('компонент после среза', cfg_stats['компонент после']),
            ('размер топ-1 после среза', cfg_stats['размер топ-1']),
            ('размер топ-5 (сумма)', cfg_stats['размер топ-5 (сумма)']),
        ],
        columns=['параметр', 'значение'],
    )

    if graph_after.number_of_nodes():
        deg_after = pd.Series(dict(graph_after.degree()), name='degree')
        top_hubs = (
            deg_after.sort_values(ascending=False)
            .head(top_n)
            .rename_axis('obj_idx')
            .reset_index()
        )
        top_hubs['метка'] = top_hubs['obj_idx'].map(lambda i: labels.get(i, _label(i)))
        top_hubs['тип'] = [
            _node_kind(i, labels.get(i, _label(i))) for i in top_hubs['obj_idx']
        ]
        top_hubs = top_hubs[['obj_idx', 'degree', 'тип', 'метка']]
    else:
        top_hubs = pd.DataFrame(columns=['obj_idx', 'degree', 'тип', 'метка'])

    scenarios = []
    for name, thr in [
        ('только keywords', 0),
        ('p90', p90),
        ('p95', p95),
        ('p99', p99),
        ('mean+2std', mean_2std),
        ('degree>50', 50),
        ('degree>100', 100),
        ('degree>200', 200),
        ('degree>500', 500),
        ('config hub_degree_n', degree_n),
    ]:
        thr_i = int(np.floor(thr))
        for use_kw, kw_tag in [(False, 'keywords: нет'), (True, 'keywords: да')]:
            if name == 'только keywords' and not use_kw:
                continue
            if name == 'только keywords' and not keywords:
                continue
            if use_kw:
                dn = 0 if name == 'только keywords' else thr_i
                drop, _ = collect_cut_nodes(
                    graph, objects_df, degree_n=dn, keywords=keywords,
                )
            else:
                drop, _ = collect_cut_nodes(
                    graph, objects_df, degree_n=thr_i, keywords=[],
                )
            stats, _ = _apply_remove(graph, drop)
            scenarios.append({
                'сценарий': name,
                'keywords': kw_tag,
                'порог N (degree > N)': 0 if name == 'только keywords' else thr_i,
                **stats,
            })
    cut_scenarios = pd.DataFrame(scenarios)
    return summary, top_hubs, cut_scenarios, config_result


def show_degree_report(group_index: int = 0, top_n: int = 40):
    """Краткий отчёт в Jupyter: итог по config + сценарии (с числом компонент)."""
    from IPython.display import display, Markdown

    summary, top_hubs, cut_scenarios, config_result = degree_report(
        group_index, top_n=top_n,
    )
    cfg_n = _hub_degree_n()
    keywords = _hub_keywords()
    n_cc = int(config_result.loc[
        config_result['параметр'] == 'компонент после среза', 'значение'
    ].iloc[0]) if len(config_result) else 0
    top1 = int(config_result.loc[
        config_result['параметр'] == 'размер топ-1 после среза', 'значение'
    ].iloc[0]) if len(config_result) else 0
    kw_preview = ', '.join(keywords[:8]) + ('…' if len(keywords) > 8 else '')

    display(Markdown(
        f'### Краткий итог по `config` (группа {group_index})\n'
        f'- `hub_degree_n` = **{cfg_n}**\n'
        f'- `hub_keywords` = **{len(keywords)}**'
        f'{f" (`{kw_preview}`)" if keywords else " (пусто → режем только хабы по degree)"}\n'
        f'- после среза: **{n_cc:,}** компонент, топ-1 = **{top1:,}** узлов\n'
        f'- срез: топ-хабы по ключу + **все их соседи**; регистр ключей не важен'
    ))
    display(config_result)
    display(Markdown(
        '### Сценарии среза '
        '(колонка **«компонент после»** — сколько кусков получится)'
    ))
    display(cut_scenarios)
    display(Markdown('### Сводка по группе (до среза)'))
    display(summary)
    display(Markdown(
        f'### Топ-{top_n} хабов **после** среза (удалённые не показываем)'
    ))
    display(top_hubs)
    return summary, top_hubs, cut_scenarios, config_result



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
        
def apply_group_cuts(graph: nx.Graph, objects_df: pd.DataFrame):
    """
    Срез по config: топ-хабы (degree > N) по hub_keywords + их соседи.

    Если keywords пусты — удаляются только сами хабы по степени.
    """
    H = graph.copy()
    remove, seeds = collect_cut_nodes(H, objects_df)
    if remove:
        H.remove_nodes_from([n for n in remove if n in H])
    n_cc = nx.number_connected_components(H) if H.number_of_nodes() else 0
    info = {
        'removed_seeds': len(seeds),
        'removed_total': len(remove),
        'removed_orgs': len(seeds),  # alias для старых логов
        'removed_hubs': max(0, len(remove) - len(seeds)),
        'nodes_left': H.number_of_nodes(),
        'components': n_cc,
    }
    return H, info


def visualize():
    global links, data, big_groups, objects, people, VIN, ID_col

    ensure_runtime_state()
    ensure_artifact_dirs()
    clear_graph_outputs()
    ensure_artifact_dirs()
    log('Отрисовка Gephi/HTML', level='header')
    log_cut_preview(group_index=0)

    def loss_text(loss_ids, limit=3):
        """До ``limit`` номеров убытков через запятую (для title/label рёбер)."""
        if not loss_ids:
            return ''
        try:
            unique_ids = list(pd.unique(data.loc[loss_ids, ID_col]))[:limit]
        except Exception:
            return ''
        return ', '.join(str(v) for v in unique_ids if pd.notna(v))

    def short_person_label(full_text: str) -> str:
        """Фамилия + инициалы + дата рождения; VIN оставляем как есть."""
        raw = str(full_text).strip()
        if not raw or raw.lower() in ('nan', 'none', 'nat'):
            return raw
        parts = raw.split()
        if not parts:
            return raw
        bd = ''
        name_parts = parts
        last = parts[-1]
        if len(last) >= 8 and (('-' in last) or ('.' in last) or last.isdigit()):
            if last.lower() not in ('nan', 'nat', 'none'):
                bd = last[:10]
            name_parts = parts[:-1]
        if not name_parts:
            return bd or raw
        # VIN / короткий код без пробелов ФИО
        if len(name_parts) == 1 and len(name_parts[0]) <= 20 and not name_parts[0].isalpha():
            return name_parts[0] + (f' {bd}' if bd else '')
        surname = name_parts[0].capitalize()
        initials = ''.join(
            f'{p[0].upper()}.' for p in name_parts[1:3] if p
        )
        short = surname + (f' {initials}' if initials else '')
        if bd:
            short = f'{short} {bd}'
        return short

    def node_labels(obj_idx, people_len):
        """Полное имя (title) и короткое (label) для узла."""
        try:
            full = str(objects.loc[obj_idx, 0])
        except Exception:
            full = str(obj_idx)
        if obj_idx >= people_len:
            short = full if len(full) <= 17 else full[:17]
            return full, short
        return full, short_person_label(full)

    def _sanitize_for_gexf(graph):
        """Строковые атрибуты; weight всегда > 0 (иначе Gephi дропает рёбра)."""
        H = graph.copy()
        for node in H.nodes:
            for key, value in list(H.nodes[node].items()):
                if key in ('x', 'y', 'physics', 'width'):
                    H.nodes[node].pop(key, None)
                    continue
                H.nodes[node][key] = '' if value is None else str(value)
        for _u, _v, attrs in H.edges(data=True):
            for key, value in list(attrs.items()):
                if key == 'weight':
                    try:
                        w = float(value)
                    except (TypeError, ValueError):
                        w = 1.0
                    attrs[key] = max(w, 1.0)
                    continue
                if key == 'width':
                    attrs.pop(key, None)
                    continue
                attrs[key] = '' if value is None else str(value)
            if 'weight' not in attrs:
                attrs['weight'] = 1.0
            # Gephi показывает Label; title дублируем в label, если label пуст
            if not attrs.get('label') and attrs.get('title'):
                attrs['label'] = attrs['title']
        return H

    def save_html(graph, html_path):
        """HTML: подпись ФИО/VIN внутри фигуры узла (ellipse), title на hover."""
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

        def _color_opts(raw):
            """Цвет заливки + контрастный текст внутри узла."""
            name = str(raw or 'grey').lower()
            palette = {
                'red': ('#e74c3c', '#ffffff'),
                'black': ('#2c3e50', '#ffffff'),
                'grey': ('#bdc3c7', '#1a1a1a'),
                'gray': ('#bdc3c7', '#1a1a1a'),
            }
            bg, fg = palette.get(name, (name, '#1a1a1a'))
            return {
                'background': bg,
                'border': bg,
                'highlight': {'background': bg, 'border': '#111'},
            }, fg

        def _inner_label(text: str) -> str:
            """Разбить подпись на 2 строки, чтобы влезала внутрь ellipse."""
            parts = str(text).split()
            if len(parts) <= 1:
                return str(text)
            if len(parts) == 2:
                return f'{parts[0]}\n{parts[1]}'
            return f'{" ".join(parts[:-1])}\n{parts[-1]}'

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
                label = str(attrs.get('label') or node)
                title = str(attrs.get('title') or label)
                color_opts, font_color = _color_opts(attrs.get('color', 'grey'))
                item = {
                    'id': str(node),
                    'label': _inner_label(label),
                    'title': title,
                    'shape': 'ellipse',
                    'color': color_opts,
                    'font': {
                        'size': 11,
                        'face': 'Arial',
                        'color': font_color,
                        'align': 'center',
                        'multi': True,
                    },
                    'margin': 12,
                    'widthConstraint': {'maximum': 160},
                    'heightConstraint': {'minimum': 36},
                }
                if use_physics:
                    item['physics'] = True
                else:
                    item['physics'] = False
                    item['x'] = float(i % side) * 160.0
                    item['y'] = float(i // side) * 90.0
                if not first:
                    fh.write(',\n')
                first = False
                json.dump(item, fh, ensure_ascii=False)
            fh.write('\n]);\nconst edges=new vis.DataSet([\n')
            first = True
            for u, v, attrs in graph.edges(data=True):
                edge_title = str(attrs.get('title') or '')
                edge_label = str(attrs.get('label') or edge_title)
                item = {
                    'from': str(u),
                    'to': str(v),
                    'color': attrs.get('color', '#999'),
                    'width': int(float(attrs.get('width', 1) or 1)),
                }
                if edge_label:
                    item['label'] = edge_label
                if edge_title:
                    item['title'] = edge_title
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
                'nodes:{shape:"ellipse",margin:12,'
                'font:{size:11,face:"Arial",align:"center",multi:true},'
                'widthConstraint:{maximum:160},scaling:{label:false}},'
                'edges:{font:{size:9,align:"middle",color:"#444"},'
                'smooth:{type:"continuous"}},'
                'interaction:{dragNodes:true,dragView:true,zoomView:true,hover:true}'
                '});\n</script></body></html>\n'
            )

    def save_gephi(graph, stem):
        if graph.number_of_nodes() == 0:
            return
        os.makedirs(GEPHI_DIR, exist_ok=True)
        gexf_path = os.path.join(GEPHI_DIR, f'{stem}.gexf')
        nx.write_gexf(_sanitize_for_gexf(graph), gexf_path)

    def _edge_loss_index(sub_links):
        a = np.minimum(sub_links['obj1'].to_numpy(), sub_links['obj2'].to_numpy())
        b = np.maximum(sub_links['obj1'].to_numpy(), sub_links['obj2'].to_numpy())
        edge_meta = sub_links.assign(_a=a, _b=b)
        edge_types = (
            edge_meta.groupby(['_a', '_b'])['link_type']
            .agg(lambda s: set(s.unique()))
            .to_dict()
        )
        edge_loss_idx = (
            edge_meta[edge_meta['Loss_idx'].notna()]
            .groupby(['_a', '_b'])['Loss_idx']
            .agg(lambda s: list(pd.unique(s)))
            .to_dict()
        )
        return edge_types, edge_loss_idx

    def style_graph(G, sub_links, people_len):
        """Цвет + короткие label / полные title; weight всегда ≥ 1."""
        n_edges = G.number_of_edges()
        edge_types, edge_loss_idx = _edge_loss_index(sub_links)

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
                full, short = node_labels(i, people_len)
                G.nodes[i]['title'] = full
                G.nodes[i]['label'] = short
            for edge in G.edges:
                left, right = sorted(edge)
                losses = loss_text(edge_loss_idx.get((left, right), []))
                G[left][right].update(
                    color='black', weight=1.0, width=1,
                    title=losses, label=losses,
                )
            return

        node_types_df = pd.concat([
            sub_links[['obj1', 'link_type']].rename(columns={'obj1': 'obj'}),
            sub_links[['obj2', 'link_type']].rename(columns={'obj2': 'obj'}),
        ], ignore_index=True)
        node_types = (
            node_types_df.groupby('obj')['link_type']
            .agg(lambda s: set(s.unique()))
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
            full, short = node_labels(i, people_len)
            G.nodes[i]['title'] = full
            G.nodes[i]['label'] = short
        for edge in G.edges:
            left, right = sorted(edge)
            key = (left, right)
            temp = edge_types.get(key, set())
            losses = loss_text(edge_loss_idx.get(key, []))
            # Gephi: weight должен быть > 0
            if 'Culprit_VINv' in temp or 'Victim_VINc' in temp:
                G[left][right].update(
                    color='black', weight=1.0, width=1, title=losses, label=losses,
                )
            elif 'VINv_VINc' in temp or 'VINc_VINv' in temp:
                G[left][right].update(
                    color='black', weight=2.0, width=2, title=losses, label=losses,
                )
            elif 'Victim_VINv' in temp or 'Culprit_VINc' in temp:
                G[left][right].update(
                    color='black', weight=3.0, width=3, title=losses, label=losses,
                )
            elif 'Victim_Culprit' in temp or 'Culprit_Victim' in temp:
                G[left][right].update(
                    color='red', weight=4.0, width=4, title=losses, label=losses,
                )
            else:
                G[left][right].update(
                    color='grey', weight=1.0, width=1, title=losses, label=losses,
                )

    objects = pd.DataFrame(objects)
    n_groups = len(big_groups[:_viz_top_n()])
    people_len = len(people)
    log(
        f'срез: hub_degree_n={_hub_degree_n()}, '
        f'hub_keywords={len(_hub_keywords())} шт.',
        level='info',
    )

    for group in range(n_groups):
        group_nodes = big_groups[group]
        sub_links = links[
            links['obj1'].isin(group_nodes) & links['obj2'].isin(group_nodes)
        ]
        G = nx.from_pandas_edgelist(sub_links, 'obj1', 'obj2', create_using=nx.Graph())
        G.remove_edges_from(nx.selfloop_edges(G))
        n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
        G, cut_info = apply_group_cuts(G, objects)
        sub_links = sub_links[
            sub_links['obj1'].isin(G.nodes) & sub_links['obj2'].isin(G.nodes)
        ]
        log(
            f'группа {group + 1}/{n_groups}:  было {n_nodes:,} узлов / {n_edges:,} рёбер → '
            f'осталось {cut_info["nodes_left"]:,}; seed={cut_info["removed_seeds"]}, '
            f'всего снято={cut_info["removed_total"]}; компонент={cut_info["components"]}',
            level='info',
        )
        if G.number_of_nodes() == 0:
            log('группа пуста после среза — пропуск', level='warn')
            continue

        style_graph(G, sub_links, people_len)
        save_gephi(G, f'Group_visualisation{group}')

        components = sorted(nx.connected_components(G), key=len, reverse=True)
        # На диск — только компоненты ≥ bound (иначе после среза хабов будет десятки тысяч файлов)
        export_comps = [c for c in components if len(c) >= bound]
        skipped = len(components) - len(export_comps)
        if skipped:
            log(
                f'HTML: пишем {len(export_comps)} комп. (≥{bound}), '
                f'пропуск мелких: {skipped}',
                level='info',
            )
        if not export_comps:
            log('нет компонент ≥ bound для HTML', level='warn')
            continue

        if len(export_comps) > 1:
            html_dir = os.path.join(HTML_DIR, f'group_{group}')
            os.makedirs(html_dir, exist_ok=True)
        else:
            html_dir = HTML_DIR
            os.makedirs(html_dir, exist_ok=True)

        for comp_i, comp in enumerate(export_comps):
            sub = G.subgraph(comp).copy()
            if len(export_comps) > 1:
                html_name = f'{group}_{comp_i}.html'
            else:
                html_name = f'Group_visualisation{group}.html'
            save_html(sub, os.path.join(html_dir, html_name))
        log(f'HTML записано: {len(export_comps)}', level='info')

    log(f'Gephi → {GEPHI_DIR}', level='ok')
    log(f'HTML  → {HTML_DIR}', level='ok')
