"""Поиск мошеннических кейсов: аудит хабов, кластеры, скоринг, viz."""

from __future__ import annotations

import json
import os
import re
import zipfile
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from . import config
from .main import (
    GEPHI_DIR,
    HTML_DIR,
    bound,
    ensure_artifact_dirs,
    ensure_runtime_state,
    label_matches_keywords,
    log,
    _clean_entity_label,
    _hub_keywords,
    _node_label,
)


FRAUD_HTML_DIR = os.path.join(HTML_DIR, 'fraud')
FRAUD_GEPHI_DIR = os.path.join(GEPHI_DIR, 'fraud')

# Кэши между ячейками ноутбука (сбрасываются в clear_fraud_cache / run_pipeline)
_BUNDLE_CACHE: dict[int, dict] = {}
_OBJECT_META_CACHE: tuple | None = None  # (key, meta_dict)
_GROUP_LINKS_CACHE: dict[int, tuple] = {}  # group_index -> (graph, group_links, link_index)


def _cfg(name, default):
    return getattr(config, name, default)


def clear_fraud_cache() -> None:
    """Сброс кэшей fraud после перезагрузки vars / смены config."""
    global _BUNDLE_CACHE, _OBJECT_META_CACHE, _GROUP_LINKS_CACHE
    _BUNDLE_CACHE = {}
    _OBJECT_META_CACHE = None
    _GROUP_LINKS_CACHE = {}


def _people_count() -> int:
    from . import main as m
    return len(m.people)


def _objects_df() -> pd.DataFrame:
    from . import main as m
    return pd.DataFrame(m.objects)


def _links_df() -> pd.DataFrame:
    from . import main as m
    return m.links


def _membership_mask(obj1: np.ndarray, obj2: np.ndarray, node_set: set) -> np.ndarray:
    """Быстрый mask: оба конца ребра ∈ node_set."""
    if not node_set or len(obj1) == 0:
        return np.zeros(len(obj1), dtype=bool)
    mx = int(max(max(node_set), int(obj1.max()), int(obj2.max()))) + 1
    if mx > 8_000_000:
        ns = node_set
        return np.fromiter(
            (a in ns and b in ns for a, b in zip(obj1, obj2)),
            dtype=bool,
            count=len(obj1),
        )
    mem = np.zeros(mx, dtype=bool)
    for n in node_set:
        if 0 <= n < mx:
            mem[n] = True
    return mem[obj1] & mem[obj2]


def _build_link_index(sub_links: pd.DataFrame) -> dict:
    """node -> np.array позиций строк в sub_links."""
    if sub_links.empty:
        return {}
    o1 = sub_links['obj1'].to_numpy()
    o2 = sub_links['obj2'].to_numpy()
    buckets: dict[int, list] = defaultdict(list)
    for i in range(len(o1)):
        buckets[int(o1[i])].append(i)
        buckets[int(o2[i])].append(i)
    return {k: np.asarray(v, dtype=np.int32) for k, v in buckets.items()}


def _group_graph_and_links(group_index: int):
    """Подграф группы + links + индекс инцидентности (с кэшем)."""
    if group_index in _GROUP_LINKS_CACHE:
        return _GROUP_LINKS_CACHE[group_index]

    ensure_runtime_state()
    from . import main as m
    if group_index < 0 or group_index >= len(m.big_groups):
        raise IndexError(f'group_index должен быть 0…{len(m.big_groups) - 1}')

    group_nodes = set(m.big_groups[group_index])
    links = m.links
    o1 = links['obj1'].to_numpy()
    o2 = links['obj2'].to_numpy()
    mask = _membership_mask(o1, o2, group_nodes)
    group_links = links.loc[mask].reset_index(drop=True)
    graph = nx.from_pandas_edgelist(
        group_links, 'obj1', 'obj2', create_using=nx.Graph(),
    )
    graph.remove_edges_from(nx.selfloop_edges(graph))
    link_index = _build_link_index(group_links)
    _GROUP_LINKS_CACHE[group_index] = (graph, group_links, link_index)
    return _GROUP_LINKS_CACHE[group_index]


def _object_meta(nodes=None) -> dict:
    """
    Предрасчёт label / keyword_hit / has_nan / ntype.

    Без nodes — все objects (редко). С nodes — только недостающие индексы (лениво).
    """
    global _OBJECT_META_CACHE
    from . import main as m
    objects_df = pd.DataFrame(m.objects)
    people_count = len(m.people)
    keywords = tuple(_hub_keywords())
    key = (len(objects_df), people_count, keywords)

    if _OBJECT_META_CACHE is None or _OBJECT_META_CACHE[0] != key:
        _OBJECT_META_CACHE = (key, {})

    meta = _OBJECT_META_CACHE[1]
    if nodes is None:
        todo = [i for i in range(len(objects_df)) if i not in meta]
    else:
        todo = [int(n) for n in nodes if int(n) not in meta]
    if not todo:
        return meta

    col = objects_df.iloc[:, 0]
    kw_list = list(keywords)
    for i in todo:
        try:
            raw = col.iloc[i] if i < len(col) else ''
        except Exception:
            raw = ''
        label = _clean_entity_label(raw)
        kw_hit = label_matches_keywords(label, kw_list) if kw_list else False
        if kw_hit:
            ntype = 'юрлицо'
        elif i >= people_count:
            ntype = 'VIN'
        else:
            ntype = 'человек'
        meta[i] = {
            'label': label,
            'keyword_hit': kw_hit,
            'has_nan_bd': has_nan_bd(label),
            'ntype': ntype,
        }
    return meta


def node_type(obj_idx: int, label: str, people_count: int, keywords) -> str:
    meta = _object_meta()
    if obj_idx in meta:
        return meta[obj_idx]['ntype']
    if label_matches_keywords(label, keywords):
        return 'юрлицо'
    if obj_idx >= people_count:
        return 'VIN'
    return 'человек'


def has_nan_bd(label: str) -> bool:
    parts = str(label).strip().split()
    if not parts:
        return False
    return parts[-1].lower() in ('nan', 'nat', 'none')


def classify_hub_basket(obj_idx, label, degree, ntype, keywords,
                        artifact_n, suspect_n, *, keyword_hit=None) -> str:
    if keyword_hit is None:
        keyword_hit = label_matches_keywords(label, keywords)
    if has_nan_bd(label) and degree > artifact_n // 2:
        return 'nan_glue'
    if keyword_hit or ntype == 'юрлицо':
        if keyword_hit:
            return 'whitelist_legal'
    if ntype in ('человек', 'VIN') and degree > suspect_n:
        return 'suspect_seed'
    if degree > artifact_n and ntype != 'юрлицо':
        return 'nan_glue' if has_nan_bd(label) else 'artifact'
    return 'other'


# ---------------------------------------------------------------------------
# 1. Аудит
# ---------------------------------------------------------------------------

def hub_audit_report(group_index: int = 0, top_n: int = 1000) -> pd.DataFrame:
    """Топ узлов по степени с типом и корзиной."""
    ensure_runtime_state()
    graph, _, _ = _group_graph_and_links(group_index)
    meta = _object_meta(graph.nodes)
    keywords = _hub_keywords()
    artifact_n = int(_cfg('artifact_degree_n', 1000))
    suspect_n = int(_cfg('suspect_degree_n', 20))

    deg = dict(graph.degree())
    rows = []
    for node, d in sorted(deg.items(), key=lambda x: -x[1])[:top_n]:
        info = meta.get(int(node), {})
        label = info.get('label') or _node_label(_objects_df(), node)
        ntype = info.get('ntype') or node_type(node, label, _people_count(), keywords)
        kw_hit = bool(info.get('keyword_hit', label_matches_keywords(label, keywords)))
        basket = classify_hub_basket(
            node, label, d, ntype, keywords, artifact_n, suspect_n,
            keyword_hit=kw_hit,
        )
        rows.append({
            'obj_idx': node,
            'degree': int(d),
            'тип': ntype,
            'has_nan_bd': bool(info.get('has_nan_bd', has_nan_bd(label))),
            'keyword_hit': kw_hit,
            'корзина': basket,
            'метка': label,
        })
    return pd.DataFrame(rows)


def show_hub_audit(group_index: int = 0, top_n: int = 40):
    from IPython.display import display, Markdown

    audit = hub_audit_report(group_index, top_n=max(top_n, 1000))
    counts = audit['корзина'].value_counts().rename_axis('корзина').reset_index(name='шт')
    display(Markdown(f'### Аудит топ-хабов группы {group_index}'))
    display(counts)
    display(Markdown(f'### Топ-{top_n} по degree'))
    display(audit.head(top_n))
    return audit


# ---------------------------------------------------------------------------
# 2. Подготовка графа
# ---------------------------------------------------------------------------

def strip_leaves_iterative(graph: nx.Graph) -> tuple[nx.Graph, int]:
    """2-core: эквивалент итеративного снятия degree<=1."""
    before = graph.number_of_nodes()
    if before == 0:
        return graph.copy(), 0
    H = nx.k_core(graph, k=2)
    return H, before - H.number_of_nodes()


def prepare_fraud_graph(group_index: int = 0):
    """
    Граф без whitelist/NaN-склеек + срез листьев.

    Возвращает (graph, meta, group_links, link_index).
    """
    graph, group_links, link_index = _group_graph_and_links(group_index)
    obj_meta = _object_meta(graph.nodes)
    keywords = _hub_keywords()
    artifact_n = int(_cfg('artifact_degree_n', 1000))
    suspect_n = int(_cfg('suspect_degree_n', 20))

    deg = dict(graph.degree())
    exclude = set()
    suspects = set()
    tags = defaultdict(list)

    for node in graph.nodes:
        info = obj_meta.get(int(node), {})
        label = info.get('label', '')
        d = deg.get(node, 0)
        ntype = info.get('ntype', 'человек')
        kw_hit = bool(info.get('keyword_hit', False))
        basket = classify_hub_basket(
            node, label, d, ntype, keywords, artifact_n, suspect_n,
            keyword_hit=kw_hit,
        )
        if basket in ('whitelist_legal', 'nan_glue', 'artifact'):
            exclude.add(node)
            for nb in graph.neighbors(node):
                tags[nb].append(f'{basket}:{node}')
        elif basket == 'suspect_seed':
            suspects.add(node)

    H = graph.copy()
    H.remove_nodes_from([n for n in exclude if n in H])
    H, n_leaves = strip_leaves_iterative(H)
    suspects = {n for n in suspects if n in H}

    # links только по оставшимся узлам (для скоринга кластеров)
    keep = set(H.nodes)
    if keep and not group_links.empty:
        o1 = group_links['obj1'].to_numpy()
        o2 = group_links['obj2'].to_numpy()
        mask = _membership_mask(o1, o2, keep)
        slim_links = group_links.loc[mask].reset_index(drop=True)
        slim_index = _build_link_index(slim_links)
    else:
        slim_links = group_links.iloc[0:0].copy()
        slim_index = {}

    meta = {
        'excluded': exclude,
        'suspects': suspects,
        'tags': dict(tags),
        'removed_leaves': n_leaves,
        'nodes_before': graph.number_of_nodes(),
        'nodes_after': H.number_of_nodes(),
        'components': nx.number_connected_components(H) if H.number_of_nodes() else 0,
    }
    log(
        f'fraud prep: {meta["nodes_before"]:,} → {meta["nodes_after"]:,} узлов; '
        f'exclude={len(exclude)}, leaves−{n_leaves}, seeds={len(suspects)}, '
        f'комп.={meta["components"]}',
        level='info',
    )
    return H, meta, slim_links, slim_index


# ---------------------------------------------------------------------------
# 3. Кластеры
# ---------------------------------------------------------------------------

def expand_from_seeds(graph: nx.Graph, seeds: set, rounds: int = 2) -> dict:
    """Label propagation: узел → seed с макс. весом связей."""
    if not seeds or graph.number_of_nodes() == 0:
        return {}
    assign = {s: s for s in seeds if s in graph}
    for _ in range(rounds):
        votes = defaultdict(lambda: defaultdict(float))
        for u, v in graph.edges:
            for src, dst in ((u, v), (v, u)):
                if src in assign:
                    votes[dst][assign[src]] += 1.0
        changed = False
        for node, hub_votes in votes.items():
            if node in seeds:
                continue
            best = max(hub_votes.items(), key=lambda x: x[1])[0]
            if assign.get(node) != best:
                assign[node] = best
                changed = True
        if not changed:
            break
    clusters = defaultdict(set)
    for node, hub in assign.items():
        clusters[hub].add(node)
        clusters[hub].add(hub)
    return dict(clusters)


def build_fraud_clusters(group_index: int = 0, *, force: bool = False):
    """Кластеры от suspect seeds + мини-CC остатка (кэш по group_index)."""
    if not force and group_index in _BUNDLE_CACHE:
        log(f'fraud clusters: кэш group={group_index}', level='info')
        return _BUNDLE_CACHE[group_index]

    H, meta, slim_links, slim_index = prepare_fraud_graph(group_index)
    max_nodes = int(_cfg('cluster_max_nodes', 500))
    seeds = meta['suspects']

    clusters = expand_from_seeds(H, seeds, rounds=2)
    covered = set()
    for members in clusters.values():
        covered |= members

    remainder = H.subgraph([n for n in H.nodes if n not in covered]).copy()
    mini = []
    needs_split = []
    for i, comp in enumerate(nx.connected_components(remainder)):
        if len(comp) < bound:
            continue
        if len(comp) <= max_nodes:
            mini.append(set(comp))
        else:
            needs_split.append(set(comp))

    node_to_cid = {}
    cluster_list = []
    for hub, members in clusters.items():
        cid = f'seed_{hub}'
        cluster_list.append({'cluster_id': cid, 'kind': 'seed', 'hub': hub, 'nodes': members})
        for n in members:
            node_to_cid[n] = cid
    for i, members in enumerate(mini):
        cid = f'mini_{i}'
        cluster_list.append({'cluster_id': cid, 'kind': 'mini', 'hub': None, 'nodes': members})
        for n in members:
            node_to_cid[n] = cid
    for i, members in enumerate(needs_split):
        cid = f'big_{i}'
        cluster_list.append({
            'cluster_id': cid, 'kind': 'needs_split', 'hub': None, 'nodes': members,
        })
        for n in members:
            node_to_cid[n] = cid

    bridges = []
    for n in H.nodes:
        neigh_cids = {node_to_cid[nb] for nb in H.neighbors(n) if nb in node_to_cid}
        own = node_to_cid.get(n)
        if own:
            neigh_cids.discard(own)
        if len(neigh_cids) >= 1 and own:
            for other in neigh_cids:
                bridges.append({'node': n, 'cluster_a': own, 'cluster_b': other})

    log(
        f'fraud clusters: seed={len(clusters)}, mini={len(mini)}, '
        f'needs_split={len(needs_split)}, bridges={len(bridges)}',
        level='info',
    )
    bundle = {
        'graph': H,
        'meta': meta,
        'clusters': cluster_list,
        'bridges': pd.DataFrame(bridges),
        'group_index': group_index,
        'group_links': slim_links,
        'link_index': slim_index,
        'ranked': None,
    }
    _BUNDLE_CACHE[group_index] = bundle
    return bundle


# ---------------------------------------------------------------------------
# 4–5. Признаки и типы
# ---------------------------------------------------------------------------

def _cluster_sublinks(nodes: set, bundle: dict | None = None) -> pd.DataFrame:
    nodes = set(nodes)
    if bundle is not None:
        sub_links = bundle.get('group_links')
        link_index = bundle.get('link_index') or {}
        if sub_links is not None and not sub_links.empty and link_index:
            parts = [link_index[n] for n in nodes if n in link_index]
            if not parts:
                return sub_links.iloc[0:0]
            idxs = np.unique(np.concatenate(parts))
            cand = sub_links.iloc[idxs]
            o1 = cand['obj1'].to_numpy()
            o2 = cand['obj2'].to_numpy()
            mask = _membership_mask(o1, o2, nodes)
            return cand.loc[mask]
        if sub_links is not None:
            if sub_links.empty:
                return sub_links
            o1 = sub_links['obj1'].to_numpy()
            o2 = sub_links['obj2'].to_numpy()
            mask = _membership_mask(o1, o2, nodes)
            return sub_links.loc[mask]
    links = _links_df()
    o1 = links['obj1'].to_numpy()
    o2 = links['obj2'].to_numpy()
    mask = _membership_mask(o1, o2, nodes)
    return links.loc[mask]


def _directed_vc_pairs(nodes: set, bundle: dict | None = None,
                       sub_links: pd.DataFrame | None = None) -> list[tuple[int, int]]:
    """Пары (виновник_idx, потерпевший_idx) внутри кластера по Loss_idx."""
    from . import main as m
    data = m.data
    obj_meta = _object_meta(nodes)
    people_count = _people_count()

    label_to_idx = {}
    for n in nodes:
        if n < people_count:
            lab = obj_meta.get(int(n), {}).get('label')
            if lab:
                label_to_idx[lab] = n

    if sub_links is None:
        sub_links = _cluster_sublinks(nodes, bundle)
    if sub_links.empty or 'Loss_idx' not in sub_links.columns:
        return []
    if 'Culprit' not in data.columns or 'Victim' not in data.columns:
        return []

    pairs = []
    for lid in sub_links['Loss_idx'].dropna().unique():
        try:
            row = data.loc[int(lid)]
        except Exception:
            continue
        c_lab = _clean_entity_label(row.get('Culprit', ''))
        v_lab = _clean_entity_label(row.get('Victim', ''))
        c_idx = label_to_idx.get(c_lab)
        v_idx = label_to_idx.get(v_lab)
        if c_idx is not None and v_idx is not None and c_idx != v_idx:
            pairs.append((c_idx, v_idx))
    return pairs


def _count_directed_triangles(pairs: list[tuple[int, int]]) -> int:
    succ = defaultdict(set)
    for a, b in pairs:
        succ[a].add(b)
    count = 0
    seen = set()
    for a, bs in succ.items():
        for b in bs:
            for c in succ.get(b, ()):
                if a in succ.get(c, ()) and a != c:
                    tri = tuple(sorted((a, b, c)))
                    if tri not in seen:
                        seen.add(tri)
                        count += 1
    return count


def _reciprocity(pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return 0.0
    s = set(pairs)
    mutual = sum(1 for a, b in s if (b, a) in s)
    return mutual / max(len(s), 1)


def _star_share(graph: nx.Graph, nodes: set) -> tuple[float, int | None]:
    sub = graph.subgraph(nodes)
    if sub.number_of_edges() == 0:
        return 0.0, None
    deg = dict(sub.degree())
    hub = max(deg, key=deg.get)
    through_hub = sum(1 for u, v in sub.edges if u == hub or v == hub)
    return through_hub / sub.number_of_edges(), hub


def _ego_density(graph: nx.Graph, hub) -> float:
    if hub is None or hub not in graph:
        return 0.0
    nbrs = list(graph.neighbors(hub))
    k = len(nbrs)
    if k < 2:
        return 0.0
    possible = k * (k - 1) / 2
    edges = graph.subgraph(nbrs).number_of_edges()
    return edges / possible


def _vin_bitki_features(nodes: set, people_count: int,
                        bundle: dict | None = None,
                        sub_links: pd.DataFrame | None = None) -> dict:
    """F4: пересечения исходящих пострадавших VIN."""
    from . import main as m
    data = m.data
    vins = {n for n in nodes if n >= people_count}
    if len(vins) < 2 or 'VINc' not in data.columns or 'VINv' not in data.columns:
        return {'bitki_jaccard_max': 0.0, 'vin_role_flip': 0.0, 'vin_star_flag': 0}

    obj_meta = _object_meta(vins)

    vin_label = {n: obj_meta.get(int(n), {}).get('label', '') for n in vins}
    label_to_vin = {v: k for k, v in vin_label.items() if v}

    out_victims = defaultdict(set)
    vin_as_fighter = defaultdict(int)
    vin_as_victim = defaultdict(int)
    vin_drivers = defaultdict(set)

    if sub_links is None:
        sub_links = _cluster_sublinks(nodes, bundle)
    loss_ids = sub_links['Loss_idx'].dropna().unique() if not sub_links.empty else []
    for lid in loss_ids:
        try:
            row = data.loc[int(lid)]
        except Exception:
            continue
        c_vin = _clean_entity_label(row.get('VINc', ''))
        v_vin = _clean_entity_label(row.get('VINv', ''))
        c_idx = label_to_vin.get(c_vin)
        v_idx = label_to_vin.get(v_vin)
        if c_idx is not None:
            vin_as_fighter[c_idx] += 1
            if 'Culprit' in row.index:
                vin_drivers[c_idx].add(_clean_entity_label(row.get('Culprit', '')))
        if v_idx is not None:
            vin_as_victim[v_idx] += 1
        if c_idx is not None and v_idx is not None and c_idx != v_idx:
            out_victims[c_idx].add(v_idx)

    j_max = 0.0
    fighters = list(out_victims.keys())
    for i in range(len(fighters)):
        for j in range(i + 1, len(fighters)):
            a, b = fighters[i], fighters[j]
            sa, sb = out_victims[a], out_victims[b]
            inter = len(sa & sb)
            union = len(sa | sb) or 1
            j_max = max(j_max, inter / union)

    flip = sum(
        1 for v in vins
        if vin_as_fighter[v] > 0 and vin_as_victim[v] > 0
    )
    flip_rate = flip / max(len(vins), 1)

    vin_star = 0
    for v in vins:
        if vin_as_fighter[v] + vin_as_victim[v] >= 5 and len(vin_drivers[v]) >= 5:
            vin_star = 1
            break

    return {
        'bitki_jaccard_max': float(j_max),
        'vin_role_flip': float(flip_rate),
        'vin_star_flag': vin_star,
    }


def score_cluster(graph: nx.Graph, nodes: set, people_count: int,
                  bundle: dict | None = None) -> dict:
    weights = dict(_cfg('fraud_score_weights', {}))
    sub = graph.subgraph(nodes)
    n = len(nodes)
    m = sub.number_of_edges()
    density = (2 * m) / (n * (n - 1)) if n > 1 else 0.0

    star_share, hub = _star_share(graph, nodes)
    ego_dens = _ego_density(graph, hub)

    sub_links = _cluster_sublinks(nodes, bundle)
    pairs = _directed_vc_pairs(nodes, bundle, sub_links=sub_links)
    cycles = _count_directed_triangles(pairs)
    recip = _reciprocity(pairs)

    undirected_tri = sum(1 for _ in nx.triangles(sub).values()) // 3 if n < 500 else 0

    if not sub_links.empty and 'Loss_idx' in sub_links.columns:
        a = np.minimum(sub_links['obj1'].to_numpy(), sub_links['obj2'].to_numpy())
        b = np.maximum(sub_links['obj1'].to_numpy(), sub_links['obj2'].to_numpy())
        w = (
            sub_links.assign(_a=a, _b=b)
            .groupby(['_a', '_b'])['Loss_idx']
            .nunique()
        )
        repeat_share = float((w >= 2).mean()) if len(w) else 0.0
    else:
        repeat_share = 0.0

    bitki = _vin_bitki_features(nodes, people_count, bundle, sub_links=sub_links)

    role_conc = 0.0
    if pairs:
        out_c = Counter(a for a, _ in pairs)
        role_conc = out_c.most_common(1)[0][1] / len(pairs)

    obj_meta = _object_meta(nodes)
    wl_share = sum(
        1 for x in nodes if obj_meta.get(int(x), {}).get('keyword_hit')
    ) / max(n, 1)
    nan_share = sum(
        1 for x in nodes if obj_meta.get(int(x), {}).get('has_nan_bd')
    ) / max(n, 1)

    size_fit = 1.0 if 3 <= n <= 50 else (0.5 if n <= 100 else 0.2)

    feats = {
        'size': n,
        'edges': m,
        'density': round(density, 4),
        'star_share': round(star_share, 4),
        'ego_density': round(ego_dens, 4),
        'directed_cycles': cycles,
        'undirected_triangles': undirected_tri,
        'reciprocity': round(recip, 4),
        'repeat_pairs': round(repeat_share, 4),
        'vin_star_flag': bitki['vin_star_flag'],
        'bitki_jaccard_max': round(bitki['bitki_jaccard_max'], 4),
        'vin_role_flip': round(bitki['vin_role_flip'], 4),
        'role_concentration': round(role_conc, 4),
        'whitelist_share': round(wl_share, 4),
        'nan_share': round(nan_share, 4),
        'size_fit': size_fit,
        'hub': hub,
    }

    score = (
        weights.get('repeat_pairs', 2) * feats['repeat_pairs']
        + weights.get('reciprocity', 2.5) * feats['reciprocity']
        + weights.get('directed_cycles', 3) * min(feats['directed_cycles'], 5) / 5
        + weights.get('vin_star', 2) * feats['vin_star_flag']
        + weights.get('bitki', 2.5) * max(feats['bitki_jaccard_max'], feats['vin_role_flip'])
        + weights.get('role_concentration', 1) * feats['role_concentration']
        + weights.get('density_bonus', 1) * min(feats['density'] * 10, 1)
        + weights.get('size_fit', 1) * feats['size_fit']
        + weights.get('whitelist_penalty', -2) * feats['whitelist_share']
    )
    if cycles == 0 and undirected_tri > 0:
        score += 1.0 * min(undirected_tri, 5) / 5

    feats['score'] = round(float(score), 3)
    feats['data_quality'] = round(1.0 - 0.5 * feats['nan_share'], 3)
    return feats


def classify_case_type(feats: dict) -> str:
    n = feats['size']
    if n <= 2 and feats['repeat_pairs'] < 0.1 and feats['directed_cycles'] == 0:
        return 'фоновая мелочь'
    if feats['bitki_jaccard_max'] >= 0.5 or (
        feats['vin_role_flip'] >= 0.3 and feats['vin_star_flag']
    ):
        return 'кольцо битов'
    if feats['directed_cycles'] >= 1 and feats['star_share'] < 0.8:
        return 'группа-колотуны'
    if feats['star_share'] >= 0.8 and feats['ego_density'] < 0.05:
        hub = feats.get('hub')
        obj_meta = _object_meta([hub] if hub is not None else [])
        if hub is not None:
            info = obj_meta.get(int(hub), {})
            if info.get('keyword_hit') and hub < _people_count():
                return 'юрлицо-хаб'
            if hub < _people_count():
                return 'соло с подставными'
            return 'кольцо битов' if feats['vin_star_flag'] else 'соло с подставными'
        return 'соло с подставными'
    if feats['directed_cycles'] >= 1 or feats['reciprocity'] >= 0.3:
        return 'группа-колотуны'
    if feats['star_share'] >= 0.7:
        return 'соло с подставными'
    return 'группа-колотуны' if n >= 3 else 'фоновая мелочь'


_TYPE_PRIORITY = {
    'группа-колотуны': 0,
    'кольцо битов': 1,
    'соло с подставными': 2,
    'юрлицо-хаб': 3,
    'фоновая мелочь': 9,
}

# Короткие ярлыки для имён файлов HTML/Gephi
_TYPE_FILE_SLUG = {
    'группа-колотуны': 'колотун',
    'кольцо битов': 'кольцевик',
    'соло с подставными': 'соло',
    'юрлицо-хаб': 'юрлицо',
    'фоновая мелочь': 'мелочь',
}


def _export_stem(cluster_id, case_type) -> str:
    """Имя файла: seed_123_колотун / mini_4_кольцевик."""
    base = str(cluster_id).replace('/', '_').replace('\\', '_')
    slug = _TYPE_FILE_SLUG.get(case_type)
    if not slug:
        slug = re.sub(r'[^\w\-]+', '_', str(case_type), flags=re.UNICODE).strip('_') or 'тип'
    return f'{base}_{slug}'



def _resolve_top_n_per_type(top_n_per_type=None):
    """int или dict[тип→N]; None → config.fraud_top_n_per_type."""
    if top_n_per_type is None:
        top_n_per_type = _cfg('fraud_top_n_per_type', 15)
    return top_n_per_type


def select_fraud_top(ranked: pd.DataFrame, top_n_per_type=None) -> pd.DataFrame:
    """
    Для каждого типа: Top-N по score↓.
    Порядок блоков — по _TYPE_PRIORITY.
    """
    if ranked is None or ranked.empty:
        return ranked if ranked is not None else pd.DataFrame()
    ncfg = _resolve_top_n_per_type(top_n_per_type)
    types_ordered = sorted(
        ranked['тип'].dropna().unique(),
        key=lambda t: _TYPE_PRIORITY.get(t, 5),
    )
    parts = []
    for t in types_ordered:
        sub = ranked[ranked['тип'] == t].sort_values('score', ascending=False)
        n = int(ncfg[t]) if isinstance(ncfg, dict) else int(ncfg)
        parts.append(sub.head(max(0, n)))
    if not parts:
        return ranked.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def evaluate_fraud_clusters(bundle: dict) -> pd.DataFrame:
    if bundle.get('ranked') is not None:
        df = bundle['ranked']
        if not df.empty and 'size' in df.columns:
            df = df[df['size'] >= bound].reset_index(drop=True)
            bundle['ranked'] = df
        return df

    graph = bundle['graph']
    people_count = _people_count()
    rows = []
    for cl in bundle['clusters']:
        nodes = cl['nodes']
        if len(nodes) < bound:
            continue
        feats = score_cluster(graph, nodes, people_count, bundle=bundle)
        ctype = classify_case_type(feats)
        if ctype == 'фоновая мелочь':
            continue
        rows.append({
            'cluster_id': cl['cluster_id'],
            'kind': cl['kind'],
            'тип': ctype,
            'score': feats['score'],
            'size': feats['size'],
            'star_share': feats['star_share'],
            'ego_density': feats['ego_density'],
            'directed_cycles': feats['directed_cycles'],
            'reciprocity': feats['reciprocity'],
            'repeat_pairs': feats['repeat_pairs'],
            'bitki_jaccard': feats['bitki_jaccard_max'],
            'vin_star': feats['vin_star_flag'],
            'data_quality': feats['data_quality'],
            'hub': feats['hub'],
            'nodes': list(nodes),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        bundle['ranked'] = df
        return df
    df['_prio'] = df['тип'].map(lambda t: _TYPE_PRIORITY.get(t, 5))
    df = df.sort_values(['_prio', 'score'], ascending=[True, False]).drop(columns='_prio')
    df = df.reset_index(drop=True)
    bundle['ranked'] = df
    return df


def quick_fraud_slices(min_pair_losses: int = 2) -> dict[str, pd.DataFrame]:
    """Быстрые срезы по таблице убытков (отдельно от кластеризации)."""
    ensure_runtime_state()
    from . import main as m
    data = m.data
    out = {}

    if 'Culprit' in data.columns and 'Victim' in data.columns:
        pairs = data.groupby(['Culprit', 'Victim']).size().reset_index(name='убытков')
        pairs = pairs[pairs['убытков'] >= min_pair_losses].sort_values('убытков', ascending=False)
        out['повторные_пары'] = pairs.head(50)

        s = set(zip(data['Culprit'], data['Victim']))
        mutual = [(a, b) for a, b in s if a != b and (b, a) in s]
        out['взаимные_пары'] = pd.DataFrame(mutual, columns=['a', 'b']).head(50)

    if 'VINc' in data.columns:
        vin_stats = data.groupby('VINc').agg(
            убытков=('VINc', 'size'),
            водителей=('Culprit', 'nunique') if 'Culprit' in data.columns else ('VINc', 'size'),
        ).reset_index()
        vin_stats = vin_stats[
            (vin_stats['убытков'] >= 5) & (vin_stats['водителей'] >= 5)
        ].sort_values('убытков', ascending=False)
        out['vin_звёзды'] = vin_stats.head(50)

    return out


def show_quick_slices():
    """Показать быстрые срезы по data (опционально, до/после кандидатов)."""
    from IPython.display import display, Markdown

    display(Markdown('### Быстрые срезы по убыткам'))
    slices = quick_fraud_slices()
    for name, df in slices.items():
        display(Markdown(f'**{name}** ({len(df)} строк)'))
        display(df.head(15))
    return slices


def show_fraud_candidates(group_index: int = 0, top_n_per_type=None,
                          *, with_slices: bool = False):
    """Отчёт: Top-N по score внутри каждого типа. Срезы — with_slices=True."""
    from IPython.display import display, Markdown

    if with_slices:
        show_quick_slices()

    display(Markdown('### Кластеры-кандидаты (fraud): Top-N по score внутри типа'))
    bundle = build_fraud_clusters(group_index)
    ranked = evaluate_fraud_clusters(bundle)
    view = select_fraud_top(ranked, top_n_per_type)
    ncfg = _resolve_top_n_per_type(top_n_per_type)
    display(Markdown(f'`fraud_top_n_per_type` = `{ncfg}` · всего в витрине: **{len(view)}**'))
    show_cols = [
        'cluster_id', 'тип', 'score', 'size', 'star_share', 'directed_cycles',
        'reciprocity', 'repeat_pairs', 'bitki_jaccard', 'vin_star', 'data_quality',
    ]
    display(view[show_cols] if len(view) else view)
    bundle['view'] = view
    return view, bundle


# ---------------------------------------------------------------------------
# 7. Визуализация
# ---------------------------------------------------------------------------

def _loss_numbers_for_ids(loss_ids, limit: int = 3) -> str:
    """До ``limit`` номеров убытков через запятую."""
    if loss_ids is None:
        return ''
    from . import main as m
    id_col = getattr(m, 'ID_col', None)
    data = getattr(m, 'data', None)
    if data is None or id_col is None:
        return ''
    try:
        ids = list(pd.unique(pd.Series(loss_ids).dropna().astype(int)))[:limit]
        vals = list(pd.unique(data.loc[ids, id_col]))[:limit]
    except Exception:
        return ''
    return ', '.join(str(v) for v in vals if pd.notna(v))


def _edge_loss_map(nodes: set, bundle: dict | None, limit: int = 3) -> dict:
    """(u,v) с u≤v → {'title': номера убытков, 'weight': n_unique Loss_idx}."""
    sub = _cluster_sublinks(nodes, bundle)
    out = {}
    if sub.empty or 'Loss_idx' not in sub.columns:
        return out
    a = np.minimum(sub['obj1'].to_numpy(), sub['obj2'].to_numpy())
    b = np.maximum(sub['obj1'].to_numpy(), sub['obj2'].to_numpy())
    tmp = sub.assign(_a=a, _b=b)
    for (u, v), grp in tmp.groupby(['_a', '_b']):
        lids = grp['Loss_idx'].dropna().tolist()
        out[(int(u), int(v))] = {
            'title': _loss_numbers_for_ids(lids, limit=limit),
            'weight': max(1, int(grp['Loss_idx'].nunique())),
        }
    return out


def _annotate_fraud_subgraph(sub: nx.Graph, nodes: set, bundle: dict | None) -> nx.Graph:
    """Направленные VC-рёбра + title/weight по убыткам + label узлов."""
    H = sub.copy()
    obj_meta = _object_meta(H.nodes)
    for node in H.nodes:
        info = obj_meta.get(int(node), {})
        lab = info.get('label') or str(node)
        H.nodes[node]['label'] = lab if len(lab) <= 40 else lab[:38] + '…'
        H.nodes[node]['title'] = lab
        H.nodes[node]['ntype'] = info.get('ntype', 'человек')

    for a, b in _directed_vc_pairs(nodes, bundle):
        if H.has_edge(a, b):
            H[a][b]['directed'] = True
            H[a][b]['color'] = '#c0392b'
        elif H.has_edge(b, a):
            H[b][a]['directed'] = True
            H[b][a]['color'] = '#c0392b'

    loss_map = _edge_loss_map(nodes, bundle, limit=3)
    for u, v in list(H.edges):
        key = (min(int(u), int(v)), max(int(u), int(v)))
        info = loss_map.get(key, {})
        title = info.get('title', '')
        weight = float(info.get('weight', 1) or 1)
        H[u][v]['title'] = title
        H[u][v]['label'] = title
        H[u][v]['weight'] = max(1.0, weight)
        if 'color' not in H[u][v]:
            H[u][v]['color'] = '#888'
        if 'width' not in H[u][v]:
            H[u][v]['width'] = min(1 + int(weight), 6)
    return H


def _sanitize_gexf(graph: nx.Graph) -> nx.Graph:
    """Строковые атрибуты; weight всегда ≥ 1 (иначе Gephi дропает рёбра)."""
    H = graph.copy()
    for node in H.nodes:
        for key, value in list(H.nodes[node].items()):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                H.nodes[node][key] = ''
            elif not isinstance(value, (str, int, float, bool)):
                H.nodes[node][key] = str(value)
    for u, v in H.edges:
        attrs = H[u][v]
        for key, value in list(attrs.items()):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                attrs[key] = ''
            elif key == 'weight':
                try:
                    attrs[key] = max(1.0, float(value))
                except Exception:
                    attrs[key] = 1.0
            elif not isinstance(value, (str, int, float, bool)):
                attrs[key] = str(value)
        if 'weight' not in attrs:
            attrs['weight'] = 1.0
        # Gephi: Label из label/title
        if not attrs.get('label') and attrs.get('title'):
            attrs['label'] = attrs['title']
    return H


def _save_fraud_gexf(graph: nx.Graph, gexf_path: str) -> None:
    parent = os.path.dirname(gexf_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    nx.write_gexf(_sanitize_gexf(graph), gexf_path)


def _save_fraud_html(graph: nx.Graph, html_path: str, title: str = ''):
    """Эго-HTML: круги/квадраты/ромбы, колпак соседей; title рёбер = убытки."""
    if graph.number_of_nodes() == 0:
        return
    parent = os.path.dirname(html_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    obj_meta = _object_meta(graph.nodes)
    cap = int(_cfg('ego_cap_neighbors', 150))

    deg = dict(graph.degree())
    hub = max(deg, key=deg.get) if deg else None
    nodes_keep = set(graph.nodes)
    tail_count = 0
    if hub is not None and graph.number_of_nodes() > cap + 1:
        nbrs = sorted(graph.neighbors(hub), key=lambda n: deg.get(n, 0), reverse=True)
        keep = {hub} | set(nbrs[:cap])
        tail_count = graph.number_of_nodes() - len(keep)
        nodes_keep = keep

    sub = graph.subgraph(nodes_keep).copy()
    with open(html_path, 'w', encoding='utf-8') as fh:
        fh.write(
            '<!DOCTYPE html><html><head><meta charset="utf-8">'
            f'<title>{title}</title>'
            '<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js">'
            '</script>'
            '<style>html,body,#m{margin:0;height:100%;width:100%;}</style>'
            '</head><body><div id="m"></div><script>\n'
            'const nodes=new vis.DataSet([\n'
        )
        first = True
        for node in sub.nodes:
            info = obj_meta.get(int(node), {})
            lab = info.get('label') or str(node)
            ntype = info.get('ntype', 'человек')
            shape = {'человек': 'ellipse', 'VIN': 'box', 'юрлицо': 'diamond'}.get(
                ntype, 'ellipse',
            )
            short = lab if len(lab) <= 28 else lab[:26] + '…'
            item = {
                'id': str(node),
                'label': short,
                'title': lab,
                'shape': shape,
                'color': '#e74c3c' if node == hub else (
                    '#3498db' if ntype == 'VIN' else (
                        '#9b59b6' if ntype == 'юрлицо' else '#95a5a6'
                    )
                ),
                'font': {'size': 11, 'face': 'Arial', 'align': 'center', 'multi': True},
                'margin': 10,
            }
            if not first:
                fh.write(',\n')
            first = False
            json.dump(item, fh, ensure_ascii=False)
        if tail_count > 0:
            fh.write(',\n')
            json.dump({
                'id': 'tail',
                'label': f'хвост\n+{tail_count}',
                'shape': 'ellipse',
                'color': '#bdc3c7',
                'font': {'size': 12},
            }, fh, ensure_ascii=False)
        fh.write('\n]);\nconst edges=new vis.DataSet([\n')
        first = True
        for u, v, attrs in sub.edges(data=True):
            item = {
                'from': str(u),
                'to': str(v),
                'color': attrs.get('color', '#888'),
                'width': int(attrs.get('width', 1) or 1),
            }
            edge_title = attrs.get('title') or attrs.get('label') or ''
            if edge_title:
                item['title'] = edge_title
            if attrs.get('directed'):
                item['arrows'] = 'to'
            if not first:
                fh.write(',\n')
            first = False
            json.dump(item, fh, ensure_ascii=False)
        if tail_count > 0 and hub is not None:
            fh.write(',\n')
            json.dump({'from': str(hub), 'to': 'tail', 'color': '#bbb', 'dashes': True}, fh)
        fh.write(
            '\n]);\n'
            'new vis.Network(document.getElementById("m"),{nodes,edges},{'
            'physics:{enabled:true,stabilization:{iterations:150}},'
            'nodes:{font:{size:11,align:"center"}},'
            'edges:{smooth:{type:"continuous"}},'
            'interaction:{dragNodes:true,dragView:true,zoomView:true,hover:true}'
            '});\n</script></body></html>\n'
        )


def _build_overview_html(view: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(
            '<!DOCTYPE html><html><head><meta charset="utf-8"><title>Fraud overview</title>'
            '<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js">'
            '</script>'
            '<style>html,body,#m{margin:0;height:100%;width:100%;}</style>'
            '</head><body><div id="m"></div><script>\nconst nodes=new vis.DataSet([\n'
        )
        first = True
        colors = {
            'группа-колотуны': '#e74c3c',
            'кольцо битов': '#e67e22',
            'соло с подставными': '#f1c40f',
            'юрлицо-хаб': '#9b59b6',
        }
        for _, row in view.iterrows():
            size = 10 + min(int(row['size']), 200) / 5
            item = {
                'id': str(row['cluster_id']),
                'label': f"{row['тип']}\n{row['score']}",
                'title': (
                    f"{row['cluster_id']} | {row['тип']} | score={row['score']} | "
                    f"n={row['size']}"
                ),
                'shape': 'dot',
                'size': size,
                'color': colors.get(row['тип'], '#7f8c8d'),
                'font': {'size': 10, 'color': '#222'},
            }
            if not first:
                fh.write(',\n')
            first = False
            json.dump(item, fh, ensure_ascii=False)
        fh.write(
            '\n]);\nconst edges=new vis.DataSet([]);\n'
            'new vis.Network(document.getElementById("m"),{nodes,edges},{'
            'physics:{enabled:true,stabilization:{iterations:200}},'
            'interaction:{hover:true,dragView:true,zoomView:true}'
            '});\n</script></body></html>\n'
        )


def _zip_directory(src_dir: str, zip_path: str) -> str:
    """Упаковать содержимое каталога в zip (файлы в корне архива)."""
    parent = os.path.dirname(zip_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.isfile(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for name in files:
                full = os.path.join(root, name)
                arc = os.path.relpath(full, start=src_dir).replace('\\', '/')
                zf.write(full, arcname=arc)
    return zip_path


def visualize_fraud(group_index: int = 0, top_n_per_type=None):
    """Обзор + эго (HTML) + Gephi + zip: та же витрина Top-N по типу."""
    import shutil

    ensure_artifact_dirs()
    log('Fraud viz', level='header')

    html_zip = os.path.join(HTML_DIR, 'fraud.zip')
    gephi_zip = os.path.join(GEPHI_DIR, 'fraud.zip')

    if os.path.isdir(FRAUD_HTML_DIR):
        shutil.rmtree(FRAUD_HTML_DIR)
    if os.path.isdir(FRAUD_GEPHI_DIR):
        shutil.rmtree(FRAUD_GEPHI_DIR)
    for path in (html_zip, gephi_zip):
        if os.path.isfile(path):
            os.remove(path)
    os.makedirs(FRAUD_HTML_DIR, exist_ok=True)
    os.makedirs(FRAUD_GEPHI_DIR, exist_ok=True)

    bundle = build_fraud_clusters(group_index)
    ranked = evaluate_fraud_clusters(bundle)
    view = select_fraud_top(ranked, top_n_per_type)
    bundle['view'] = view
    if view.empty:
        log('нет кандидатов для отрисовки', level='warn')
        return view

    ncfg = _resolve_top_n_per_type(top_n_per_type)
    log(f'витрина: fraud_top_n_per_type={ncfg}, строк={len(view)}', level='info')
    if 'тип' in view.columns:
        counts = view['тип'].value_counts().to_dict()
        log(f'по типам: {counts}', level='info')

    overview = os.path.join(FRAUD_HTML_DIR, 'overview.html')
    _build_overview_html(view, overview)
    log(f'обзор → {overview}', level='ok')

    graph = bundle['graph']
    id_map = {cl['cluster_id']: cl['nodes'] for cl in bundle['clusters']}

    for _, row in view.iterrows():
        nodes = set(row['nodes']) if row['nodes'] is not None else set()
        if not nodes:
            nodes = set(id_map.get(row['cluster_id'], ()))
        sub = _annotate_fraud_subgraph(graph.subgraph(nodes), nodes, bundle)
        stem = _export_stem(row['cluster_id'], row['тип'])
        html_path = os.path.join(FRAUD_HTML_DIR, f'{stem}.html')
        gexf_path = os.path.join(FRAUD_GEPHI_DIR, f'{stem}.gexf')
        _save_fraud_html(sub, html_path, title=f"{row['тип']} {row['score']}")
        _save_fraud_gexf(sub, gexf_path)

    _zip_directory(FRAUD_HTML_DIR, html_zip)
    _zip_directory(FRAUD_GEPHI_DIR, gephi_zip)
    log(f'эго HTML: {len(view)} → {FRAUD_HTML_DIR}', level='ok')
    log(f'Gephi:    {len(view)} → {FRAUD_GEPHI_DIR}', level='ok')
    log(f'zip HTML:  → {html_zip}', level='ok')
    log(f'zip Gephi: → {gephi_zip}', level='ok')
    return view


def run_fraud_pipeline(group_index: int = 0, top_n_per_type=None):
    """Аудит → кандидаты → viz (удобный one-shot после run_pipeline)."""
    show_hub_audit(group_index, top_n=40)
    view, bundle = show_fraud_candidates(group_index, top_n_per_type=top_n_per_type)
    visualize_fraud(group_index, top_n_per_type=top_n_per_type)
    return view, bundle
