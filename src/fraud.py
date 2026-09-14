"""Поиск мошеннических кейсов: аудит хабов, кластеры, скоринг, viz."""

from __future__ import annotations

import json
import os
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from . import config
from .main import (
    HTML_DIR,
    bound,
    ensure_artifact_dirs,
    ensure_runtime_state,
    label_matches_keywords,
    log,
    _clean_entity_label,
    _hub_keywords,
    _node_label,
    _subgraph_for_group,
)


FRAUD_HTML_DIR = os.path.join(HTML_DIR, 'fraud')


def _cfg(name, default):
    return getattr(config, name, default)


def _people_count() -> int:
    from . import main as m
    return len(m.people)


def _objects_df() -> pd.DataFrame:
    from . import main as m
    return pd.DataFrame(m.objects)


def _links_df() -> pd.DataFrame:
    from . import main as m
    return m.links


def node_type(obj_idx: int, label: str, people_count: int, keywords) -> str:
    if label_matches_keywords(label, keywords):
        return 'юрлицо'
    if obj_idx >= people_count:
        return 'VIN'
    # юрлицо без keyword в списке people (ОПФ могли не попасть)
    upper = label.upper()
    if any(x in upper for x in ('ООО', 'АО ', 'ПАО', 'ИП ', '"', '«')):
        if label_matches_keywords(label, keywords) or '"' in label or '«' in label:
            return 'юрлицо'
    return 'человек'


def has_nan_bd(label: str) -> bool:
    parts = str(label).strip().split()
    if not parts:
        return False
    return parts[-1].lower() in ('nan', 'nat', 'none')


def classify_hub_basket(obj_idx, label, degree, ntype, keywords,
                        artifact_n, suspect_n) -> str:
    if has_nan_bd(label) and degree > artifact_n // 2:
        return 'nan_glue'
    if label_matches_keywords(label, keywords) or ntype == 'юрлицо':
        if label_matches_keywords(label, keywords):
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
    graph = _subgraph_for_group(group_index)
    objects_df = _objects_df()
    people_count = _people_count()
    keywords = _hub_keywords()
    artifact_n = int(_cfg('artifact_degree_n', 1000))
    suspect_n = int(_cfg('suspect_degree_n', 20))

    deg = dict(graph.degree())
    rows = []
    for node, d in sorted(deg.items(), key=lambda x: -x[1])[:top_n]:
        label = _node_label(objects_df, node)
        ntype = node_type(node, label, people_count, keywords)
        basket = classify_hub_basket(
            node, label, d, ntype, keywords, artifact_n, suspect_n,
        )
        rows.append({
            'obj_idx': node,
            'degree': int(d),
            'тип': ntype,
            'has_nan_bd': has_nan_bd(label),
            'keyword_hit': label_matches_keywords(label, keywords),
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
    """Итеративно удалить degree==1 (2-core)."""
    H = graph.copy()
    removed = 0
    while H.number_of_nodes():
        leaves = [n for n, d in H.degree() if d <= 1]
        if not leaves:
            break
        H.remove_nodes_from(leaves)
        removed += len(leaves)
    return H, removed


def prepare_fraud_graph(group_index: int = 0):
    """
    Граф без whitelist/NaN-склеек + срез листьев.

    Возвращает (graph, meta) где meta содержит множества и счётчики.
    """
    ensure_runtime_state()
    graph = _subgraph_for_group(group_index)
    objects_df = _objects_df()
    people_count = _people_count()
    keywords = _hub_keywords()
    artifact_n = int(_cfg('artifact_degree_n', 1000))
    suspect_n = int(_cfg('suspect_degree_n', 20))

    deg = dict(graph.degree())
    exclude = set()
    suspects = set()
    tags = defaultdict(list)

    for node in list(graph.nodes):
        label = _node_label(objects_df, node)
        d = deg.get(node, 0)
        ntype = node_type(node, label, people_count, keywords)
        basket = classify_hub_basket(
            node, label, d, ntype, keywords, artifact_n, suspect_n,
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
    # seeds могли стать листьями — оставляем только живые
    suspects = {n for n in suspects if n in H}

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
    return H, meta


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


def build_fraud_clusters(group_index: int = 0):
    """Кластеры от suspect seeds + мини-CC остатка."""
    H, meta = prepare_fraud_graph(group_index)
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

    # мосты: узлы с соседями в разных кластерах
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
    return {
        'graph': H,
        'meta': meta,
        'clusters': cluster_list,
        'bridges': pd.DataFrame(bridges),
        'group_index': group_index,
    }


# ---------------------------------------------------------------------------
# 4–5. Признаки и типы
# ---------------------------------------------------------------------------

def _cluster_sublinks(nodes: set) -> pd.DataFrame:
    links = _links_df()
    return links[links['obj1'].isin(nodes) & links['obj2'].isin(nodes)]


def _directed_vc_pairs(nodes: set) -> list[tuple[int, int]]:
    """Пары (виновник_idx, потерпевший_idx) внутри кластера по Loss_idx."""
    from . import main as m
    data = m.data
    objects_df = _objects_df()
    # map label -> idx for people in cluster
    label_to_idx = {}
    for n in nodes:
        if n < _people_count():
            label_to_idx[_node_label(objects_df, n)] = n

    sub = _cluster_sublinks(nodes)
    if sub.empty or 'Loss_idx' not in sub.columns:
        return []
    loss_ids = sub['Loss_idx'].dropna().unique()
    pairs = []
    if 'Culprit' not in data.columns or 'Victim' not in data.columns:
        return pairs
    for lid in loss_ids:
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


def _vin_bitki_features(nodes: set, people_count: int) -> dict:
    """F4: пересечения исходящих пострадавших VIN."""
    from . import main as m
    data = m.data
    objects_df = _objects_df()
    vins = {n for n in nodes if n >= people_count}
    if len(vins) < 2 or 'VINc' not in data.columns or 'VINv' not in data.columns:
        return {'bitki_jaccard_max': 0.0, 'vin_role_flip': 0.0, 'vin_star_flag': 0}

    vin_label = {n: _node_label(objects_df, n) for n in vins}
    label_to_vin = {v: k for k, v in vin_label.items() if v}

    out_victims = defaultdict(set)  # fighter_vin -> set(victim_vins)
    vin_as_fighter = defaultdict(int)
    vin_as_victim = defaultdict(int)
    vin_drivers = defaultdict(set)

    sub = _cluster_sublinks(nodes)
    loss_ids = sub['Loss_idx'].dropna().unique() if not sub.empty else []
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

    flip = 0
    for v in vins:
        if vin_as_fighter[v] > 0 and vin_as_victim[v] > 0:
            flip += 1
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


def score_cluster(graph: nx.Graph, nodes: set, people_count: int) -> dict:
    weights = dict(_cfg('fraud_score_weights', {}))
    sub = graph.subgraph(nodes)
    n = len(nodes)
    m = sub.number_of_edges()
    density = (2 * m) / (n * (n - 1)) if n > 1 else 0.0

    star_share, hub = _star_share(graph, nodes)
    ego_dens = _ego_density(graph, hub)
    pairs = _directed_vc_pairs(nodes)
    cycles = _count_directed_triangles(pairs)
    recip = _reciprocity(pairs)

    # undirected triangles fallback
    undirected_tri = sum(1 for _ in nx.triangles(sub).values()) // 3 if n < 500 else 0

    sub_links = _cluster_sublinks(nodes)
    if not sub_links.empty and 'Loss_idx' in sub_links.columns:
        a = np.minimum(sub_links['obj1'], sub_links['obj2'])
        b = np.maximum(sub_links['obj1'], sub_links['obj2'])
        w = sub_links.assign(_a=a, _b=b).groupby(['_a', '_b'])['Loss_idx'].nunique()
        repeat_share = float((w >= 2).mean()) if len(w) else 0.0
    else:
        repeat_share = 0.0

    bitki = _vin_bitki_features(nodes, people_count)

    # role concentration: доля пар где один всегда culprit
    role_conc = 0.0
    if pairs:
        from collections import Counter
        out_c = Counter(a for a, _ in pairs)
        role_conc = out_c.most_common(1)[0][1] / len(pairs)

    keywords = _hub_keywords()
    objects_df = _objects_df()
    wl_share = sum(
        1 for x in nodes
        if label_matches_keywords(_node_label(objects_df, x), keywords)
    ) / max(n, 1)

    nan_share = sum(
        1 for x in nodes if has_nan_bd(_node_label(objects_df, x))
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
    # undirected bonus if no directed cycles but triangles exist
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
        return 'кольцо битков'
    if feats['directed_cycles'] >= 1 and feats['star_share'] < 0.8:
        return 'группа-колотуны'
    if feats['star_share'] >= 0.8 and feats['ego_density'] < 0.05:
        objects_df = _objects_df()
        hub = feats.get('hub')
        if hub is not None:
            lab = _node_label(objects_df, hub)
            keywords = _hub_keywords()
            if label_matches_keywords(lab, keywords) or hub >= _people_count():
                # VIN star already handled; legal hub
                if hub < _people_count() and label_matches_keywords(lab, keywords):
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


def evaluate_fraud_clusters(bundle: dict) -> pd.DataFrame:
    graph = bundle['graph']
    people_count = _people_count()
    rows = []
    for cl in bundle['clusters']:
        nodes = cl['nodes']
        if len(nodes) < 2:
            continue
        feats = score_cluster(graph, nodes, people_count)
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
        return df
    df['_prio'] = df['тип'].map(lambda t: _TYPE_PRIORITY.get(t, 5))
    df = df.sort_values(['_prio', 'score'], ascending=[True, False]).drop(columns='_prio')
    return df.reset_index(drop=True)


def quick_fraud_slices(min_pair_losses: int = 2) -> dict[str, pd.DataFrame]:
    """Быстрые срезы по таблице убытков (до кластеризации)."""
    ensure_runtime_state()
    from . import main as m
    data = m.data
    out = {}

    if 'Culprit' in data.columns and 'Victim' in data.columns:
        pairs = data.groupby(['Culprit', 'Victim']).size().reset_index(name='убытков')
        pairs = pairs[pairs['убытков'] >= min_pair_losses].sort_values('убытков', ascending=False)
        out['повторные_пары'] = pairs.head(50)

        # взаимность
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


def show_fraud_candidates(group_index: int = 0, top_n: int = 50):
    """Отчёт: быстрые срезы + топ кластеров по типу/скору."""
    from IPython.display import display, Markdown

    display(Markdown('### Быстрые срезы по убыткам'))
    slices = quick_fraud_slices()
    for name, df in slices.items():
        display(Markdown(f'**{name}** ({len(df)} строк)'))
        display(df.head(15))

    display(Markdown('### Кластеры-кандидаты (fraud)'))
    bundle = build_fraud_clusters(group_index)
    ranked = evaluate_fraud_clusters(bundle)
    show_cols = [
        'cluster_id', 'тип', 'score', 'size', 'star_share', 'directed_cycles',
        'reciprocity', 'repeat_pairs', 'bitki_jaccard', 'vin_star', 'data_quality',
    ]
    display(ranked[show_cols].head(top_n) if len(ranked) else ranked)
    # keep nodes column in returned frame for viz
    return ranked, bundle


# ---------------------------------------------------------------------------
# 7. Визуализация
# ---------------------------------------------------------------------------

def _save_fraud_html(graph: nx.Graph, html_path: str, title: str = ''):
    """Эго-HTML: круги/квадраты/ромбы, колпак соседей."""
    if graph.number_of_nodes() == 0:
        return
    parent = os.path.dirname(html_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    people_count = _people_count()
    objects_df = _objects_df()
    keywords = _hub_keywords()
    cap = int(_cfg('ego_cap_neighbors', 150))

    deg = dict(graph.degree())
    hub = max(deg, key=deg.get) if deg else None
    nodes_keep = set(graph.nodes)
    tail_count = 0
    if hub is not None and graph.number_of_nodes() > cap + 1:
        nbrs = sorted(graph.neighbors(hub), key=lambda n: deg.get(n, 0), reverse=True)
        keep = {hub} | set(nbrs[:cap])
        # add edges among keep
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
        for i, node in enumerate(sub.nodes):
            lab = _node_label(objects_df, node)
            ntype = node_type(node, lab, people_count, keywords)
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


def _build_overview_html(ranked: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    top = ranked.head(int(_cfg('fraud_viz_top_n', 50)))
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
        for _, row in top.iterrows():
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


def visualize_fraud(group_index: int = 0, top_n: int | None = None):
    """Обзор + эго топ-кластеров в ./output/html/fraud/."""
    ensure_artifact_dirs()
    top_n = top_n or int(_cfg('fraud_viz_top_n', 50))
    log('Fraud viz', level='header')

    # clear fraud html only
    if os.path.isdir(FRAUD_HTML_DIR):
        import shutil
        shutil.rmtree(FRAUD_HTML_DIR)
    os.makedirs(FRAUD_HTML_DIR, exist_ok=True)

    bundle = build_fraud_clusters(group_index)
    ranked = evaluate_fraud_clusters(bundle)
    if ranked.empty:
        log('нет кандидатов для отрисовки', level='warn')
        return ranked

    overview = os.path.join(FRAUD_HTML_DIR, 'overview.html')
    _build_overview_html(ranked, overview)
    log(f'обзор → {overview}', level='ok')

    graph = bundle['graph']
    id_map = {cl['cluster_id']: cl['nodes'] for cl in bundle['clusters']}

    for _, row in ranked.head(top_n).iterrows():
        nodes = set(row['nodes']) if row['nodes'] is not None else set()
        if not nodes:
            nodes = set(id_map.get(row['cluster_id'], ()))
        sub = graph.subgraph(nodes).copy()
        # mark directed VC edges
        for a, b in _directed_vc_pairs(nodes):
            if sub.has_edge(a, b):
                sub[a][b]['directed'] = True
                sub[a][b]['color'] = '#c0392b'
            elif sub.has_edge(b, a):
                sub[b][a]['directed'] = True
                sub[b][a]['color'] = '#c0392b'
        safe_id = str(row['cluster_id']).replace('/', '_')
        path = os.path.join(FRAUD_HTML_DIR, f'{safe_id}.html')
        _save_fraud_html(sub, path, title=f"{row['тип']} {row['score']}")
    log(f'эго записано: {min(top_n, len(ranked))} → {FRAUD_HTML_DIR}', level='ok')
    return ranked


def run_fraud_pipeline(group_index: int = 0, top_n: int = 50):
    """Аудит → кандидаты → viz (удобный one-shot после run_pipeline)."""
    show_hub_audit(group_index, top_n=40)
    ranked, bundle = show_fraud_candidates(group_index, top_n=top_n)
    visualize_fraud(group_index, top_n=top_n)
    return ranked, bundle
