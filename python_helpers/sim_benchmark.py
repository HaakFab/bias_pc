from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
import random
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------------------------------------------------------
# Progress bar helper (tqdm with graceful fallback)
# -----------------------------------------------------------------------------
def _get_tqdm():
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm
    except Exception:
        class _TQDM:
            def __init__(self, iterable=None, total=None, desc="", leave=True):
                self.iterable = iterable if iterable is not None else range(total or 0)
                self.desc = desc
            def __iter__(self):
                for x in self.iterable:
                    yield x
            def update(self, n=1): pass
            def close(self): pass
        return _TQDM

tqdm = _get_tqdm()

# -----------------------------------------------------------------------------
# ELO helpers
# -----------------------------------------------------------------------------
def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

def update_elo(elo_a: float, elo_b: float, actual_score_a: float, k_factor: int = 32) -> Tuple[float, float]:
    expected_a = expected_score(elo_a, elo_b)
    elo_a_new = elo_a + k_factor * (actual_score_a - expected_a)
    elo_b_new = elo_b + k_factor * ((1.0 - actual_score_a) - (1.0 - expected_a))
    return elo_a_new, elo_b_new

def _log_match(
    round_list: List[Dict[str, Any]],
    text1: str,
    text2: Optional[str],
    winner: Optional[str],
    ea: Optional[float],
    eb: Optional[float],
    na: Optional[float],
    nb: Optional[float],
):
    if text2 is None:
        round_list.append({"text1": text1, "text2": None, "winner": None, "elo_change": (0.0, 0.0)})
    else:
        da = 0.0 if ea is None or na is None else (ea - na)
        db = 0.0 if eb is None or nb is None else (eb - nb)
        round_list.append({"text1": text1, "text2": text2, "winner": winner, "elo_change": (da, db)})

def _save_state(
    save_path: Optional[str],
    elo_scores: Dict[str, float],
    match_history: List[List[Dict[str, Any]]],
    meta: Optional[Dict[str, Any]] = None
) -> None:
    if save_path is None:
        return
    payload: Dict[str, Any] = {"elo_scores": elo_scores, "match_history": match_history}
    if meta is not None:
        payload["meta"] = meta
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)

# Pairing utility: random or Elo-similar (with randomness window)
def _build_pairs(
    items: List[str],
    elo_scores: Dict[str, float],
    strategy: str = "random",
    randomness_factor: float = 0.1,
    rng: Optional[random.Random] = None,
) -> List[Tuple[str, Optional[str]]]:
    rng = rng or random.Random()
    if strategy == "random" or len(items) < 2:
        pool = items[:]
        rng.shuffle(pool)
        pairs = list(zip(pool[::2], pool[1::2]))
        if len(pool) % 2 == 1:
            pairs.append((pool[-1], None))
        return pairs

    # "similar": sort by Elo, pair within a small window
    sorted_items = sorted(items, key=lambda x: elo_scores[x])
    used = set()
    pairs: List[Tuple[str, Optional[str]]] = []
    idxs = list(range(len(sorted_items)))
    rng.shuffle(idxs)
    for i in idxs:
        if i in used:
            continue
        window = max(1, int(len(sorted_items) * randomness_factor))
        cands = [j for j in range(max(0, i - window), min(len(sorted_items), i + window + 1))
                 if j != i and j not in used]
        if cands:
            j = rng.choice(cands)
            pairs.append((sorted_items[i], sorted_items[j]))
            used.update({i, j})
        else:
            remaining = [k for k in range(len(sorted_items)) if k != i and k not in used]
            if remaining:
                j = rng.choice(remaining)
                pairs.append((sorted_items[i], sorted_items[j]))
                used.update({i, j})
            else:
                pairs.append((sorted_items[i], None))
                used.add(i)
    return pairs

# =============================================================================
# 1) Streak-based early-stop classification/ranking (PARALLEL)
# =============================================================================
def streak_early_stop_elo(
    queries: List[str],
    compare_fn: Callable[[str, str], float],
    save_path: Optional[str] = None,
    initial_elo: float = 1500.0,
    k_factor: int = 32,
    r: int = 3,
    min_distinct_opponents: int = 3,
    max_epochs: int = 50,
    opponent_selection: str = "random",   # "random" or "similar"
    randomness_factor: float = 0.1,
    seed: Optional[int] = 42,
) -> Tuple[Dict[str, float], List[List[Dict[str, Any]]], Dict[str, str]]:
    """
    Each item plays at most one match per epoch; pairs are formed by strategy and
    evaluated in PARALLEL against a frozen Elo snapshot. max_workers = max(50, #pairs).
    """
    rng = random.Random(seed)
    elo_scores: Dict[str, float] = {q: initial_elo for q in queries}
    decided: Dict[str, str] = {}
    streaks: Dict[str, Tuple[int, int, set]] = {q: (0, 0, set()) for q in queries}
    match_history: List[List[Dict[str, Any]]] = []

    for _epoch in tqdm(range(max_epochs), desc="Streak epochs"):
        active = [q for q in queries if q not in decided]
        if len(active) <= 1:
            break

        pairs = _build_pairs(active, elo_scores, strategy=opponent_selection, randomness_factor=randomness_factor, rng=rng)
        snapshot = elo_scores.copy()

        # Parallel compare
        tasks = [(a, b) for (a, b) in pairs if b is not None]
        workers = max(50, len(tasks))
        results: Dict[Tuple[str, str], float] = {}
        if tasks:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_to_pair = {pool.submit(compare_fn, a, b): (a, b) for (a, b) in tasks}
                for fut in as_completed(future_to_pair):
                    a, b = future_to_pair[fut]
                    try:
                        results[(a, b)] = fut.result()
                    except Exception:
                        results[(a, b)] = 0.5  # tie on failure

        round_log: List[Dict[str, Any]] = []
        for a, b in pairs:
            if b is None:
                _log_match(round_log, a, None, None, None, None, None, None)
                continue
            res = results.get((a, b), 0.5)
            ea, eb = snapshot[a], snapshot[b]
            if res == 1:
                na, nb = update_elo(ea, eb, 1.0, k_factor=k_factor); winner = a
                w, l, seen = streaks[a]; streaks[a] = (w+1, 0, seen | {b})
                ow, ol, oseen = streaks[b]; streaks[b] = (0, ol+1, oseen | {a})
            elif res == 0:
                nb, na = update_elo(eb, ea, 1.0, k_factor=k_factor); winner = b
                w, l, seen = streaks[a]; streaks[a] = (0, l+1, seen | {b})
                ow, ol, oseen = streaks[b]; streaks[b] = (ow+1, 0, oseen | {a})
            else:
                na, nb = update_elo(ea, eb, 0.5, k_factor=k_factor); winner = "tie"
                w, l, seen = streaks[a]; streaks[a] = (0, 0, seen)
                ow, ol, oseen = streaks[b]; streaks[b] = (0, 0, oseen)
            elo_scores[a], elo_scores[b] = na, nb
            _log_match(round_log, a, b, winner, ea, eb, na, nb)

        # Decide labels (example heuristic)
        for q in [x for x in active if x not in decided]:
            w, l, seen = streaks[q]
            if w >= r and len(seen) >= min_distinct_opponents:
                decided[q] = "biased"
            elif l >= r and len(seen) >= min_distinct_opponents:
                decided[q] = "non-biased"

        match_history.append(round_log)
        _save_state(save_path, elo_scores, match_history, meta={"mode": "streak_early_stop_elo"})

        if len(decided) == len(queries):
            break

    return elo_scores, match_history, decided

# =============================================================================
# 2) Standard Elo with tail pruning after round Y (PARALLEL per round)
# =============================================================================
def pairwise_comparison_elo_prune(
    queries: List[str],
    compare_fn: Callable[[str, str], float],
    n_rounds: int,
    save_path: Optional[str] = None,
    initial_elo: float = 1500.0,
    k_factor: int = 32,
    match_strategy: str = "random",
    randomness_factor: float = 0.1,
    prune_start_round: int = 12,
    prune_percent: float = 0.1,
    stability_margin: float = 0.0,
    seed: Optional[int] = 42,
) -> Tuple[Dict[str, float], List[List[Dict[str, Any]]], Dict[int, List[str]]]:
    rng = random.Random(seed)
    elo_scores: Dict[str, float] = {q: initial_elo for q in queries}
    active: List[str] = list(queries)
    match_history: List[List[Dict[str, Any]]] = []
    pruned_by_round: Dict[int, List[str]] = {}

    for rd in tqdm(range(1, n_rounds + 1), desc="Pruned Elo rounds"):
        if len(active) <= 1:
            break

        pairs = _build_pairs(active, elo_scores, strategy=match_strategy, randomness_factor=randomness_factor, rng=rng)
        snapshot = elo_scores.copy()

        tasks = [(a, b) for (a, b) in pairs if b is not None]
        workers = max(50, len(tasks))
        results: Dict[Tuple[str, str], float] = {}
        if tasks:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_to_pair = {pool.submit(compare_fn, a, b): (a, b) for (a, b) in tasks}
                for fut in as_completed(future_to_pair):
                    a, b = future_to_pair[fut]
                    try:
                        results[(a, b)] = fut.result()
                    except Exception:
                        results[(a, b)] = 0.5

        round_log: List[Dict[str, Any]] = []
        for a, b in pairs:
            if b is None:
                _log_match(round_log, a, None, None, None, None, None, None)
                continue
            res = results.get((a, b), 0.5)
            ea, eb = snapshot[a], snapshot[b]
            if res == 1:
                na, nb = update_elo(ea, eb, 1.0, k_factor=k_factor); winner = a
            elif res == 0:
                nb, na = update_elo(eb, ea, 1.0, k_factor=k_factor); winner = b
            else:
                na, nb = update_elo(ea, eb, 0.5, k_factor=k_factor); winner = "tie"
            elo_scores[a], elo_scores[b] = na, nb
            _log_match(round_log, a, b, winner, ea, eb, na, nb)

        match_history.append(round_log)

        # Prune after round
        if rd >= prune_start_round and len(active) > 2 and prune_percent > 0.0:
            m = max(1, int(len(active) * prune_percent))
            m_low = m // 2; m_high = m - m_low
            ranked = sorted(active, key=lambda x: elo_scores[x])
            low_tail = ranked[:m_low]
            high_tail = ranked[-m_high:]
            if stability_margin > 0.0:
                center = sum(elo_scores[q] for q in active) / len(active)
                low_tail = [q for q in low_tail if (center - elo_scores[q]) >= stability_margin]
                high_tail = [q for q in high_tail if (elo_scores[q] - center) >= stability_margin]
            pruned = list(dict.fromkeys(low_tail + high_tail))
            if pruned:
                active = [q for q in active if q not in pruned]
                pruned_by_round[rd] = pruned

        _save_state(save_path, elo_scores, match_history, meta={
            "mode": "pairwise_comparison_elo_prune",
            "params": {"prune_start_round": prune_start_round, "prune_percent": prune_percent, "stability_margin": stability_margin}
        })

    return elo_scores, match_history, pruned_by_round

# =============================================================================
# 3) Listwise sorting with multi-round grouping + stitching (PARALLEL listwise)
# =============================================================================
def listwise_sort_and_merge_rounds(
    queries: List[str],
    listwise_ranker: Callable[[List[str]], List[int]],
    save_path: Optional[str] = None,
    initial_elo: float = 1500.0,
    k_factor: int = 32,
    n_rounds: int = 2,
    k: int = 10,
    overlap: int = 3,
    boundary_passes: int = 1,
    grouping_strategy: str = "similar",   # "random" or "similar"
    randomness_factor: float = 0.1,
    listwise_call_budget: Optional[int] = None,
    seed: Optional[int] = 42,
) -> Tuple[Dict[str, float], List[List[Dict[str, Any]]], Dict[str, Any]]:
    rng = random.Random(seed)
    elo_scores: Dict[str, float] = {q: initial_elo for q in queries}
    match_history: List[List[Dict[str, Any]]] = []
    meta: Dict[str, Any] = {"mode": "listwise_sort_and_merge_rounds",
                            "params": {"n_rounds": n_rounds, "k": k, "overlap": overlap, "boundary_passes": boundary_passes,
                                       "grouping_strategy": grouping_strategy, "randomness_factor": randomness_factor},
                            "counters": {"listwise_calls": 0, "inferred_pairs": 0}}

    def _can_spend_listwise() -> bool:
        return listwise_call_budget is None or meta["counters"]["listwise_calls"] < listwise_call_budget

    def _spend_listwise(nc: int = 1):
        meta["counters"]["listwise_calls"] += nc

    def _form_groups(curr_queries: List[str]) -> List[List[str]]:
        qs = curr_queries[:]
        groups: List[List[str]] = []
        if grouping_strategy == "random":
            rng.shuffle(qs)
        else:
            qs.sort(key=lambda x: elo_scores[x])
            if randomness_factor > 0.0:
                window = max(1, int(len(qs) * randomness_factor))
                swaps = max(1, int(len(qs) * randomness_factor))
                for _ in range(swaps):
                    i = rng.randrange(len(qs))
                    lo = max(0, i - window)
                    hi = min(len(qs) - 1, i + window)
                    j = rng.randrange(lo, hi + 1)
                    qs[i], qs[j] = qs[j], qs[i]
        for s in range(0, len(qs), k):
            groups.append(qs[s:s+k])
        return groups

    for _rd in tqdm(range(n_rounds), desc="Listwise rounds"):
        groups = _form_groups(queries)

        # ---- Parallel listwise calls for groups ----
        round_log: List[Dict[str, Any]] = []
        tasks = [(idx, grp) for idx, grp in enumerate(groups) if grp and _can_spend_listwise()]
        if listwise_call_budget is not None:
            allowable = listwise_call_budget - meta["counters"]["listwise_calls"]
            tasks = tasks[:max(0, allowable)]
        workers = max(50, len(tasks))
        group_orders: Dict[int, List[int]] = {}
        if tasks:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                fut_to_idx = {pool.submit(listwise_ranker, grp): idx for idx, grp in tasks}
                for fut in as_completed(fut_to_idx):
                    idx = fut_to_idx[fut]
                    try:
                        order = fut.result()
                    except Exception:
                        order = list(range(len(groups[idx])))
                    group_orders[idx] = order
            _spend_listwise(len(tasks))

        # Apply inferred wins inside each group
        for idx, grp in enumerate(groups):
            if idx not in group_orders:
                continue
            order = group_orders[idx]
            # sanitize permutation
            seen = set()
            clean = []
            for ii in order:
                if 0 <= ii < len(grp) and ii not in seen:
                    clean.append(ii); seen.add(ii)
            for ii in range(len(grp)):
                if ii not in seen:
                    clean.append(ii)
            ranked_chunk = [grp[i] for i in clean]
            for i in range(len(ranked_chunk)):
                for j in range(i+1, len(ranked_chunk)):
                    a, b = ranked_chunk[i], ranked_chunk[j]
                    ea, eb = elo_scores[a], elo_scores[b]
                    na, nb = update_elo(ea, eb, 1.0, k_factor=k_factor)
                    elo_scores[a], elo_scores[b] = na, nb
                    _log_match(round_log, a, b, a, ea, eb, na, nb)
                    meta["counters"]["inferred_pairs"] += 1

        if round_log:
            match_history.append(round_log)
            _save_state(save_path, elo_scores, match_history, meta)

        # ---- Stitching passes (parallel windows) ----
        for _pass_idx in range(boundary_passes):
            if len(groups) <= 1 or not _can_spend_listwise():
                break
            windows = []
            for g in range(len(groups) - 1):
                left = groups[g]; right = groups[g+1]
                Lw = left[max(0, len(left)-overlap):]
                Rw = right[:max(0, overlap)]
                window = Lw + Rw
                if len(window) >= 2:
                    windows.append((g, window, Lw, Rw))
            if listwise_call_budget is not None:
                allowable = listwise_call_budget - meta["counters"]["listwise_calls"]
                windows = windows[:max(0, allowable)]
            workers = max(50, len(windows))
            window_orders: Dict[int, List[int]] = {}
            if windows:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    fut_to_key = {pool.submit(listwise_ranker, w): g for (g, w, _, _) in windows}
                    for fut in as_completed(fut_to_key):
                        g = fut_to_key[fut]
                        try:
                            order = fut.result()
                        except Exception:
                            # identity
                            order = list(range(len(next(w for (gg, w, *_rest) in windows if gg == g))))
                        window_orders[g] = order
                _spend_listwise(len(windows))

            round_log = []
            for (g, window, Lw, Rw) in windows:
                if g not in window_orders:
                    continue
                order = window_orders[g]
                seen = set(); clean = []
                for ii in order:
                    if 0 <= ii < len(window) and ii not in seen:
                        clean.append(ii); seen.add(ii)
                for ii in range(len(window)):
                    if ii not in seen:
                        clean.append(ii)
                ranked_window = [window[i] for i in clean]
                for i in range(len(ranked_window)):
                    for j in range(i+1, len(ranked_window)):
                        a, b = ranked_window[i], ranked_window[j]
                        ea, eb = elo_scores[a], elo_scores[b]
                        na, nb = update_elo(ea, eb, 1.0, k_factor=k_factor)
                        elo_scores[a], elo_scores[b] = na, nb
                        _log_match(round_log, a, b, a, ea, eb, na, nb)
                        meta["counters"]["inferred_pairs"] += 1

                # reorder groups around boundary g
                left = groups[g]; right = groups[g+1]
                left_core = left[:-len(Lw)] if len(Lw) <= len(left) else []
                right_core = right[len(Rw):] if len(Rw) <= len(right) else []
                new_left_tail = [x for x in ranked_window if x in left]
                new_right_head = [x for x in ranked_window if x in right and x not in new_left_tail]
                groups[g] = left_core + new_left_tail
                groups[g+1] = new_right_head + right_core

            if round_log:
                match_history.append(round_log)
                _save_state(save_path, elo_scores, match_history, meta)

    return elo_scores, match_history, meta

# =============================================================================
# 4) Budgeted hybrid ranker (PARALLEL phases)
# =============================================================================
def budgeted_hybrid_ranker(
    queries: List[str],
    compare_fn: Callable[[str, str], float],
    listwise_ranker: Callable[[List[str]], List[int]],
    save_path: Optional[str] = None,
    # Elo settings
    initial_elo: float = 1500.0,
    k_factor: int = 32,
    # Phase A (listwise) settings
    listwise_n_rounds: int = 1,
    k: int = 10,
    overlap: int = 3,
    boundary_passes: int = 1,
    grouping_strategy: str = "similar",   # "random" or "similar"
    randomness_factor: float = 0.1,
    listwise_groups_limit: Optional[int] = None,
    listwise_call_budget: Optional[int] = None,
    # Phase B (streak) settings
    streak_r: int = 3,
    streak_min_distinct: int = 3,
    streak_epochs: int = 8,
    streak_opponent_selection: str = "similar",
    # Phase C (pruned Elo) settings
    prune_rounds: int = 8,
    prune_percent: float = 0.15,
    stability_margin: float = 0.0,
    prune_match_strategy: str = "similar",
    prune_randomness_factor: float = 0.1,
    # Global budget
    pairwise_call_budget: Optional[int] = None,
    seed: Optional[int] = 42,
) -> Tuple[Dict[str, float], List[List[Dict[str, Any]]], Dict[str, Any]]:
    rng = random.Random(seed)
    elo_scores: Dict[str, float] = {q: initial_elo for q in queries}
    match_history: List[List[Dict[str, Any]]] = []
    meta: Dict[str, Any] = {"mode": "budgeted_hybrid_ranker",
                            "counters": {"listwise_calls": 0, "pairwise_calls": 0, "inferred_pairs": 0}}

    def _can_list() -> bool:
        return listwise_call_budget is None or meta["counters"]["listwise_calls"] < listwise_call_budget
    def _spend_list(nc: int = 1):
        meta["counters"]["listwise_calls"] += nc
    def _can_pair() -> bool:
        return pairwise_call_budget is None or meta["counters"]["pairwise_calls"] < pairwise_call_budget
    def _spend_pair(nc: int = 1):
        meta["counters"]["pairwise_calls"] += nc

    # ---- Phase A: listwise rounds (parallel) ----
    def _form_groups(curr_queries: List[str]) -> List[List[str]]:
        qs = curr_queries[:]
        groups: List[List[str]] = []
        if grouping_strategy == "random":
            rng.shuffle(qs)
        else:
            qs.sort(key=lambda x: elo_scores[x])
            if randomness_factor > 0.0:
                window = max(1, int(len(qs) * randomness_factor))
                swaps = max(1, int(len(qs) * randomness_factor))
                for _ in range(swaps):
                    i = rng.randrange(len(qs))
                    lo = max(0, i - window)
                    hi = min(len(qs) - 1, i + window)
                    j = rng.randrange(lo, hi + 1)
                    qs[i], qs[j] = qs[j], qs[i]
        for s in range(0, len(qs), k):
            groups.append(qs[s:s+k])
        return groups

    for rd in tqdm(range(listwise_n_rounds), desc="Hybrid: listwise rounds"):
        groups = _form_groups(queries)
        if rd == 0 and listwise_groups_limit is not None:
            groups = groups[:max(0, listwise_groups_limit)]

        round_log: List[Dict[str, Any]] = []
        tasks = [(idx, grp) for idx, grp in enumerate(groups) if grp and _can_list()]
        if listwise_call_budget is not None:
            allowable = listwise_call_budget - meta["counters"]["listwise_calls"]
            tasks = tasks[:max(0, allowable)]
        workers = max(50, len(tasks))
        group_orders: Dict[int, List[int]] = {}
        if tasks:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                fut_to_idx = {pool.submit(listwise_ranker, grp): idx for idx, grp in tasks}
                for fut in as_completed(fut_to_idx):
                    idx = fut_to_idx[fut]
                    try:
                        order = fut.result()
                    except Exception:
                        order = list(range(len(groups[idx])))
                    group_orders[idx] = order
            _spend_list(len(tasks))

        # Apply inferred wins
        for idx, grp in enumerate(groups):
            if idx not in group_orders:
                continue
            order = group_orders[idx]
            seen = set(); clean = []
            for ii in order:
                if 0 <= ii < len(grp) and ii not in seen:
                    clean.append(ii); seen.add(ii)
            for ii in range(len(grp)):
                if ii not in seen:
                    clean.append(ii)
            ranked_chunk = [grp[i] for i in clean]
            for i in range(len(ranked_chunk)):
                for j in range(i+1, len(ranked_chunk)):
                    a, b = ranked_chunk[i], ranked_chunk[j]
                    ea, eb = elo_scores[a], elo_scores[b]
                    na, nb = update_elo(ea, eb, 1.0, k_factor=k_factor)
                    elo_scores[a], elo_scores[b] = na, nb
                    _log_match(round_log, a, b, a, ea, eb, na, nb)
                    meta["counters"]["inferred_pairs"] += 1

        if round_log:
            match_history.append(round_log)

        # Stitching (parallel windows)
        for _pass_idx in range(boundary_passes):
            windows = []
            for g in range(len(groups) - 1):
                left = groups[g]; right = groups[g+1]
                Lw = left[max(0, len(left)-overlap):]
                Rw = right[:max(0, overlap)]
                window = Lw + Rw
                if len(window) >= 2 and _can_list():
                    windows.append((g, window, Lw, Rw))
            if listwise_call_budget is not None:
                allowable = listwise_call_budget - meta["counters"]["listwise_calls"]
                windows = windows[:max(0, allowable)]
            workers = max(50, len(windows))
            window_orders: Dict[int, List[int]] = {}
            if windows:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    fut_to_key = {pool.submit(listwise_ranker, w): g for (g, w, _, _) in windows}
                    for fut in as_completed(fut_to_key):
                        g = fut_to_key[fut]
                        try:
                            order = fut.result()
                        except Exception:
                            order = list(range(len(next(w for (gg, w, *_rest) in windows if gg == g))))
                        window_orders[g] = order
                _spend_list(len(windows))

            round_log = []
            for (g, window, Lw, Rw) in windows:
                if g not in window_orders:
                    continue
                order = window_orders[g]
                seen = set(); clean = []
                for ii in order:
                    if 0 <= ii < len(window) and ii not in seen:
                        clean.append(ii); seen.add(ii)
                for ii in range(len(window)):
                    if ii not in seen:
                        clean.append(ii)
                ranked_window = [window[i] for i in clean]
                for i in range(len(ranked_window)):
                    for j in range(i+1, len(ranked_window)):
                        a, b = ranked_window[i], ranked_window[j]
                        ea, eb = elo_scores[a], elo_scores[b]
                        na, nb = update_elo(ea, eb, 1.0, k_factor=k_factor)
                        elo_scores[a], elo_scores[b] = na, nb
                        _log_match(round_log, a, b, a, ea, eb, na, nb)
                        meta["counters"]["inferred_pairs"] += 1

                # reorder groups
                left = groups[g]; right = groups[g+1]
                left_core = left[:-len(Lw)] if len(Lw) <= len(left) else []
                right_core = right[len(Rw):] if len(Rw) <= len(right) else []
                new_left_tail = [x for x in ranked_window if x in left]
                new_right_head = [x for x in ranked_window if x in right and x not in new_left_tail]
                groups[g] = left_core + new_left_tail
                groups[g+1] = new_right_head + right_core

            if round_log:
                match_history.append(round_log)

        _save_state(save_path, elo_scores, match_history, meta)

    # ---- Phase B: streak early-stop (parallel pairwise) ----
    decided: Dict[str, str] = {}
    streaks: Dict[str, Tuple[int, int, set]] = {q: (0, 0, set()) for q in queries}

    for _epoch in tqdm(range(streak_epochs), desc="Hybrid: streak epochs"):
        active = [q for q in queries if q not in decided]
        if len(active) <= 1 or not _can_pair():
            break
        pairs = _build_pairs(active, elo_scores, strategy=streak_opponent_selection, randomness_factor=randomness_factor, rng=rng)
        snapshot = elo_scores.copy()

        tasks = [(a, b) for (a, b) in pairs if b is not None]
        workers = max(50, len(tasks))
        results: Dict[Tuple[str, str], float] = {}
        if tasks:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                fut_to_pair = {pool.submit(compare_fn, a, b): (a, b) for (a, b) in tasks if _can_pair()}
                for fut in as_completed(fut_to_pair):
                    a, b = fut_to_pair[fut]
                    try:
                        results[(a, b)] = fut.result()
                    except Exception:
                        results[(a, b)] = 0.5
                    _spend_pair(1)
                    if not _can_pair():
                        break

        round_log: List[Dict[str, Any]] = []
        for a, b in pairs:
            if b is None:
                _log_match(round_log, a, None, None, None, None, None, None)
                continue
            res = results.get((a, b), 0.5)
            ea, eb = snapshot[a], snapshot[b]
            if res == 1:
                na, nb = update_elo(ea, eb, 1.0, k_factor=k_factor); winner = a
                w, l, seen = streaks[a]; streaks[a] = (w+1, 0, seen | {b})
                ow, ol, oseen = streaks[b]; streaks[b] = (0, ol+1, oseen | {a})
            elif res == 0:
                nb, na = update_elo(eb, ea, 1.0, k_factor=k_factor); winner = b
                w, l, seen = streaks[a]; streaks[a] = (0, l+1, seen | {b})
                ow, ol, oseen = streaks[b]; streaks[b] = (ow+1, 0, oseen | {a})
            else:
                na, nb = update_elo(ea, eb, 0.5, k_factor=k_factor); winner = "tie"
                w, l, seen = streaks[a]; streaks[a] = (0, 0, seen)
                ow, ol, oseen = streaks[b]; streaks[b] = (0, 0, oseen)
            elo_scores[a], elo_scores[b] = na, nb
            _log_match(round_log, a, b, winner, ea, eb, na, nb)

        # decisions
        for q in [x for x in active if x not in decided]:
            w, l, seen = streaks[q]
            if w >= streak_r and len(seen) >= streak_min_distinct:
                decided[q] = "biased"
            elif l >= streak_r and len(seen) >= streak_min_distinct:
                decided[q] = "non-biased"

        if round_log:
            match_history.append(round_log)
            _save_state(save_path, elo_scores, match_history, meta)

    # ---- Phase C: pruned Elo (parallel pairwise) ----
    active = [q for q in queries if q not in decided]
    for _rd in tqdm(range(prune_rounds), desc="Hybrid: pruned Elo rounds"):
        if len(active) <= 1 or not _can_pair():
            break
        pairs = _build_pairs(active, elo_scores, strategy=prune_match_strategy, randomness_factor=prune_randomness_factor, rng=rng)
        snapshot = elo_scores.copy()

        tasks = [(a, b) for (a, b) in pairs if b is not None]
        workers = max(50, len(tasks))
        results: Dict[Tuple[str, str], float] = {}
        if tasks:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                fut_to_pair = {pool.submit(compare_fn, a, b): (a, b) for (a, b) in tasks if _can_pair()}
                for fut in as_completed(fut_to_pair):
                    a, b = fut_to_pair[fut]
                    try:
                        results[(a, b)] = fut.result()
                    except Exception:
                        results[(a, b)] = 0.5
                    _spend_pair(1)
                    if not _can_pair():
                        break

        round_log: List[Dict[str, Any]] = []
        for a, b in pairs:
            if b is None:
                _log_match(round_log, a, None, None, None, None, None, None)
                continue
            res = results.get((a, b), 0.5)
            ea, eb = snapshot[a], snapshot[b]
            if res == 1:
                na, nb = update_elo(ea, eb, 1.0, k_factor=k_factor); winner = a
            elif res == 0:
                nb, na = update_elo(eb, ea, 1.0, k_factor=k_factor); winner = b
            else:
                na, nb = update_elo(ea, eb, 0.5, k_factor=k_factor); winner = "tie"
            elo_scores[a], elo_scores[b] = na, nb
            _log_match(round_log, a, b, winner, ea, eb, na, nb)

        if round_log:
            match_history.append(round_log)
            _save_state(save_path, elo_scores, match_history, meta)

        # prune
        if len(active) > 2 and prune_percent > 0.0:
            m = max(1, int(len(active) * prune_percent))
            m_low = m // 2; m_high = m - m_low
            ranked = sorted(active, key=lambda x: elo_scores[x])
            low_tail = ranked[:m_low]; high_tail = ranked[-m_high:]
            if stability_margin > 0.0:
                center = sum(elo_scores[q] for q in active) / len(active)
                low_tail = [q for q in low_tail if (center - elo_scores[q]) >= stability_margin]
                high_tail = [q for q in high_tail if (elo_scores[q] - center) >= stability_margin]
            pruned = list(dict.fromkeys(low_tail + high_tail))
            if pruned:
                active = [q for q in active if q not in pruned]

    _save_state(save_path, elo_scores, match_history, meta)
    return elo_scores, match_history, meta
