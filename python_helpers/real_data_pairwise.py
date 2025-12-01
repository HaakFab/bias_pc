from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os

from tqdm import tqdm

# Reuse ELO math from helpers to avoid duplication
from .bias_ranking_methods import expected_score, update_elo


# -----------------------------------------------------------------------------
# LLM result parsing helpers
# -----------------------------------------------------------------------------
def _safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        return None


def _vote_to_score(vote: Any) -> float:
    """
    Map a single LLM vote to {1.0 (A wins), 0.0 (B wins), 0.5 (tie)}.
    Accepts various shapes:
      - {"winner": "A"|"B"|"tie"} or {"winner_index": 0|1|None}
      - raw string "A"/"B"/"tie"
      - int 0/1 or None for tie
    """
    if isinstance(vote, dict):
        if "winner" in vote:
            w = str(vote["winner"]).strip().lower()
            if w in ("a", "left", "first"):  # A
                return 1.0
            if w in ("b", "right", "second"):  # B
                return 0.0
            return 0.5
        if "winner_index" in vote:
            idx = vote["winner_index"]
            if idx == 0:
                return 1.0
            if idx == 1:
                return 0.0
            return 0.5
    if isinstance(vote, str):
        w = vote.strip().lower()
        if w in ("a", "left", "first"):
            return 1.0
        if w in ("b", "right", "second"):
            return 0.0
        return 0.5
    if isinstance(vote, (int, float)):
        if vote == 0:
            return 1.0
        if vote == 1:
            return 0.0
        return 0.5
    return 0.5


# -----------------------------------------------------------------------------
# Pairwise comparator factories (OpenAI / Gemini)
# -----------------------------------------------------------------------------
def make_pairwise_comparator(
    *,
    prompt_system: str,
    prompt_user_template: str,
    model: str,
    provider: str = "openai",
    n_votes: int = 3,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    max_workers: int = 8,
) -> Callable[[str, str], float]:
    """
    Build a deterministic pairwise comparator function backed by an LLM.
    Returns: compare_fn(a, b) -> float in {1.0, 0.0, 0.5}.
    - provider: "openai" or "gemini"
    - The function aggregates n_votes via majority; ties -> 0.5.
    Environment:
      - OPENAI_API_KEY for provider='openai'
      - GOOGLE_API_KEY for provider='gemini'
    """
    if provider not in {"openai", "gemini"}:
        raise ValueError("provider must be 'openai' or 'gemini'")
    if provider == "openai":
        try:
            import openai
            client = openai.OpenAI()
        except Exception as e:
            raise RuntimeError("Failed to initialize OpenAI client. Ensure `pip install openai` and OPENAI_API_KEY set.") from e
    else:
        try:
            import google.generativeai as genai
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GOOGLE_API_KEY not set in environment.")
            genai.configure(api_key=api_key)
            client = genai.GenerativeModel(model)
        except Exception as e:
            raise RuntimeError("Failed to initialize Gemini client. Ensure `pip install google-generativeai` and GOOGLE_API_KEY set.") from e

    def _single_vote(a: str, b: str) -> float:
        user_msg = prompt_user_template.format(a=a, b=b)
        try:
            if provider == "openai":
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    seed=seed,
                    messages=[
                        {"role": "system", "content": prompt_system},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or "{}"
                data = _safe_parse_json(content) or {}
                return _vote_to_score(data)
            else:
                # Gemini JSON content; be lenient and parse best-effort
                resp = client.generate_content(
                    [{"text": prompt_system}, {"text": user_msg}],
                    safety_settings=None,
                )
                text = (resp.text or "").strip()
                data = _safe_parse_json(text) or {"winner": text}
                return _vote_to_score(data)
        except Exception:
            return 0.5

    def compare_fn(a: str, b: str) -> float:
        # parallelize votes for latency
        votes: List[float] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_single_vote, a, b) for _ in range(max(1, n_votes))]
            for fut in as_completed(futures):
                try:
                    votes.append(float(fut.result()))
                except Exception:
                    votes.append(0.5)
        # aggregate: majority of winners, else tie
        wins_a = sum(1 for v in votes if v == 1.0)
        wins_b = sum(1 for v in votes if v == 0.0)
        if wins_a > wins_b:
            return 1.0
        if wins_b > wins_a:
            return 0.0
        return 0.5

    return compare_fn


# -----------------------------------------------------------------------------
# Plain ELO pairwise runner for real data (uses a user-supplied compare_fn)
# -----------------------------------------------------------------------------
def pairwise_comparison_elo(
    texts: List[str],
    compare_fn: Callable[[str, str], float],
    *,
    n_rounds: int = 24,
    initial_elo: float = 1500.0,
    k_factor: int = 32,
    seed: Optional[int] = 42,
    progress: bool = True,
) -> Tuple[Dict[str, float], List[List[Dict[str, Any]]]]:
    """
    Lightweight ELO tournament on arbitrary texts using any compare_fn.
    Returns:
      - elo_scores: {text -> elo}
      - match_history: per-round logs with winners and elo deltas
    """
    import random

    rng = random.Random(seed)
    elo: Dict[str, float] = {t: initial_elo for t in texts}
    match_history: List[List[Dict[str, Any]]] = []

    rounds_iter = range(n_rounds)
    rounds_iter = tqdm(rounds_iter, desc="Running ELO rounds") if progress else rounds_iter

    for _ in rounds_iter:
        # random shuffle and pair off
        pool = texts[:]
        rng.shuffle(pool)
        pairs = list(zip(pool[::2], pool[1::2]))
        if len(pool) % 2 == 1:
            # odd one out -> no-op record to keep history aligned
            pairs.append((pool[-1], None))

        snapshot = elo.copy()
        round_log: List[Dict[str, Any]] = []
        for a, b in pairs:
            if b is None:
                round_log.append({"text1": a, "text2": None, "winner": None, "elo_change": (0.0, 0.0)})
                continue
            ea, eb = snapshot[a], snapshot[b]
            outcome = compare_fn(a, b)  # 1.0 A wins, 0.0 B wins, 0.5 tie
            if outcome == 1.0:
                na, nb = update_elo(ea, eb, 1.0, k_factor=k_factor)
                winner = a
            elif outcome == 0.0:
                nb, na = update_elo(eb, ea, 1.0, k_factor=k_factor)
                winner = b
            else:
                na, nb = update_elo(ea, eb, 0.5, k_factor=k_factor)
                winner = "tie"
            elo[a], elo[b] = na, nb
            round_log.append({"text1": a, "text2": b, "winner": winner, "elo_change": (na - ea, nb - eb)})

        match_history.append(round_log)

    return elo, match_history


# -----------------------------------------------------------------------------
# Convenience: OpenAI/Gemini pairwise runner with a prompt
# -----------------------------------------------------------------------------
def run_pairwise_with_llm(
    texts: List[str],
    *,
    model: str,
    provider: str = "openai",
    n_rounds: int = 24,
    prompt_system: str,
    prompt_user_template: str,
    n_votes: int = 3,
    temperature: float = 0.0,
    seed: Optional[int] = 42,
) -> Tuple[Dict[str, float], List[List[Dict[str, Any]]]]:
    """
    High-level helper: build comparator and run ELO in one call.
    """
    compare_fn = make_pairwise_comparator(
        prompt_system=prompt_system,
        prompt_user_template=prompt_user_template,
        model=model,
        provider=provider,
        n_votes=n_votes,
        temperature=temperature,
        seed=seed,
    )
    return pairwise_comparison_elo(
        texts,
        compare_fn,
        n_rounds=n_rounds,
        seed=seed,
    )


