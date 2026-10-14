"""Reciprocal Rank Fusion (RRF).

RRF combines multiple ranked lists (here: vector search and BM25) into one
ranking without needing to normalize or compare scores across retrievers
that produce scores on entirely different scales (cosine distance vs. BM25
score). Each item's fused score is the sum, over every list it appears in,
of 1 / (k + rank). A chunk ranked highly by *either* retriever rises to
the top; a chunk ranked highly by *both* rises even further. This is the
standard formulation from Cormack, Clarke & Buettcher (2009).
"""
from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Each ranked list is a list of IDs ordered best-first (rank 1 = best).

    Returns a list of (id, fused_score) sorted by fused_score descending.
    Deterministic: ties are broken by first-seen order across lists.
    """
    scores: dict[str, float] = {}
    first_seen_order: dict[str, int] = {}
    counter = 0

    for ranked_list in ranked_lists:
        for rank, item_id in enumerate(ranked_list, start=1):
            if item_id not in first_seen_order:
                first_seen_order[item_id] = counter
                counter += 1
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)

    return sorted(
        scores.items(),
        key=lambda item: (-item[1], first_seen_order[item[0]]),
    )
