from app.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_ranks_item_in_both_lists_above_single_list_items():
    vector_ranked = ["a", "b", "c"]
    bm25_ranked = ["b", "d", "e"]

    fused = reciprocal_rank_fusion([vector_ranked, bm25_ranked], k=60)
    fused_ids = [item_id for item_id, _ in fused]

    # "b" appears in both lists, so it should fuse to the top.
    assert fused_ids[0] == "b"
    assert set(fused_ids) == {"a", "b", "c", "d", "e"}


def test_rrf_score_matches_formula():
    fused = dict(reciprocal_rank_fusion([["x"]], k=60))
    assert fused["x"] == 1.0 / (60 + 1)


def test_rrf_is_deterministic_for_ties():
    # Two items tied in score (each appears once, same rank) should keep
    # first-seen order.
    fused = reciprocal_rank_fusion([["a"], ["b"]], k=60)
    assert [i for i, _ in fused] == ["a", "b"]


def test_rrf_handles_empty_lists():
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []
