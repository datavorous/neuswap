from compare_personas import remap, summarize


def test_anonymized_order_maps_back_to_candidate_and_preserves_separate_axes():
    raw = {"target_winner": "B", "target_margin": 2, "coherence_winner": "A",
           "world_conflict_A": False, "world_conflict_B": False,
           "unsupported_world_A": True, "unsupported_world_B": False,
           "personality_change": 3, "behavior_change": "The character schemes."}
    judgment = remap(raw, is_a=False)
    assert judgment["target_winner"] == "candidate"
    assert judgment["coherence_winner"] == "baseline"
    assert judgment["candidate_world_conflict"] is False
    assert judgment["baseline_unsupported_world"] is True


def test_paired_rate_keeps_ties_and_coherence_separate():
    rows = [{"target_winner": winner, "target_margin": margin,
             "coherence_winner": coherence, "personality_change": change,
             "candidate_world_conflict": False, "candidate_unsupported_world": False}
            for winner, margin, coherence, change in
            [("candidate", 1, "baseline", 2), ("tie", 0, "tie", 1),
             ("baseline", 2, "candidate", 3)]]
    result = summarize(rows)
    assert result["paired_win_rate"] == 0.5
    assert result["mean_signed_margin"] == -1 / 3
    assert result["coherence_better"] == result["coherence_worse"] == 1
