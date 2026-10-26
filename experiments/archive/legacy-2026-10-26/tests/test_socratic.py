from socratic import rubric
from socratic_data import HELDOUT, TRAIN


def test_dataset_sizes_and_separation():
    assert sum(map(len, TRAIN.values())) == 60
    assert sum(map(len, HELDOUT.values())) == 100
    training_claims = {s for pairs in TRAIN.values() for s, _ in pairs}
    heldout_claims = {s for claims in HELDOUT.values() for s in claims}
    assert training_claims.isdisjoint(heldout_claims)


def test_rubric_rejects_generic_question():
    claim = "A program that passes tests is correct."
    assert not rubric(claim, "What do you think?")["substantive_proxy"]
    assert not rubric(claim, "Is the statement correct, or not?")["substantive_proxy"]
    assert rubric(claim, "Which untested inputs could make the program fail?")["substantive_proxy"]


def test_rubric_recalls_hand_written_examples():
    labeled = [rubric(claim, question)["substantive_proxy"]
               for pairs in TRAIN.values() for claim, question in pairs]
    assert sum(labeled) >= 57
