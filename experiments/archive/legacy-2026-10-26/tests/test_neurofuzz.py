import numpy as np

from neurofuzz import rank_neurons, tic_metrics
from writeup_experiment import audit_pair, clean_rewrite


def test_tic_metrics_detects_prespecified_patterns():
    text = "It is not simple, but useful—more than just a test. It helps. It helps."
    metrics = tic_metrics(text)
    assert metrics["em_dash"] == 1
    assert metrics["contrast"] == 2
    assert metrics["repeated_openings"] == 1


def test_ranking_favors_consistent_meaningful_activation():
    differences = np.array([[[0.5, 0.001]], [[0.6, 0.001]], [[0.4, 0.001]]])
    assert rank_neurons(differences, 1)[0].neuron == 0


def test_rewrite_screen_catches_changed_numbers_and_prefaces():
    assert clean_rewrite("Here's a rewritten version:\n\nThe chip holds 4 MB.") == "The chip holds 4 MB."
    assert audit_pair("The chip holds 4 MB of data.", "The chip holds 8 MB of data.") == ["number_mismatch"]
