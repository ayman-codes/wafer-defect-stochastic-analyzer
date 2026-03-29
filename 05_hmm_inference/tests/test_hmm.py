import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from hmm_solver import HMMSolver

@pytest.fixture
def baseline_system():
    states = ["S0", "S1"]
    P = [[0.6, 0.4], [0.3, 0.7]]
    emissions = {
        0: {"OK": 0.9, "Defect": 0.1},
        1: {"OK": 0.95, "Defect": 0.05}
    }
    return HMMSolver(states, P, emissions)

def test_forward_algorithm_deterministic(baseline_system):
    pi_initial = np.array([0.6, 0.4]) 
    observations = ["Defect", "Defect", "Defect"]
    expected_prob = 0.0004595
    
    result = baseline_system.solve_forward(observations, pi_initial)
    assert np.isclose(result, expected_prob, atol=1e-6), f"Forward calculation deviation. Expected {expected_prob}, got {result}"

def test_viterbi_algorithm_deterministic(baseline_system):
    pi_initial = np.array([0.6, 0.4])
    observations = ["OK", "OK", "OK"]
    expected_path = [1, 1, 1]
    expected_prob = 0.1680455
    
    path, prob = baseline_system.solve_viterbi(observations, pi_initial)
    assert path == expected_path, f"Viterbi path deviation. Expected {expected_path}, got {path}"
    assert np.isclose(prob, expected_prob, atol=1e-6), f"Viterbi probability deviation. Expected {expected_prob}, got {prob}"

def test_beam_search_preserves_viterbi_at_high_k(baseline_system):
    pi_initial = np.array([0.6, 0.4])
    observations = ["OK", "OK", "OK"]
    
    v_path, v_prob = baseline_system.solve_viterbi(observations, pi_initial)
    beams = baseline_system.solve_beam_search(observations, pi_initial, k=2)
    
    assert beams[0]['path'] == v_path
    assert np.isclose(beams[0]['prob'], v_prob, atol=1e-6)

def test_matrix_dimensionality(baseline_system):
    assert len(baseline_system.P) == baseline_system.num_states
    assert len(baseline_system.P[0]) == baseline_system.num_states
    assert len(baseline_system.emissions) == baseline_system.num_states