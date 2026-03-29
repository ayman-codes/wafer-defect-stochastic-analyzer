import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ctmc import CTMC

@pytest.fixture
def valid_ctmc():
    states = ["Source 0", "Source 1"]
    Q = [[-0.5, 0.5], [0.3333, -0.3333]]
    pi_0 = [1.0, 0.0]
    return states, Q, pi_0

def test_ctmc_initialization(valid_ctmc):
    states, Q, pi_0 = valid_ctmc
    ctmc = CTMC(states, Q, pi_0)
    assert ctmc.Q.shape == (2, 2)

def test_invalid_generator_matrix_sums(valid_ctmc):
    states, _, pi_0 = valid_ctmc
    invalid_Q = [[-0.5, 0.6], [0.3333, -0.3333]] # Row 0 sums to 0.1
    with pytest.raises(ValueError):
        CTMC(states, invalid_Q, pi_0)

def test_invalid_generator_matrix_signs(valid_ctmc):
    states, _, pi_0 = valid_ctmc
    invalid_Q = [[0.5, -0.5], [-0.3333, 0.3333]] # Negative off-diagonals
    with pytest.raises(ValueError):
        CTMC(states, invalid_Q, pi_0)

def test_discretization_stability(valid_ctmc):
    states, Q, pi_0 = valid_ctmc
    ctmc = CTMC(states, Q, pi_0)
    
    # max_rate is 0.5, so stability_limit is 2.0. Delta 3.0 should raise ValueError due to negative probabilities.
    with pytest.raises(ValueError):
        ctmc.discretize(3.0)
        
    dtmc_safe = ctmc.discretize(1.0)
    np.testing.assert_allclose(dtmc_safe.P, [[0.5, 0.5], [0.3333, 0.6667]], atol=1e-4)

def test_analytical_distribution(valid_ctmc):
    states, Q, pi_0 = valid_ctmc
    ctmc = CTMC(states, Q, pi_0)
    pi_t = ctmc.get_analytical_dist(8.0)
    
    # Ground truth analytical solution at t=8 for this system
    expected_pi_t = [0.40076, 0.59923]
    np.testing.assert_allclose(pi_t, expected_pi_t, atol=1e-4)