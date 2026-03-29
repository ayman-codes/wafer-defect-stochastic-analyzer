import pytest
import numpy as np
import sys
import os

# Dynamically append src to path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from dtmc import DTMC

@pytest.fixture
def valid_system():
    states = ["Source 0", "Source 1"]
    P = [[0.6, 0.4], [0.3, 0.7]]
    pi_0 = [1.0, 0.0]
    return states, P, pi_0

def test_dtmc_initialization(valid_system):
    states, P, pi_0 = valid_system
    dtmc = DTMC(states, P, pi_0)
    assert dtmc.P.shape == (2, 2)

def test_invalid_stochastic_matrix(valid_system):
    states, _, pi_0 = valid_system
    invalid_P = [[0.6, 0.5], [0.3, 0.7]] # Row 0 sums to 1.1
    with pytest.raises(ValueError):
        DTMC(states, invalid_P, pi_0)

def test_transient_analysis(valid_system):
    states, P, pi_0 = valid_system
    dtmc = DTMC(states, P, pi_0)
    pi_1 = dtmc.solve_transient(1)
    np.testing.assert_allclose(pi_1, [0.6, 0.4])

def test_steady_state_convergence(valid_system):
    states, P, pi_0 = valid_system
    dtmc = DTMC(states, P, pi_0)
    steady_state = dtmc.solve_steady_state_auto()
    expected = [0.42857143, 0.57142857]
    np.testing.assert_allclose(steady_state, expected, atol=1e-6)