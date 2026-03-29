import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../02_ctmc_kinetics/src')))
from ctmc import CTMC

@pytest.fixture
def gspn_system():
    states = ['S0', 'S1']
    pi_0 = [1.0, 0.0]
    rate_s0_to_s1 = 1.0 / 7.0
    rate_s1_to_s0 = 1.0 / 10.0
    Q = [
        [-rate_s0_to_s1, rate_s0_to_s1],
        [rate_s1_to_s0, -rate_s1_to_s0]
    ]
    return CTMC(states, Q, pi_0)

def test_gspn_steady_state_and_throughput(gspn_system):
    dtmc_instance = gspn_system.discretize(0.1)
    steady_dist = dtmc_instance.solve_steady_state_auto()
    
    pi_s0 = steady_dist[0]
    pi_s1 = steady_dist[1]
    
    # Analytical expected values: 7/17 and 10/17
    np.testing.assert_allclose(pi_s0, 7/17, atol=1e-4)
    np.testing.assert_allclose(pi_s1, 10/17, atol=1e-4)
    
    throughput_s0_s1 = pi_s0 * (1.0 / 7.0)
    throughput_s1_prod = pi_s1 * 1.0
    
    np.testing.assert_allclose(throughput_s0_s1, 0.0588, atol=1e-4)
    np.testing.assert_allclose(throughput_s1_prod, 0.5882, atol=1e-4)