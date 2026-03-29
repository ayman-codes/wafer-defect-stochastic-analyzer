import numpy as np
import scipy.linalg
import logging
import sys
import os

# Inter-module bridging: Import Phase 1 DTMC solver
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../01_dtmc_baseline/src')))
from dtmc import DTMC

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class CTMC:
    """
    Continuous-Time Markov Chain engine.
    Implements Uniformization to interface with discrete-time solvers.
    """
    def __init__(self, states, generator_matrix, initial_probs, emissions=None):
        self.states = states
        self.Q = np.array(generator_matrix, dtype=float)
        self.pi_0 = np.array(initial_probs, dtype=float)
        self.emissions = emissions if emissions else {}
        self._validate_Q()

    def _validate_Q(self):
        N = len(self.states)
        if self.Q.shape != (N, N):
            raise ValueError(f"Generator matrix Q must be square with dimensions ({N}, {N}).")
        
        rows_sum = np.sum(self.Q, axis=1)
        if not np.all(np.isclose(rows_sum, 0)):
            raise ValueError("Invalid Generator Matrix: Each row must sum to 0.")
        
        Q_check = self.Q.copy()
        np.fill_diagonal(Q_check, np.inf)
        if np.any(Q_check < 0) and np.any(Q_check != np.inf):
            raise ValueError("Invalid Generator Matrix: Off-diagonal elements must be non-negative.")

    def discretize(self, delta_t=None):
        I = np.eye(len(self.states))
        max_rate = np.max(np.abs(np.diag(self.Q)))
        stability_limit = 1.0 / max_rate

        if delta_t is None:
            delta_t = 1.0 / (max_rate + 1.0)
            logging.info(f"Delta t not provided. Using stable default: {delta_t:.4f}")
        elif delta_t > stability_limit:
            logging.warning(f"Unstable Delta! {delta_t} > {stability_limit:.4f} (1/max_rate). "
                            "Probabilities may become negative.")

        P = I + (self.Q * delta_t)

        if np.any(P < 0) or np.any(P > 1):
            raise ValueError("Discretized matrix P has invalid probabilities.")

        return DTMC(self.states, P, self.pi_0, self.emissions)

    def get_analytical_dist(self, time):
        P_t = scipy.linalg.expm(self.Q * time)
        return self.pi_0.dot(P_t)

    def get_steps(self, time, delta_t):
        return int(time / delta_t)