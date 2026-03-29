import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class DTMC:
    """
    A class to represent and solve Discrete-Time Markov Chains.
    Includes Transient Analysis capabilities.
    """
    def __init__(self, states, transition_matrix, initial_probs, emissions=None):
        self.states = states
        self.P = np.array(transition_matrix, dtype=float)
        self.pi_0 = np.array(initial_probs, dtype=float)
        self.emissions = emissions if emissions else {}
        self.convergence_steps = None
        self._validate()

    def _validate(self):
        N = len(self.states)
        if self.P.shape != (N, N):
            raise ValueError(f"Transition matrix shape {self.P.shape} does not match number of states ({N}).")
        if self.pi_0.shape[0] != N:
            raise ValueError(f"Initial probability vector length {self.pi_0.shape[0]} does not match number of states ({N}).")
        row_sums = self.P.sum(axis=1)
        if not np.all(np.isclose(row_sums, 1.0)):
            raise ValueError(f"Rows do not sum to 1.0. Found sums: {row_sums}")
        if not np.isclose(self.pi_0.sum(), 1.0):
            raise ValueError(f"Initial probability vector sums to {self.pi_0.sum()}, expected 1.0.")
        if np.any(self.P < 0) or np.any(self.pi_0 < 0):
            raise ValueError("Probabilities cannot be negative.")
        if self.emissions:
            for state_idx, output_dict in self.emissions.items():
                if state_idx >= N:
                    raise ValueError(f"Emission defined for non-existent state index {state_idx}.")
                total_prob = sum(output_dict.values())
                if not np.isclose(total_prob, 1.0):
                    raise ValueError(f"Emissions for State {state_idx} sum to {total_prob}, expected 1.0.")

    def solve_transient(self, t):
        if t < 0:
            raise ValueError("Time step t must be non-negative.")
        if t == 0:
            return self.pi_0
        P_t = np.linalg.matrix_power(self.P, t)
        return self.pi_0.dot(P_t)

    def get_emission_probability(self, t, emission_name):
        pi_t = self.solve_transient(t)
        total_prob = 0.0
        for state_idx, state_prob in enumerate(pi_t):
            state_emissions = self.emissions.get(state_idx, {})
            emission_prob = state_emissions.get(emission_name, 0.0)
            total_prob += state_prob * emission_prob
        return total_prob

    def is_limiting_distribution(self, pi):
        pi = np.array(pi, dtype=float)
        epsilon = 1e-15
        pi_next = pi.dot(self.P)
        is_stationary = np.all(np.abs(pi_next - pi) <= epsilon)
        if not is_stationary:
            logging.warning("Distribution is not stationary.")
            return False
        n_states = len(self.states)
        check_power = (n_states**2 * 2) + 2 if n_states < 4 else n_states**2 + 2
        P_high = np.linalg.matrix_power(self.P, check_power)
        is_ergodic = np.all(P_high > 1e-20)
        if not is_ergodic:
            logging.warning("Markov chain is not regular (ergodic).")
            return False
        elif is_stationary and is_ergodic:
            logging.info("Distribution is a limiting distribution.")
            return True

    def solve_steady_state_power(self, max_iter=100000, epsilon=1e-15):
        pi = self.pi_0.copy()
        for i in range(max_iter):
            pi_next = pi.dot(self.P)
            if np.all(np.abs(pi_next - pi) <= epsilon):
                logging.info(f"Power Method converged at step {i+1}.")
                self.convergence_steps = i + 1
                return pi_next
            pi = pi_next
        logging.critical("Power Method failed to converge.")
        return None

    def solve_steady_state_direct(self):
        try:
            N = len(self.states)
            A = self.P.T - np.eye(N)
            A[-1] = np.ones(N)
            b = np.zeros(N)
            b[-1] = 1
            pi = np.linalg.solve(A, b)
            logging.info("Direct Method solved successfully.")
            return pi
        except np.linalg.LinAlgError:
            logging.error("Direct Method failed. Matrix is Singular.")
            return None

    def solve_steady_state_auto(self):
        logging.info("Attempting Power Method...")
        pi_power = self.solve_steady_state_power(max_iter=100000)
        if pi_power is not None:
            return pi_power
        logging.warning("Possible NCD behavior detected. Switching to Direct Method...")
        return self.solve_steady_state_direct()