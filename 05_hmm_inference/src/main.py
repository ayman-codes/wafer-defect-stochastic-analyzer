import sys
import os
import itertools
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../01_dtmc_baseline/src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from dtmc import DTMC
from hmm_solver import HMMSolver

def get_most_likely_observations(solver: HMMSolver, initial_pi: np.ndarray, steps: int = 3) -> tuple[tuple, float]:
    obs_options = ["OK", "Defect"]
    all_seqs = list(itertools.product(obs_options, repeat=steps))
    best_seq = None
    max_prob = -1.0

    for seq in all_seqs:
        prob = solver.solve_forward(list(seq), initial_pi)
        if prob > max_prob:
            max_prob = prob
            best_seq = seq
    return best_seq, max_prob

def execute_inference():
    states = ["Source 0", "Source 1"]
    P = [[0.6, 0.4], [0.3, 0.7]]
    emissions = {
        0: {"OK": 0.9, "Defect": 0.1},
        1: {"OK": 0.95, "Defect": 0.05}
    }
    
    solver = HMMSolver(states, P, emissions)

    # Dynamic cross-module steady-state resolution
    pi_start_dtmc = [1.0, 0.0]
    dtmc_engine = DTMC(states, P, pi_start_dtmc, emissions)
    steady_state_vector = np.array(dtmc_engine.solve_steady_state_auto())

    pi_start = np.array([1.0, 0.0])
    pi_step1 = pi_start @ np.array(P)
    
    print("=" * 60)
    print(" ENTERPRISE HMM INFERENCE SYSTEM")
    print("=" * 60)

    # 1. Optimal Sequence Prediction (Transient)
    seq, prob = get_most_likely_observations(solver, pi_step1)
    print(f"Optimal Observation Sequence (Transient T=3): {seq}")
    print(f"Sequence Probability: {prob:.6f}")
    
    # 2. Sequential Failure Probability (Transient)
    p_fail_transient = solver.solve_forward(["Defect", "Defect", "Defect"], pi_step1)
    print(f"P(Defect, Defect, Defect | Transient): {p_fail_transient:.8f}")

    # 3. Sequential Failure Probability (Equilibrium)
    p_fail_steady = solver.solve_forward(["Defect", "Defect", "Defect"], steady_state_vector)
    print(f"P(Defect, Defect, Defect | Equilibrium): {p_fail_steady:.8f}")

    # 4. Hidden State Decoding (Transient)
    path_trans, prob_trans = solver.solve_viterbi(["OK", "OK", "OK"], pi_step1)
    print(f"Viterbi Path (OK, OK, OK | Transient): {[states[i] for i in path_trans]}")
    print(f"Joint Probability: {prob_trans:.6f}")

    # 5. Hidden State Decoding (Equilibrium)
    path_steady, prob_steady = solver.solve_viterbi(["OK", "OK", "OK"], steady_state_vector)
    print(f"Viterbi Path (OK, OK, OK | Equilibrium): {[states[i] for i in path_steady]}")
    print(f"Joint Probability: {prob_steady:.6f}")
    print("=" * 60)

if __name__ == "__main__":
    execute_inference()