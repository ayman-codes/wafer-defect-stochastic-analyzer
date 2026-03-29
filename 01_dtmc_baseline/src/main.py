import numpy as np
import time
from dtmc import DTMC

def run_mcmc_verification(transition_matrix, emissions, num_trials=100000):
    """Executes MCMC simulations to verify Transient and Steady State values."""
    print(f"Running {num_trials} simulations...")
    start_time = time.time()
    
    s0_count_at_t8 = 0
    choices = [0, 1]
    
    for _ in range(num_trials):
        current_state = 0
        for _ in range(8):
            current_state = np.random.choice(choices, p=transition_matrix[current_state])
        if current_state == 0:
            s0_count_at_t8 += 1
            
    experimental_q1 = s0_count_at_t8 / num_trials
    
    current_state = 0 
    ok_item_count = 0
    
    for _ in range(num_trials):
        current_state = np.random.choice(choices, p=transition_matrix[current_state])
        prob_ok = emissions[current_state]["OK"]
        if np.random.random() < prob_ok:
            ok_item_count += 1

    experimental_q6 = ok_item_count / num_trials
    end_time = time.time()
    
    print(f"Simulation Complete in {end_time - start_time:.4f} seconds.")
    return experimental_q1, experimental_q6

if __name__ == "__main__":
    states = ["Source 0", "Source 1"]
    P = [
        [0.6, 0.4],
        [0.3, 0.7]
    ]
    pi_0 = [1.0, 0.0]
    emissions = {
        0: {"OK": 0.9, "Defect": 0.1},
        1: {"OK": 0.95, "Defect": 0.05}
    }

    dtmc_solver = DTMC(states, P, pi_0, emissions)

    # Q1: Probability of Source 0 after 8 minutes
    t = 8
    pi_8 = dtmc_solver.solve_transient(t)
    print(f"Time {t}: Vector = {pi_8.round(4)}")
    print(f"Question 1 (Prob Source 0): {pi_8[0]:.4f}")

    # Q2: Probability of 'OK' item in next minute (t=1)
    t_next = 1
    prob_ok = dtmc_solver.get_emission_probability(t_next, "OK")
    print(f"Question 2 (Prob OK at t={t_next}): {prob_ok:.4f}")

    # Q3/Q4: Steady State and Limiting Distribution Check
    steady_state = dtmc_solver.solve_steady_state_auto()
    if steady_state is not None:
        print(f"Steady State Distribution: {steady_state.round(4)}")
        is_limiting = dtmc_solver.is_limiting_distribution(steady_state)
        print(f"Model has a limiting distribution: {is_limiting}")
        
        # Q5: Probability for source 0 to be active in steady state
        print(f"Question 5 (Prob S0 Steady State): {steady_state[0]:.4f}")
        
        # Q6: Average probability of producing an OK item in steady state
        avg_ok_steady_state = (steady_state[0] * emissions[0]["OK"]) + (steady_state[1] * emissions[1]["OK"])
        print(f"Question 6 (Avg OK Rate Steady State): {avg_ok_steady_state:.4f}")

    # MCMC Verification
    exp_q1, exp_q6 = run_mcmc_verification(P, emissions)
    print(f"MCMC Q1 (Prob S0 at t=8): {exp_q1:.4f}")
    print(f"MCMC Q6 (Avg OK Rate Steady State): {exp_q6:.4f}")