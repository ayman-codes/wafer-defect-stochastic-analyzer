import numpy as np
from ctmc import CTMC

def execute_kinetics_pipeline():
    states = ["Source 0", "Source 1"]
    pi_0 = [1.0, 0.0]
    
    lambda_01 = 1.0 / 2.0 
    lambda_10 = 1.0 / 3.0 
    
    Q = [
        [-lambda_01,  lambda_01],
        [ lambda_10, -lambda_10]
    ]
    
    emissions = {
        0: {"OK": 0.9, "Defect": 0.1},
        1: {"OK": 0.95, "Defect": 0.05}
    }

    ctmc_system = CTMC(states, Q, pi_0, emissions)
    
    target_time = 8.0
    deltas = [2.0, 1.0, 0.5, 0.25, 0.1]
    
    exact_dist = ctmc_system.get_analytical_dist(target_time)
    exact_s0_t8 = exact_dist[0]
    
    print(f"ANALYTICAL GROUND TRUTH (t={target_time})")
    print(f"Exact Prob(Source 0): {exact_s0_t8:.4f}\n")
    print(f"{'Delta':<10} | {'Steps':<10} | {'P(S0) @ t=8':<15} | {'Error':<15} | {'Steady P(S0)':<15} | {'Avg P(OK) Steady':<15}")
    print("-" * 95)

    for delta in deltas:
        dtmc_instance = ctmc_system.discretize(delta)
        steps = ctmc_system.get_steps(target_time, delta)
        
        transient_dist = dtmc_instance.solve_transient(steps)
        prob_s0_t8 = transient_dist[0]
        error = abs(prob_s0_t8 - exact_s0_t8)
        
        steady_dist = dtmc_instance.solve_steady_state_auto()
        if steady_dist is not None:
            prob_s0_ss = steady_dist[0]
            prob_ok_ss = (steady_dist[0] * emissions[0]["OK"]) + (steady_dist[1] * emissions[1]["OK"])
        else:
            prob_s0_ss = float('nan')
            prob_ok_ss = float('nan')
            
        print(f"{delta:<10.4f} | {steps:<10} | {prob_s0_t8:<15.4f} | {error:<15.4f} | {prob_s0_ss:<15.4f} | {prob_ok_ss:<15.4f}")

if __name__ == "__main__":
    execute_kinetics_pipeline()