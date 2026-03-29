import numpy as np
import pandas as pd
import sys
import os
import logging

# Inter-module bridging: Import Phase 2 CTMC solver
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../02_ctmc_kinetics/src')))
from ctmc import CTMC

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def execute_gspn_pipeline():
    states = ['S0', 'S1']
    pi_0 = [1.0, 0.0] 
    
    rate_s0_to_s1 = 1.0 / 7.0
    rate_s1_to_s0 = 1.0 / 10.0
    rate_production_s1 = 1.0
    
    Q = [
        [-rate_s0_to_s1, rate_s0_to_s1],
        [rate_s1_to_s0, -rate_s1_to_s0]
    ]
    
    emissions = {
        0: {"OK": 0.9, "Defect": 0.1},
        1: {"OK": 0.95, "Defect": 0.05}
    }

    ctmc_system = CTMC(states, Q, pi_0, emissions)
    deltas_to_test = [2.0, 1.0, 0.5, 0.25, 0.1]
    
    print(f"{'Delta':<10} | {'Steady P(S0)':<15} | {'Steady P(S1)':<15} | {'TP(S0->S1)':<15} | {'P(S1 Empty)':<15} | {'TP(S1 Prod)':<15}")
    print("-" * 95)

    for delta in deltas_to_test:
        dtmc_instance = ctmc_system.discretize(delta)
        steady_dist = dtmc_instance.solve_steady_state_auto()
        
        if steady_dist is not None:
            pi_s0 = steady_dist[0]
            pi_s1 = steady_dist[1]
            
            throughput_s0_s1 = pi_s0 * rate_s0_to_s1
            prob_s1_empty = pi_s0
            throughput_s1_prod = pi_s1 * rate_production_s1
            
            print(f"{delta:<10.4f} | {pi_s0:<15.4f} | {pi_s1:<15.4f} | {throughput_s0_s1:<15.4f} | {prob_s1_empty:<15.4f} | {throughput_s1_prod:<15.4f}")
        else:
            print(f"Failed to solve for Delta={delta}")

if __name__ == "__main__":
    execute_gspn_pipeline()