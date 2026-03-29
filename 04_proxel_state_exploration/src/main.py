import sys
import os
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from engine import ProxelEngine

# Output handling isolated to main script
logging.basicConfig(level=logging.WARNING)

def run_system_evaluation():
    print("=" * 60)
    print(" STOCHASTIC PROXEL ENGINE: WAFER QUALITY TESTER")
    print("=" * 60)
    
    simulation_duration = 3600.0  # 1 hour
    delta_t_configs = [20.0, 10.0, 5.0, 2.0]
    
    # Statistical Defect Probabilities
    DEFECT_RATE_S0 = 0.10  # 1 - 0.90 OK Probability
    DEFECT_RATE_S1 = 0.05  # 1 - 0.95 OK Probability

    print(f"Target Duration: {simulation_duration}s\n")
    print(f"{'dt (s)':<8} | {'Total Tested':<15} | {'Defective S0':<15} | {'Defective S1':<15}")
    print("-" * 60)

    for dt in delta_t_configs:
        engine = ProxelEngine(dt=dt)
        throughput_s0, throughput_s1 = engine.run(total_time=simulation_duration)
        
        total_tested = throughput_s0 + throughput_s1
        defective_s0 = throughput_s0 * DEFECT_RATE_S0
        defective_s1 = throughput_s1 * DEFECT_RATE_S1
        
        print(f"{dt:<8.1f} | {total_tested:<15.2f} | {defective_s0:<15.2f} | {defective_s1:<15.2f}")
        
    print("=" * 60)

if __name__ == "__main__":
    sys.exit(run_system_evaluation())