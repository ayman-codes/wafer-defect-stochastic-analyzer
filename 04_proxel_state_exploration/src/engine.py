import logging
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from proxel import Proxel, ProxelTree
from physics import NormalHazard

logger = logging.getLogger(__name__)

class ProxelEngine:
    """
    Race-Age discrete simulation model mapping concurrent stochastic events
    via probability elements (Proxels).
    """
    def __init__(self, dt: float):
        self.dt = dt
        self.tree = ProxelTree()
        
        # Hardware failure parameters per enterprise spec
        self.hazard_s0 = NormalHazard(mu=150.0, sigma=25.0)
        self.hazard_s1 = NormalHazard(mu=120.0, sigma=20.0)
        
        self.throughput_s0 = 0.0
        self.throughput_s1 = 0.0
        self.current_time = 0.0
        
        # Initialize initial state space
        self.tree.add(Proxel(state=1, age_vector=(0.0, 0.0), probability=1.0))
        logger.info(f"Proxel Engine initialized with dt={self.dt}")
        
    def step(self) -> None:
        """Discretizes and advances the stochastic process by dt."""
        next_tree = ProxelTree()
        next_tree.accumulated_loss = self.tree.accumulated_loss
        children_buffer = [] 
        
        for parent in self.tree:
            age_s0, age_s1 = parent.age_vector
            
            # Midpoint Rule Integration
            rate_0 = self.hazard_s0.get_rate(age_s0 + self.dt / 2)
            rate_1 = self.hazard_s1.get_rate(age_s1 + self.dt / 2)
            
            prob_fire_s0 = rate_0 * self.dt
            prob_fire_s1 = rate_1 * self.dt
            
            total_fire_prob = prob_fire_s0 + prob_fire_s1
            
            if total_fire_prob > 1.0:
                scale = 1.0 / total_fire_prob
                prob_fire_s0 *= scale
                prob_fire_s1 *= scale
                prob_stay = 0.0
            else:
                prob_stay = 1.0 - total_fire_prob
                
            # Child Node A: Survivor (System Continues)
            if prob_stay > 0.0:
                children_buffer.append(Proxel(
                    state=1,
                    age_vector=(age_s0 + self.dt, age_s1 + self.dt),
                    probability=parent.probability * prob_stay,
                    rewards=parent.rewards
                ))
            
            # Child Node B: Event S0 Occurs
            if prob_fire_s0 > 0:
                flow = parent.probability * prob_fire_s0
                self.throughput_s0 += flow
                children_buffer.append(Proxel( 
                    state=1,
                    age_vector=(0.0, age_s1 + self.dt),
                    probability=flow,
                    rewards=parent.rewards + 1
                ))
                
            # Child Node C: Event S1 Occurs
            if prob_fire_s1 > 0:
                flow = parent.probability * prob_fire_s1
                self.throughput_s1 += flow
                children_buffer.append(Proxel( 
                    state=1,
                    age_vector=(age_s0 + self.dt, 0.0),
                    probability=flow,
                    rewards=parent.rewards + 1
                )) 
       
        next_tree.batch_add(children_buffer)
        next_tree.prune()
        next_tree.merge(age_tolerance=1e-1) 
        
        self.tree = next_tree
        self.current_time += self.dt

    def run(self, total_time: float):
        """Executes the simulation to continuous completion."""
        steps = int(total_time / self.dt)
        for _ in range(steps):
            self.step()
        logger.info(f"Simulation completed. Final State Space Size: {len(self.tree)}")
        return self.throughput_s0, self.throughput_s1