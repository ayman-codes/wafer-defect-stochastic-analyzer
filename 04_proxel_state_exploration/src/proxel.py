import bisect
import logging
from dataclasses import dataclass, field
from typing import Tuple, List

# Core Math logging isolation
logger = logging.getLogger(__name__)

MIN_PROB_THRESHOLD = 1e-7

@dataclass(order=True)
class Proxel:
    """
    The probability flow representing a specific path in the expanded state space.
    Sorting Order: state -> age_vector
    """
    state: int
    age_vector: Tuple[float, ...] 
    
    probability: float = field(compare=False)
    rewards: float = field(default=0.0, compare=False)
    
    def __post_init__(self):
        logger.debug(f"Proxel initialized: State={self.state}, P={self.probability:.2e}")


class ProxelTree:
    """
    Manages a sorted collection of Proxels to maintain O(log N) insertion 
    and O(N) contiguous merging without requiring complex C++ tree extensions.
    """
    def __init__(self):
        self.proxels: List[Proxel] = []
        self.accumulated_loss: float = 0.0
        
    def add(self, proxel: Proxel) -> None:
        bisect.insort(self.proxels, proxel)
        
    def prune(self, threshold: float = MIN_PROB_THRESHOLD) -> None:
        """Prunes probability mass below operational threshold to combat state space explosion."""
        initial_size = len(self.proxels)
        for i in range(initial_size - 1, -1, -1):
            if self.proxels[i].probability < threshold:
                self.accumulated_loss += self.proxels[i].probability
                self.proxels.pop(i)
                
    def merge(self, age_tolerance: float = 1e-1) -> None:
        """
        Combines contiguous proxels sharing similar state and age vectors.
        Utilizes weighted averages to conserve probability mass and rewards.
        """
        if not self.proxels:
            return
        
        new_proxels = []
        current = self.proxels[0]
        
        for next_proxel in self.proxels[1:]:
            if current.state == next_proxel.state:
                age_diffs = [abs(a - b) for a, b in zip(current.age_vector, next_proxel.age_vector)]
                
                if max(age_diffs) < age_tolerance:
                    total_p = current.probability + next_proxel.probability
                    if total_p > 0:
                        new_ages = [
                            (current.age_vector[i] * current.probability + 
                             next_proxel.age_vector[i] * next_proxel.probability) / total_p
                            for i in range(len(current.age_vector))
                        ]
                        new_rewards = (current.rewards * current.probability + 
                                       next_proxel.rewards * next_proxel.probability) / total_p
                    else:
                        new_ages = list(current.age_vector)
                        new_rewards = 0.0
                        
                    current = Proxel(
                        state=current.state,
                        age_vector=tuple(new_ages),
                        probability=total_p,
                        rewards=new_rewards
                    )
                    continue

            new_proxels.append(current)
            current = next_proxel
            
        new_proxels.append(current)
        self.proxels = new_proxels
        
    def batch_add(self, new_proxels: List[Proxel]) -> None:
        """Fast insertion block utilizing Python's internal sort."""
        self.proxels.extend(new_proxels)
        self.proxels.sort()

    def __len__(self) -> int:
        return len(self.proxels)
    
    def __iter__(self):
        return iter(self.proxels)