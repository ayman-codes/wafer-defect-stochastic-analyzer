import logging
from scipy.stats import norm

logger = logging.getLogger(__name__)

class NormalHazard:
    """
    Computes the instantaneous rate function (IRF) for Normal Distributions.
    Logic: $\mu(t) = f(t) / (1 - F(t))$
    """
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma
        
    def get_rate(self, age: float) -> float:
        survival = norm.sf(age, self.mu, self.sigma)
        
        # Safety gate preventing division by zero as CDF approaches 1
        if survival < 1e-7:
            logger.debug(f"Survival {survival} below threshold. Returning infinity.")
            return float('inf')
        
        pdf = norm.pdf(age, self.mu, self.sigma)
        return pdf / survival