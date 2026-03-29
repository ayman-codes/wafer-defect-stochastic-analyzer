# Copyright (c) 2026
# License: Custom MIT-Style.
# Authorized strictly for recruiter evaluation and enterprise software review.

import logging
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

import numpy as np
import scipy.stats as stats
import pandas as pd
from scipy.special import logsumexp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)

@dataclass
class Proxel:
    """
    Core probability element for state-space traversal.
    Maintains discrete state, continuous memory vectors, and log-probability mass.
    """
    state: str
    age_0: float
    age_1: float
    log_probability: float
    path: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (f"Proxel(State={self.state}, Age0={self.age_0:.2f}, "
                f"Age1={self.age_1:.2f}, P={self.log_probability:.4e})")


class ProxelSimulation:
    """
    Continuous-Time Non-Markovian solver utilizing log-space Probability Elements.
    Implements Race Age (RA) policies and multi-strategy state space bounding.
    """
    def __init__(self, dt: float, pruning_strategy: str = "standard", beam_width: int = 100):
        self.dt = dt
        self.pruning_strategy = pruning_strategy
        self.beam_width = beam_width
        self.current_time = 0.0
        self.proxels: List[Proxel] = []
        
        self.emissions = {
            'Source0': {'OK': 0.95, 'D': 0.05}, 
            'Source1': {'OK': 0.95, 'D': 0.05}
        }
        
        self.MIN_PROB_LOG = np.log(1e-4)
        
        self.mu1, self.sig1 = 120.0, 20.0
        self.mu0, self.sig0 = 150.0, 25.0
        
        self.proxels.append(Proxel("Wait", 0.0, 0.0, 0.0))

    def hr(self, age: float, mu: float, sigma: float) -> float:
        """Computes instantaneous hazard rate h(t) = f(t) / S(t)."""
        sf = stats.norm.sf(age, loc=mu, scale=sigma)
        if sf < 1e-15:
            return 1.0  # Limit condition bounded to prevent overflow
        return stats.norm.pdf(age, loc=mu, scale=sigma) / sf

    def hr_trapezoid(self, age: float, dt: float, mu: float, sigma: float) -> float:
        """Computes continuous hazard rate interval via Trapezoidal integration."""
        h_start = self.hr(age, mu, sigma)
        h_end = self.hr(age + dt, mu, sigma)
        return (h_start + h_end) / 2.0

    def get_log_physics(self, age: float, mu: float, sigma: float) -> tuple[float, float]:
        """Returns stable log(PDF) and log(Survival Function)."""
        log_pdf = stats.norm.logpdf(age, loc=mu, scale=sigma)
        log_sf = stats.norm.logsf(age, loc=mu, scale=sigma)
        return log_pdf, log_sf

    def merge(self):
        """State compression via dt-proximity grouping and log-space summation."""
        if not self.proxels:
            return

        self.proxels.sort(key=lambda p: (p.age_0, p.age_1))
        merged_proxels = []
        current = self.proxels[0]
        
        for next_p in self.proxels[1:]:
            diff_0 = abs(current.age_0 - next_p.age_0)
            diff_1 = abs(current.age_1 - next_p.age_1)
            
            if diff_0 < self.dt and diff_1 < self.dt:
                new_log_prob = logsumexp([current.log_probability, next_p.log_probability])
                
                w_curr = np.exp(current.log_probability - new_log_prob)
                w_next = np.exp(next_p.log_probability - new_log_prob)
                
                new_age_0 = (current.age_0 * w_curr) + (next_p.age_0 * w_next)
                new_age_1 = (current.age_1 * w_curr) + (next_p.age_1 * w_next)
                
                best_path = current.path if current.log_probability >= next_p.log_probability else next_p.path
                current = Proxel(current.state, new_age_0, new_age_1, new_log_prob, best_path)
            else:
                merged_proxels.append(current)
                current = next_p
        
        merged_proxels.append(current)
        self.proxels = merged_proxels

    def step(self, step_index: int = 0):
        """Advances state matrix by fixed dt interval."""
        next_proxels = []
        
        for p in self.proxels:
            raw_p0 = self.hr_trapezoid(p.age_0, self.dt, self.mu0, self.sig0) * self.dt
            raw_p1 = self.hr_trapezoid(p.age_1, self.dt, self.mu1, self.sig1) * self.dt
            
            prob_fire_0 = min(0.999999, raw_p0)
            prob_fire_1 = min(0.999999, raw_p1)
            
            prob_stay = (1.0 - prob_fire_0) * (1.0 - prob_fire_1)
            if prob_stay > 0:
                next_proxels.append(Proxel(
                    "Wait", p.age_0 + self.dt, p.age_1 + self.dt, 
                    p.log_probability + np.log(prob_stay), p.path
                ))
            
            term_s0 = prob_fire_0 * (1.0 - prob_fire_1)
            if term_s0 > 0:
                next_proxels.append(Proxel(
                    "Wait", 0.0, p.age_1 + self.dt, 
                    p.log_probability + np.log(term_s0), p.path + ["Source0"]
                ))
                
            term_s1 = prob_fire_1 * (1.0 - prob_fire_0)
            if term_s1 > 0:
                next_proxels.append(Proxel(
                    "Wait", p.age_0 + self.dt, 0.0, 
                    p.log_probability + np.log(term_s1), p.path + ["Source1"]
                ))
            
            term_both = prob_fire_0 * prob_fire_1
            if term_both > 0:
                next_proxels.append(Proxel(
                    "Wait", 0.0, 0.0,
                    p.log_probability + np.log(term_both), p.path + ["Both"]
                ))

        self.proxels = next_proxels
        self.current_time += self.dt
        self.prune(step_index)

    def prune(self, step_index: int):
        """Dispatches bounded state space truncation strategy."""
        if self.pruning_strategy == "standard":
            self.prune_standard()
        elif self.pruning_strategy == "beam":
            self.prune_beam()
        elif self.pruning_strategy == "lookahead":
            self.prune_lookahead(step_index)
        else:
            logging.critical(f"Invalid bounding configuration: {self.pruning_strategy}")
            raise ValueError("Configuration Exception: Unknown Strategy")

    def prune_standard(self):
        """Truncates vector states falling below absolute probability threshold."""
        self.proxels = [p for p in self.proxels if p.log_probability >= self.MIN_PROB_LOG]

    def prune_beam(self):
        """Truncates state matrix to fixed integer k width."""
        if len(self.proxels) <= self.beam_width:
            return
        self.proxels.sort(key=lambda p: p.log_probability, reverse=True) 
        self.proxels = self.proxels[:self.beam_width]
        
    def compute_lookahead_heuristic(self, trace: List[str]):
        """Generates maximum likelihood potential vector for trace completion."""
        self.heuristic_vector = [0.0] * (len(trace) + 1)
        cumulative_log_prob = 0.0
        
        for i in range(len(trace) - 1, -1, -1):
            symbol = trace[i]
            p_s0 = self.emissions["Source0"].get(symbol, 0.0)
            p_s1 = self.emissions["Source1"].get(symbol, 0.0)
            max_p = max(p_s0, p_s1)
            
            if max_p > 0:
                cumulative_log_prob += np.log(max_p)
            else:
                cumulative_log_prob += -1e9
            
            self.heuristic_vector[i] = cumulative_log_prob

    def prune_lookahead(self, step_index: int):
        """Truncates paths failing relative heuristic threshold logic."""
        if not hasattr(self, 'heuristic_vector') or not self.heuristic_vector:
            logging.error("Heuristic vector uninitialized prior to state evaluation.")
            return

        h_val_log = self.heuristic_vector[step_index] if step_index < len(self.heuristic_vector) else 0.0
    
        if not self.proxels:
            return

        max_current_log_prob = max(p.log_probability for p in self.proxels)
        best_log_potential = max_current_log_prob + h_val_log
        log_cutoff = best_log_potential + self.MIN_PROB_LOG

        original_count = len(self.proxels)
        
        self.proxels = [
            p for p in self.proxels 
            if (p.log_probability + h_val_log) >= log_cutoff
        ]
        
        if not self.proxels and original_count > 0:
            logging.critical(f"Heuristic bounding triggered total state collapse at index {step_index}.")
        
    def get_cache_filepath(self, trace_name: str) -> Path:
        """Resolves target file path for execution state persistence."""
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)

        base_name = f"results_{trace_name}_{self.pruning_strategy}_dt{self.dt}"
        
        if self.pruning_strategy == "beam":
            base_name += f"_k{self.beam_width}"
        elif self.pruning_strategy == "standard":
            base_name += f"_thresh{np.exp(self.MIN_PROB_LOG):.0e}"
            
        return cache_dir / f"{base_name}.csv"
            
    def memorize(self, trace_df: pd.DataFrame, trace_name: str = "default_trace", log_interval: int = 10, return_stats: bool = False) -> pd.DataFrame:
        """
        Executes event-driven $\Delta T$ inference with file-based memoization.
        """
        cache_path = self.get_cache_filepath(trace_name)
        
        if cache_path.exists():
            logging.info(f"Targeting persistent state file: {cache_path}")
            try:
                cached_data = pd.read_csv(cache_path)
                if len(cached_data) >= len(trace_df):
                    return cached_data
                logging.warning("Persistent state truncated. Forcing execution reset.")
            except Exception as e:
                logging.error(f"Persistence evaluation exception: {e}. Executing runtime sequence.")

        logging.info(f"Initiating event-driven inference loop: {trace_name}")
        
        if self.pruning_strategy == "lookahead":
            trace_symbols = trace_df['Result'].tolist() if 'Result' in trace_df.columns else []
            self.compute_lookahead_heuristic(trace_symbols)

        results = []
        self.proxels = [Proxel("Wait", 0.0, 0.0, 0.0)] 
        self.current_time = trace_df.iloc[0]['Time'] if not trace_df.empty else 0.0
        prev_time = self.current_time

        for i, row in trace_df.iterrows():
            current_time = row['Time']
            observed_symbol = row['Result']
            
            step_dt = current_time - prev_time
            if i == 0 and 'Delta_T' in trace_df.columns: 
                step_dt = row['Delta_T']
            if step_dt <= 0: 
                step_dt = 1e-6
            self.dt = step_dt 
            
            next_proxels = []
            p_emit_s0 = self.emissions["Source0"][observed_symbol]
            p_emit_s1 = self.emissions["Source1"][observed_symbol]
            log_emit_s0 = np.log(p_emit_s0)
            log_emit_s1 = np.log(p_emit_s1)

            for p in self.proxels:
                aged_0, aged_1 = p.age_0 + step_dt, p.age_1 + step_dt
                log_f0, log_s0 = self.get_log_physics(aged_0, self.mu0, self.sig0)
                log_f1, log_s1 = self.get_log_physics(aged_1, self.mu1, self.sig1)
                
                log_lik_s0 = log_f0 + log_s1 + log_emit_s0
                next_proxels.append(Proxel("Wait", 0.0, aged_1, p.log_probability + log_lik_s0, p.path + ["Source0"]))

                log_lik_s1 = log_f1 + log_s0 + log_emit_s1
                next_proxels.append(Proxel("Wait", aged_0, 0.0, p.log_probability + log_lik_s1, p.path + ["Source1"]))

            self.proxels = next_proxels
            prev_time = current_time

            if not self.proxels:
                logging.critical(f"State space collapse detected at step index {i}.")
                break
            
            current_log_probs = [p.log_probability for p in self.proxels]
            total_log_mass = logsumexp(current_log_probs)
            for p in self.proxels: 
                p.log_probability -= total_log_mass
                
            self.merge()
            self.prune(i)

            if return_stats:
                results.append({
                    "Step": i,
                    "Alive_Proxels": len(self.proxels),
                    "Total_Mass": 1.0 
                })

            if log_interval > 0 and (i % log_interval == 0):
                logging.info(f"[Idx: {i:03}] Bounding: {self.pruning_strategy:<10} | Matrix Nodes: {len(self.proxels):03}")

        if return_stats:
            stats_df = pd.DataFrame(results)
            stats_df.to_csv(cache_path, index=False)
            return stats_df

        if self.proxels:
            self.proxels.sort(key=lambda p: p.log_probability, reverse=True)
            winner = self.proxels[0]
            path_labels = winner.path
            if len(path_labels) != len(trace_df):
                 min_len = min(len(path_labels), len(trace_df))
                 path_labels = path_labels[:min_len]
                 trace_df = trace_df.iloc[:min_len]
            trace_df['Source_Label'] = path_labels
            trace_df.to_csv(cache_path, index=False)
            
        return trace_df