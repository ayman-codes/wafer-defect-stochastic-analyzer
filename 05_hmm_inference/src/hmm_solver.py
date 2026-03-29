import numpy as np
import logging

logger = logging.getLogger(__name__)

class HMMSolver:
    """Enterprise HMM Inference Engine for Decoding and Evaluation."""

    def __init__(self, states: list, transition_matrix: list, emission_probs: dict):
        self.states = states
        self.P = transition_matrix
        self.emissions = emission_probs
        self.num_states = len(states)
        logger.info("Initialized HMMSolver for %d-state system.", self.num_states)

    def _get_emission_prob(self, state_idx: int, observation: str) -> float:
        if state_idx not in self.emissions:
            return 0.0
        return self.emissions[state_idx].get(observation, 0.0)

    def solve_forward(self, observations: list, initial_pi: np.ndarray) -> float:
        """Computes marginal probability of observation sequence via Forward Algorithm."""
        logger.debug("Executing Forward algorithm for sequence length %d", len(observations))
        N = self.num_states
        T = len(observations)
        alpha = np.zeros(N)
        first_obs = observations[0]

        for s in range(N):
            alpha[s] = initial_pi[s] * self._get_emission_prob(s, first_obs)

        for t in range(1, T):
            obs = observations[t]
            new_alpha = np.zeros(N)
            for s_curr in range(N):
                sum_prev = sum(alpha[s_prev] * self.P[s_prev][s_curr] for s_prev in range(N))
                new_alpha[s_curr] = sum_prev * self._get_emission_prob(s_curr, obs)
            alpha = new_alpha

        return float(np.sum(alpha))

    def solve_viterbi(self, observations: list, initial_pi: np.ndarray) -> tuple[list[int], float]:
        """Computes most probable hidden state sequence via Viterbi Algorithm."""
        logger.debug("Executing Viterbi algorithm for sequence length %d", len(observations))
        N = self.num_states
        T = len(observations)
        viterbi = np.zeros(N)
        paths = [[s] for s in range(N)]
        first_obs = observations[0]

        for s in range(N):
            viterbi[s] = initial_pi[s] * self._get_emission_prob(s, first_obs)

        for t in range(1, T):
            obs = observations[t]
            new_viterbi = np.zeros(N)
            new_paths = [[] for _ in range(N)]

            for s_curr in range(N):
                best_prob = -1.0
                best_prev_s = -1

                for s_prev in range(N):
                    prob = viterbi[s_prev] * self.P[s_prev][s_curr]
                    if prob > best_prob:
                        best_prob = prob
                        best_prev_s = s_prev

                new_viterbi[s_curr] = best_prob * self._get_emission_prob(s_curr, obs)
                if best_prev_s != -1:
                    new_paths[s_curr] = paths[best_prev_s] + [s_curr]

            viterbi = new_viterbi
            paths = new_paths

        best_end_state = int(np.argmax(viterbi))
        max_prob = float(viterbi[best_end_state])
        return paths[best_end_state], max_prob

    def solve_beam_search(self, observations: list, initial_pi: np.ndarray, k: int = 2) -> list[dict]:
        """Prunes low-probability sequence paths maintaining top K beams."""
        logger.debug("Executing Beam Search with beam width K=%d", k)
        N = self.num_states
        T = len(observations)
        current_beams = []
        first_obs = observations[0]

        for s in range(N):
            prob = initial_pi[s] * self._get_emission_prob(s, first_obs)
            if prob > 0:
                current_beams.append({'path': [s], 'prob': prob, 'last_idx': s})

        current_beams = sorted(current_beams, key=lambda x: x['prob'], reverse=True)[:k]

        for t in range(1, T):
            obs = observations[t]
            next_beams = []

            for beam in current_beams:
                prev_s = beam['last_idx']
                prev_prob = beam['prob']
                prev_path = beam['path']

                for next_s in range(N):
                    trans_prob = self.P[prev_s][next_s]
                    emit_prob = self._get_emission_prob(next_s, obs)
                    new_prob = prev_prob * trans_prob * emit_prob

                    next_beams.append({
                        'path': prev_path + [next_s],
                        'prob': new_prob,
                        'last_idx': next_s
                    })

            current_beams = sorted(next_beams, key=lambda x: x['prob'], reverse=True)[:k]

        return current_beams