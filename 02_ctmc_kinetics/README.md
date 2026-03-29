# Phase 2: CTMC Kinetics Modeling

## Objective
Expand the stochastic engine to support Continuous-Time Markov Chains (CTMC). This module evaluates state transitions governed by continuous failure rates, utilizing uniformization (discretization) to convert generator matrices into transition probability matrices. Numerical convergence is verified against exact analytical matrix exponentials.

## System Parameters
* **State Space:** $S = \{S_0, S_1\}$
* **Initial State:** $\pi_0 = [1.0, 0.0]$
* **Average Durations:** Source 0 = 2 min, Source 1 = 3 min.
* **Transition Rates ($\lambda$):** $\lambda_{0,1} = 0.5$, $\lambda_{1,0} \approx 0.3333$
* **Generator Matrix ($Q$):**
  $$Q = \begin{bmatrix} -0.5 & 0.5 \\ 0.3333 & -0.3333 \end{bmatrix}$$
* **Emission Probabilities:** Target operational yields are 0.9 for source 0 and 0.95 for source 1.

## Execution Logic
* **`ctmc.py`**: Mathematical engine. Validates infinitesimal generator matrices, calculates analytical continuous-time distributions via `scipy.linalg.expm`, and executes uniformization to bridge CTMC logic into the foundational DTMC solver.
* **`main.py`**: Executes discretization across varied time deltas ($\Delta t \in \{2, 1, 0.5, 0.25, 0.1\}$) to prove the stability limits and numerical convergence at $t=8$ minutes.
* **`visualizer.py`**: Topographical rate-mapping of the CTMC utilizing `networkx`.