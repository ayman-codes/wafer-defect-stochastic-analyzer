# Stochastic Defect Analyzer: Wafer Production Line

## Architecture Overview
This system provides mathematical and simulative reconstruction of a wafer production line to isolate hardware failures. The architecture models two upstream material flows merging into a single quality tester. The tester protocols high-frequency timestamps and binary states (Ok/Defect) but lacks source telemetry. 


![system_description](./system_description.png)

The engine processes this incomplete log to statistically identify the upstream machine responsible for an anomalous defect rate.

## Engine Progression
The analysis pipeline iterates through six stochastic and probabilistic frameworks to achieve maximum inference fidelity:
1. **01_dtmc_baseline**: Discrete-Time Markov Chains for foundational state transitioning.
2. **02_ctmc_kinetics**: Continuous-Time Markov Chains for time-dependent failure rates.
3. **03_gspn_concurrency**: Generalized Stochastic Petri Nets for asynchronous material flow modeling.
4. **04_proxel_state_exploration**: Probability Elements for continuous-time, discrete-event simulation.
5. **05_hmm_inference**: Hidden Markov Models to map observable test outputs to hidden machine states.
6. **06_hnmm_advanced_inference**: Hidden non-Markovian Models incorporating arbitrary distribution times.

## Tech Stack
* Python 3.12+
* NumPy, SciPy (Iterative Power Methods, Linear Algebra Solvers)
* Tkinter (Asynchronous Monte Carlo Visualizations)