# Copyright (c) 2026
# License: Custom MIT-Style.
# Authorized strictly for recruiter evaluation and enterprise software review.


import sys
import os
import glob
from pathlib import Path
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from proxel import ProxelSimulation
from features import FeatureEngineer
from classifier import XGBoostStudentModel

def generate_ground_truth(data_dir: str, output_path: str) -> pd.DataFrame:
    print(f"Executing batch HnMM solver on {data_dir}...")
    search_path = os.path.join(data_dir, "protocol*.txt")
    files = sorted(glob.glob(search_path))
    
    if not files:
        print("Exception: No protocol files located.")
        sys.exit(1)

    all_labeled_dfs = []
    for trace_path in files:
        filename = Path(trace_path).stem
        print(f"Processing trace: {filename}")
        
        df_trace = pd.read_csv(trace_path, sep=" ", header=None, names=["Time", "Result"])
        sim = ProxelSimulation(dt=1.0, pruning_strategy="beam", beam_width=100)
        
        labeled_df = sim.memorize(df_trace, trace_name=f"{filename}_log_v1", log_interval=0)
        
        if 'Source_Label' in labeled_df.columns:
            labeled_df['Original_File'] = filename
            all_labeled_dfs.append(labeled_df)
        else:
            print(f"Warning: Label generation failed for {filename}")

    if all_labeled_dfs:
        final_df = pd.concat(all_labeled_dfs, ignore_index=True)
        final_df['Source_Label'] = final_df['Source_Label'].map({'Source0': 0, 'Source1': 1})
        final_df.to_csv(output_path, index=False)
        print(f"Ground truth generation complete. Total vectors: {len(final_df)}")
        return final_df
    return None

def execute_pipeline():
    data_dir = "data"
    ground_truth_path = os.path.join(data_dir, "labeled_dataset_combined.csv")
    features_path = os.path.join(data_dir, "labeled_dataset_with_features.csv")
    
    os.makedirs(data_dir, exist_ok=True)
    
    if not os.path.exists(ground_truth_path):
        df = generate_ground_truth(data_dir, ground_truth_path)
    else:
        print(f"Loading existing ground truth: {ground_truth_path}")
        df = pd.read_csv(ground_truth_path)
        
    print("Executing feature engineering...")
    df_features = FeatureEngineer.process_batch(df)
    df_features.to_csv(features_path, index=False)
    
    print("Executing XGBoost Student Model cross-validation...")
    ml_system = XGBoostStudentModel()
    scores = ml_system.evaluate_cv(df_features)
    
    print("\n=== Pipeline Execution Results ===")
    for i, score in enumerate(scores):
        print(f"Fold {i+1} Accuracy: {score:.4f}")
    print(f"Mean Accuracy:    {scores.mean():.4f}")
    print(f"Standard Dev:     {scores.std():.4f}")

if __name__ == "__main__":
    execute_pipeline()