# Copyright (c) 2026
# License: Custom MIT-Style.
# Authorized strictly for recruiter evaluation and enterprise software review.


import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from features import FeatureEngineer

@pytest.fixture
def deterministic_trace_matrix():
    """Generates immutable state trace for feature pipeline validation."""
    return pd.DataFrame({
        'Time': [100.0, 220.0, 370.0, 490.0, 640.0],
        'Result': ['OK', 'OK', 'D', 'OK', 'D'],
        'Original_File': ['protocol_test'] * 5
    })

def test_feature_matrix_dimensionality(deterministic_trace_matrix):
    """Asserts structural integrity and column expansion of the feature matrix."""
    initial_cols = len(deterministic_trace_matrix.columns)
    expected_engineered_cols = 7 
    
    df_features = FeatureEngineer.process_batch(deterministic_trace_matrix)
    
    assert len(df_features.columns) == initial_cols + expected_engineered_cols
    
    required_vectors = [
        'Delta_T', 'Rolling_Mean_3', 'Rolling_Std_3', 
        'Lag_Delta_1', 'Lag_Delta_2', 'Is_Defect', 'Lag_Defect_1'
    ]
    for vec in required_vectors:
        assert vec in df_features.columns

def test_markovian_lag_accuracy(deterministic_trace_matrix):
    """Validates deterministic mathematical execution of time-series lags and discrete mappings."""
    df_features = FeatureEngineer.process_batch(deterministic_trace_matrix)
    
    expected_delta_t = [100.0, 120.0, 150.0, 120.0, 150.0]
    np.testing.assert_array_almost_equal(df_features['Delta_T'].values, expected_delta_t)
    
    expected_is_defect = [0, 0, 1, 0, 1]
    np.testing.assert_array_equal(df_features['Is_Defect'].values, expected_is_defect)
    
    expected_lag_1 = [0.0, 100.0, 120.0, 150.0, 120.0]
    np.testing.assert_array_almost_equal(df_features['Lag_Delta_1'].values, expected_lag_1)

    expected_lag_defect = [0, 0, 0, 1, 0]
    np.testing.assert_array_equal(df_features['Lag_Defect_1'].values, expected_lag_defect)

def test_rolling_statistical_moments(deterministic_trace_matrix):
    """Validates rolling context window calculations."""
    df_features = FeatureEngineer.process_batch(deterministic_trace_matrix)
    
    expected_rolling_mean = [100.0, 110.0, 123.333333, 130.0, 140.0]
    np.testing.assert_array_almost_equal(df_features['Rolling_Mean_3'].values, expected_rolling_mean, decimal=5)
    
    expected_rolling_std = [0.0, 14.142135, 25.166114, 17.320508, 17.320508]
    np.testing.assert_array_almost_equal(df_features['Rolling_Std_3'].values, expected_rolling_std, decimal=5)

def test_missing_partition_key():
    """Asserts fallback execution path when batch partition keys are absent."""
    df_no_key = pd.DataFrame({
        'Time': [100.0, 200.0],
        'Result': ['OK', 'OK']
    })
    
    df_features = FeatureEngineer.process_batch(df_no_key)
    assert 'Delta_T' in df_features.columns
    assert len(df_features) == 2