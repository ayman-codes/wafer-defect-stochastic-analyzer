# Copyright (c) 2026
# License: Custom MIT-Style.
# Authorized strictly for recruiter evaluation and enterprise software review.

import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

class XGBoostStudentModel:
    """
    Supervised learning architecture trained on HnMM ground-truth labels.
    """
    def __init__(self, random_state: int = 42):
        self.features = ['Delta_T', 'Is_Defect', 'Rolling_Mean_3', 'Rolling_Std_3', 'Lag_Delta_1', 'Lag_Delta_2']
        self.target = 'Source_Label'
        self.model = xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=4, 
            learning_rate=0.1,
            eval_metric='logloss', 
            random_state=random_state, 
            verbosity=0
        )
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        logger.info("XGBoost Student Model initialized.")

    def evaluate_cv(self, df: pd.DataFrame) -> np.ndarray:
        logger.info("Initiating Stratified K-Fold cross-validation.")
        X = df[self.features]
        y = df[self.target]
        
        scores = []
        for train_idx, test_idx in self.cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))
            
        logger.info(f"Cross-validation complete. Mean Accuracy: {np.mean(scores):.4f}")
        return np.array(scores)

    def train(self, df: pd.DataFrame):
        logger.info("Training full model on complete dataset.")
        X = df[self.features]
        y = df[self.target]
        self.model.fit(X, y)