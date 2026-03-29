# Copyright (c) 2026
# License: Custom MIT-Style.
# Authorized strictly for recruiter evaluation and enterprise software review.

import logging
import pandas as pd

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Constructs time-series features including Markovian lags and rolling statistical moments.
    """
    @staticmethod
    def build_features(df_group: pd.DataFrame) -> pd.DataFrame:
        df_group = df_group.sort_values('Time').copy()
        
        df_group['Delta_T'] = df_group['Time'].diff().fillna(df_group['Time'].iloc[0])
        
        df_group['Rolling_Mean_3'] = df_group['Delta_T'].rolling(window=3, min_periods=1).mean()
        df_group['Rolling_Std_3'] = df_group['Delta_T'].rolling(window=3, min_periods=1).std().fillna(0.0)
        
        df_group['Lag_Delta_1'] = df_group['Delta_T'].shift(1).fillna(0.0)
        df_group['Lag_Delta_2'] = df_group['Delta_T'].shift(2).fillna(0.0)
        
        df_group['Is_Defect'] = (df_group['Result'] == 'D').astype(int)
        df_group['Lag_Defect_1'] = df_group['Is_Defect'].shift(1).fillna(0)
        
        return df_group

    @classmethod
    def process_batch(cls, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Executing feature engineering pipeline across batch partitions.")
        if 'Original_File' not in df.columns:
            logger.warning("Original_File partition key missing. Applying features globally.")
            return cls.build_features(df)
            
        # Pandas >= 2.2.0 compatibility: explicit partition reconstruction
        results = []
        for file_key, group in df.groupby('Original_File'):
            processed_group = cls.build_features(group)
            processed_group['Original_File'] = file_key
            results.append(processed_group)
            
        return pd.concat(results, ignore_index=True)