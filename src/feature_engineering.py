"""Feature engineering module."""
import pandas as pd
import numpy as np


def engineer_features(df):
    """Create new features for ML model."""
    # Removal efficiency features
    df['BOD_removal_eff'] = ((df['DBO_E'] - df['DBO_S']) / df['DBO_E'] * 100).clip(0, 100)
    df['COD_removal_eff'] = ((df['DQO_E'] - df['DQO_S']) / df['DQO_E'] * 100).clip(0, 100)
    df['SS_removal_eff'] = ((df['SS_E'] - df['SS_S']) / df['SS_E'] * 100).clip(0, 100)
    df['COND_removal_eff'] = ((df['COND_E'] - df['COND_S']) / df['COND_E'] * 100).clip(0, 100)

    # Ratio features
    df['BOD_COD_ratio'] = df['DBO_E'] / (df['DQO_E'] + 1)
    df['SS_SSV_ratio'] = df['SS_E'] / (df['SSV_E'] + 1)

    # Aggregate features
    df['avg_pH'] = df[['PH_E', 'PH_P', 'PH_D', 'PH_S']].mean(axis=1)
    df['avg_DBO'] = df[['DBO_E', 'DBO_P', 'DBO_D', 'DBO_S']].mean(axis=1)
    df['avg_DQO'] = df[['DQO_E', 'DQO_P', 'DQO_D', 'DQO_S']].mean(axis=1)
    df['avg_COND'] = df[['COND_E', 'COND_P', 'COND_D', 'COND_S']].mean(axis=1)

    # Time-based features
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['DayOfYear'] = df['Date'].dt.dayofyear

    # Season feature
    df['Season'] = df['Month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Summer', 4: 'Summer', 5: 'Summer',
        6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
        10: 'Post-Monsoon', 11: 'Post-Monsoon'
    })

    return df
