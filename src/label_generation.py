"""Label generation module for water quality and risk classification."""
import pandas as pd
import numpy as np


def generate_quality_labels(df):
    """
    Generate water quality labels (GOOD/MODERATE/POOR) based on CPCB standards.
    """
    def calculate_quality_score(row):
        score = 0
        # BOD scoring (25%)
        if row['DBO_S'] < 10:
            score += 25
        elif row['DBO_S'] < 30:
            score += 15
        # COD scoring (25%)
        if row['DQO_S'] < 50:
            score += 25
        elif row['DQO_S'] < 100:
            score += 15
        # pH scoring (20%)
        if 6.5 <= row['PH_S'] <= 8.5:
            score += 20
        elif 6.0 <= row['PH_S'] <= 9.0:
            score += 10
        # Suspended Solids scoring (15%)
        if row['SS_S'] < 50:
            score += 15
        elif row['SS_S'] < 100:
            score += 8
        # Conductivity scoring (15%)
        if row['COND_S'] < 500:
            score += 15
        elif row['COND_S'] < 1500:
            score += 8
        return score

    df['Quality_Score'] = df.apply(calculate_quality_score, axis=1)
    df['Quality_Class'] = pd.cut(df['Quality_Score'],
                                  bins=[-np.inf, 40, 70, np.inf],
                                  labels=['POOR', 'MODERATE', 'GOOD'])
    return df


def generate_risk_labels(df):
    """
    Generate contamination risk labels (LOW/MEDIUM/HIGH).
    """
    def calculate_risk_score(row):
        risk_points = 0
        # BOD risk (0-20)
        if row['DBO_S'] > 30:
            risk_points += 20
        elif row['DBO_S'] > 10:
            risk_points += 10
        # COD risk (0-20)
        if row['DQO_S'] > 100:
            risk_points += 20
        elif row['DQO_S'] > 50:
            risk_points += 10
        # pH risk (0-20)
        if row['PH_S'] < 6.0 or row['PH_S'] > 9.0:
            risk_points += 20
        elif row['PH_S'] < 6.5 or row['PH_S'] > 8.5:
            risk_points += 10
        # SS risk (0-15)
        if row['SS_S'] > 100:
            risk_points += 15
        elif row['SS_S'] > 50:
            risk_points += 8
        # COND risk (0-15)
        if row['COND_S'] > 1500:
            risk_points += 15
        elif row['COND_S'] > 500:
            risk_points += 8
        # Zinc risk (0-10)
        if row['ZN_E'] > 5:
            risk_points += 10
        elif row['ZN_E'] > 3:
            risk_points += 5
        return risk_points

    df['Risk_Score'] = df.apply(calculate_risk_score, axis=1)
    df['Risk_Class'] = pd.cut(df['Risk_Score'],
                               bins=[-np.inf, 15, 35, np.inf],
                               labels=['LOW', 'MEDIUM', 'HIGH'])
    return df
