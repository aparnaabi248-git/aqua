"""Contamination risk analysis module."""
import pandas as pd
import numpy as np


def analyze_risk_factors(df):
    """Analyze which parameters contribute most to contamination risk."""
    high_risk = df[df['Risk_Class'] == 'HIGH']

    if len(high_risk) == 0:
        return None

    risk_params = ['DBO_S', 'DQO_S', 'SS_S', 'COND_S', 'PH_S', 'ZN_E']
    high_risk_avg = high_risk[risk_params].mean()
    overall_avg = df[risk_params].mean()

    comparison = pd.DataFrame({
        'High Risk Average': high_risk_avg,
        'Overall Average': overall_avg
    })
    comparison['Difference'] = comparison['High Risk Average'] - comparison['Overall Average']
    comparison['Ratio'] = comparison['High Risk Average'] / comparison['Overall Average']

    return comparison


def get_risk_summary(df):
    """Get summary statistics for risk levels."""
    summary = {}
    for risk_level in ['LOW', 'MEDIUM', 'HIGH']:
        subset = df[df['Risk_Class'] == risk_level]
        summary[risk_level] = {
            'count': len(subset),
            'percentage': len(subset) / len(df) * 100,
            'avg_quality_score': subset['Quality_Score'].mean() if len(subset) > 0 else 0,
        }
    return summary
