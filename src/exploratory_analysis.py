"""Exploratory Data Analysis module."""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports', 'figures')


def perform_eda(df):
    """Perform complete EDA and generate visualizations."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    numeric_df = df.select_dtypes(include=[np.number])
    stats = numeric_df.describe()

    # Distribution plots
    key_params = ['PH_E', 'DBO_E', 'DQO_E', 'SS_E', 'COND_E', 'ZN_E']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, param in enumerate(key_params):
        axes[i].hist(df[param].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[i].set_title(f'{param} Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'parameter_distributions.png'), dpi=150)
    plt.close()

    # Correlation heatmap
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'correlation_heatmap.png'), dpi=150)
    plt.close()

    return stats
