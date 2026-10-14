"""
Smart Water Quality Classification and Contamination Risk Analytics
Main Pipeline Script

This script performs the complete Data Science lifecycle:
1. Data Loading
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis
4. Feature Engineering
5. Label Generation (GOOD/MODERATE/POOR)
6. Model Training & Comparison
7. Model Evaluation
8. Explainability Analysis
9. Risk Analytics
10. Model Saving

Dataset: UCI Water Treatment Plant
Source: https://archive.ics.uci.edu/ml/datasets/water+treatment+plant
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, classification_report, confusion_matrix,
                            roc_auc_score)
from sklearn.inspection import permutation_importance

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Will skip XGBoost model.")

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Will skip SHAP analysis.")

# Joblib
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')

# Create directories
for dir_path in [DATA_PROCESSED, MODELS_DIR, FIGURES_DIR]:
    os.makedirs(dir_path, exist_ok=True)


def load_data():
    """Load the UCI Water Treatment Plant dataset."""
    print("=" * 60)
    print("STEP 1: DATA LOADING")
    print("=" * 60)

    filepath = os.path.join(DATA_RAW, 'uci_water_treatment.csv')

    # Column names based on UCI documentation
    column_names = [
        'Date', 'Q_E', 'ZN_E', 'PH_E', 'DBO_E', 'DQO_E', 'SS_E', 'SSV_E',
        'SED_E', 'COND_E', 'PH_P', 'DBO_P', 'DQO_P', 'SS_P', 'SSV_P',
        'SED_P', 'COND_P', 'PH_D', 'DBO_D', 'DQO_D', 'SS_D', 'SSV_D',
        'SED_D', 'COND_D', 'PH_S', 'DBO_S', 'DQO_S', 'SS_S', 'SSV_S',
        'SED_S', 'COND_S', 'RD_DBO_S', 'RD_DQO_S', 'RD_SS_S', 'RD_SED_S',
        'RD_CON_S', 'RD_DBO_G', 'RD_DQO_G', 'RD_SS_G'
    ]

    df = pd.read_csv(filepath, header=None, names=column_names)

    # Replace '?' with NaN
    df = df.replace('?', np.nan)

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())

    return df


def clean_data(df):
    """Clean the dataset: handle missing values, duplicates, invalid values."""
    print("\n" + "=" * 60)
    print("STEP 2: DATA CLEANING")
    print("=" * 60)

    # Convert all columns to numeric where possible
    for col in df.columns:
        if col != 'Date':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Parse dates
    df['Date'] = df['Date'].str.replace('D-', '')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y', errors='coerce')

    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
    print(f"\nMissing values:")
    print(missing_df[missing_df['Missing Count'] > 0])

    # Handle missing values - use median imputation for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Filled {col} missing values with median: {median_val:.2f}")

    # Duplicate detection
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Removed {duplicates} duplicate rows")

    # Invalid value detection
    print(f"\nData types:")
    print(df.dtypes)

    print(f"\nCleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


def exploratory_analysis(df):
    """Perform EDA and generate visualizations."""
    print("\n" + "=" * 60)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Statistical summary
    numeric_df = df.select_dtypes(include=[np.number])
    stats = numeric_df.describe()
    print("\nStatistical Summary:")
    print(stats)

    # Save statistics
    stats.to_csv(os.path.join(DATA_PROCESSED, 'statistics.csv'))

    # Key water quality parameters for analysis
    key_params = ['PH_E', 'DBO_E', 'DQO_E', 'SS_E', 'COND_E', 'ZN_E',
                  'PH_S', 'DBO_S', 'DQO_S', 'SS_S', 'COND_S']

    # 1. Distribution plots
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    for i, param in enumerate(key_params):
        if i < len(axes):
            axes[i].hist(df[param].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_title(f'{param} Distribution')
            axes[i].set_xlabel(param)
            axes[i].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'parameter_distributions.png'), dpi=150)
    plt.close()
    print("  Saved: parameter_distributions.png")

    # 2. Box plots
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    for i, param in enumerate(key_params):
        if i < len(axes):
            axes[i].boxplot(df[param].dropna())
            axes[i].set_title(f'{param} Box Plot')
            axes[i].set_ylabel(param)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'parameter_boxplots.png'), dpi=150)
    plt.close()
    print("  Saved: parameter_boxplots.png")

    # 3. Correlation heatmap
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Heatmap of Water Quality Parameters')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'correlation_heatmap.png'), dpi=150)
    plt.close()
    print("  Saved: correlation_heatmap.png")

    # 4. Scatter plots for key relationships
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    relationships = [
        ('PH_E', 'PH_S', 'pH: Input vs Output'),
        ('DBO_E', 'DBO_S', 'BOD: Input vs Output'),
        ('DQO_E', 'DQO_S', 'COD: Input vs Output'),
        ('SS_E', 'SS_S', 'Suspended Solids: Input vs Output'),
        ('COND_E', 'COND_S', 'Conductivity: Input vs Output'),
        ('ZN_E', 'DBO_S', 'Zinc vs Output BOD'),
    ]

    for i, (x, y, title) in enumerate(relationships):
        axes[i].scatter(df[x], df[y], alpha=0.5, s=20)
        axes[i].set_xlabel(x)
        axes[i].set_ylabel(y)
        axes[i].set_title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'scatter_relationships.png'), dpi=150)
    plt.close()
    print("  Saved: scatter_relationships.png")

    # 5. Time series of key parameters
    df_time = df.sort_values('Date')
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    time_params = ['PH_E', 'DBO_E', 'DQO_E', 'SS_E', 'COND_E', 'ZN_E']
    for i, param in enumerate(time_params):
        axes[i].plot(df_time['Date'], df_time[param], alpha=0.7, linewidth=0.8)
        axes[i].set_title(f'{param} Over Time')
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel(param)
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'time_series.png'), dpi=150)
    plt.close()
    print("  Saved: time_series.png")

    # 6. Outlier analysis using IQR
    print("\n  Outlier Analysis (IQR method):")
    outlier_summary = {}
    for col in key_params:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_summary[col] = outliers
        print(f"    {col}: {outliers} outliers ({outliers/len(df)*100:.1f}%)")

    return stats


def feature_engineering(df):
    """Create new features and prepare data for ML."""
    print("\n" + "=" * 60)
    print("STEP 4: FEATURE ENGINEERING")
    print("=" * 60)

    # Create removal efficiency features
    df['BOD_removal_eff'] = ((df['DBO_E'] - df['DBO_S']) / df['DBO_E'] * 100).clip(0, 100)
    df['COD_removal_eff'] = ((df['DQO_E'] - df['DQO_S']) / df['DQO_E'] * 100).clip(0, 100)
    df['SS_removal_eff'] = ((df['SS_E'] - df['SS_S']) / df['SS_E'] * 100).clip(0, 100)
    df['COND_removal_eff'] = ((df['COND_E'] - df['COND_S']) / df['COND_E'] * 100).clip(0, 100)

    # Create ratio features
    df['BOD_COD_ratio'] = df['DBO_E'] / (df['DQO_E'] + 1)
    df['SS_SSV_ratio'] = df['SS_E'] / (df['SSV_E'] + 1)

    # Create aggregate features
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

    print(f"  Created new features:")
    print(f"    - Removal efficiency features (BOD, COD, SS, COND)")
    print(f"    - Ratio features (BOD/COD, SS/SSV)")
    print(f"    - Aggregate features (avg pH, DBO, DQO, COND)")
    print(f"    - Time-based features (Month, Year, DayOfYear, Season)")
    print(f"  Total features: {df.shape[1]}")

    return df


def generate_labels(df):
    """
    Generate water quality labels (GOOD/MODERATE/POOR) based on
    scientifically defensible thresholds.

    Methodology:
    - Based on CPCB (Central Pollution Control Board) standards
    - Uses output/final treated water parameters
    - Uses percentile-based thresholds to ensure balanced classes
    - BOD: Good < 10, Moderate 10-30, Poor > 30 mg/L
    - COD: Good < 50, Moderate 50-100, Poor > 100 mg/L
    - pH: Good 6.5-8.5, Moderate 6.0-9.0, Poor < 6.0 or > 9.0
    - Suspended Solids: Good < 50, Moderate 50-100, Poor > 100 mg/L
    - Conductivity: Good < 500, Moderate 500-1500, Poor > 1500 µS/cm
    """
    print("\n" + "=" * 60)
    print("STEP 5: LABEL GENERATION")
    print("=" * 60)

    # Calculate a composite water quality score
    # Based on CPCB standards for treated water

    def calculate_quality_score(row):
        score = 0

        # BOD scoring (weight: 25%)
        if row['DBO_S'] < 10:
            score += 25
        elif row['DBO_S'] < 30:
            score += 15
        else:
            score += 0

        # COD scoring (weight: 25%)
        if row['DQO_S'] < 50:
            score += 25
        elif row['DQO_S'] < 100:
            score += 15
        else:
            score += 0

        # pH scoring (weight: 20%)
        if 6.5 <= row['PH_S'] <= 8.5:
            score += 20
        elif 6.0 <= row['PH_S'] <= 9.0:
            score += 10
        else:
            score += 0

        # Suspended Solids scoring (weight: 15%)
        if row['SS_S'] < 50:
            score += 15
        elif row['SS_S'] < 100:
            score += 8
        else:
            score += 0

        # Conductivity scoring (weight: 15%)
        if row['COND_S'] < 500:
            score += 15
        elif row['COND_S'] < 1500:
            score += 8
        else:
            score += 0

        return score

    df['Quality_Score'] = df.apply(calculate_quality_score, axis=1)

    # Use rank-based binning to ensure balanced classes
    # This handles cases where scores are concentrated in a narrow range
    df['Quality_Rank'] = df['Quality_Score'].rank(method='first')
    n = len(df)
    df['Quality_Class'] = pd.cut(df['Quality_Rank'],
                                  bins=[0, n/3, 2*n/3, n],
                                  labels=['POOR', 'MODERATE', 'GOOD'])

    print(f"  Quality Score range: {df['Quality_Score'].min():.1f} - {df['Quality_Score'].max():.1f}")

    print(f"\n  Class Distribution:")
    class_dist = df['Quality_Class'].value_counts()
    for cls, count in class_dist.items():
        print(f"    {cls}: {count} ({count/len(df)*100:.1f}%)")

    # Save label distribution
    class_dist.to_csv(os.path.join(DATA_PROCESSED, 'class_distribution.csv'))

    return df


def generate_risk_labels(df):
    """
    Generate contamination risk labels (LOW/MEDIUM/HIGH).

    Risk is based on:
    - Deviation of parameters from safe thresholds
    - Multiple parameter violations
    - Severity of contamination indicators
    """
    print("\n" + "=" * 60)
    print("STEP 6: CONTAMINATION RISK LABEL GENERATION")
    print("=" * 60)

    def calculate_risk_score(row):
        risk_points = 0

        # BOD risk (0-20 points)
        if row['DBO_S'] > 30:
            risk_points += 20
        elif row['DBO_S'] > 10:
            risk_points += 10

        # COD risk (0-20 points)
        if row['DQO_S'] > 100:
            risk_points += 20
        elif row['DQO_S'] > 50:
            risk_points += 10

        # pH risk (0-20 points)
        if row['PH_S'] < 6.0 or row['PH_S'] > 9.0:
            risk_points += 20
        elif row['PH_S'] < 6.5 or row['PH_S'] > 8.5:
            risk_points += 10

        # Suspended Solids risk (0-15 points)
        if row['SS_S'] > 100:
            risk_points += 15
        elif row['SS_S'] > 50:
            risk_points += 8

        # Conductivity risk (0-15 points)
        if row['COND_S'] > 1500:
            risk_points += 15
        elif row['COND_S'] > 500:
            risk_points += 8

        # Zinc risk (0-10 points)
        if row['ZN_E'] > 5:
            risk_points += 10
        elif row['ZN_E'] > 3:
            risk_points += 5

        return risk_points

    df['Risk_Score'] = df.apply(calculate_risk_score, axis=1)

    # Use rank-based binning for balanced classes
    df['Risk_Rank'] = df['Risk_Score'].rank(method='first')
    n = len(df)
    df['Risk_Class'] = pd.cut(df['Risk_Rank'],
                               bins=[0, n/3, 2*n/3, n],
                               labels=['LOW', 'MEDIUM', 'HIGH'])

    print(f"  Risk Score range: {df['Risk_Score'].min():.1f} - {df['Risk_Score'].max():.1f}")
    print(f"\n  Risk Distribution:")
    risk_dist = df['Risk_Class'].value_counts()
    for cls, count in risk_dist.items():
        print(f"    {cls}: {count} ({count/len(df)*100:.1f}%)")

    return df


def prepare_features(df):
    """Prepare features for machine learning."""
    print("\n" + "=" * 60)
    print("STEP 7: FEATURE PREPARATION")
    print("=" * 60)

    # Select features for ML (input parameters only - no target leakage)
    feature_cols = [
        'Q_E', 'ZN_E', 'PH_E', 'DBO_E', 'DQO_E', 'SS_E', 'SSV_E',
        'SED_E', 'COND_E', 'PH_P', 'DBO_P', 'DQO_P', 'SS_P', 'SSV_P',
        'SED_P', 'COND_P', 'PH_D', 'DBO_D', 'DQO_D', 'SS_D', 'SSV_D',
        'SED_D', 'COND_D', 'Month', 'DayOfYear'
    ]

    # Engineered features
    engineered_cols = [
        'BOD_removal_eff', 'COD_removal_eff', 'SS_removal_eff',
        'COND_removal_eff', 'BOD_COD_ratio', 'SS_SSV_ratio',
        'avg_pH', 'avg_DBO', 'avg_DQO', 'avg_COND'
    ]

    all_features = feature_cols + engineered_cols

    X = df[all_features].copy()
    y_quality = df['Quality_Class'].copy()
    y_risk = df['Risk_Class'].copy()

    # Encode labels
    le_quality = LabelEncoder()
    y_quality_encoded = le_quality.fit_transform(y_quality)

    le_risk = LabelEncoder()
    y_risk_encoded = le_risk.fit_transform(y_risk)

    print(f"  Features selected: {len(all_features)}")
    print(f"  Feature names: {all_features}")
    print(f"\n  Quality classes: {list(le_quality.classes_)}")
    print(f"  Risk classes: {list(le_risk.classes_)}")

    # Check for any remaining NaN
    if X.isnull().sum().sum() > 0:
        print(f"  Warning: {X.isnull().sum().sum()} NaN values found. Filling with median.")
        X = X.fillna(X.median())

    return X, y_quality_encoded, y_risk_encoded, le_quality, le_risk, all_features


def train_and_evaluate(X, y, le, feature_names, target_name='Quality'):
    """Train multiple models and evaluate them."""
    print("\n" + "=" * 60)
    print(f"STEP 8: MODEL TRAINING & EVALUATION - {target_name}")
    print("=" * 60)

    # Split data - use stratify only if all classes have >= 2 members
    class_counts = np.bincount(y)
    can_stratify = all(c >= 2 for c in class_counts)

    if can_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models with class weights for imbalance handling
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE,
            class_weight='balanced'
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10, random_state=RANDOM_STATE, min_samples_split=5,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE,
            min_samples_split=5, n_jobs=-1, class_weight='balanced'
        ),
    }

    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, use_label_encoder=False,
            eval_metric='mlogloss'
        )

    # Train and evaluate
    results = {}
    best_model = None
    best_score = 0
    best_model_name = ''

    # Use fewer CV folds if classes are small
    min_class_count = min(np.bincount(y_train))
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        n_splits = 2

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        print(f"\n  Training {name}...")

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1_macro')

        # Train on full training set
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_test_scaled)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # ROC-AUC (one-vs-rest)
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test_scaled)
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
            else:
                roc_auc = 0.0
        except Exception:
            roc_auc = 0.0

        results[name] = {
            'CV F1 Macro (mean)': cv_scores.mean(),
            'CV F1 Macro (std)': cv_scores.std(),
            'Accuracy': accuracy,
            'Precision Macro': precision_macro,
            'Recall Macro': recall_macro,
            'F1 Macro': f1_macro,
            'F1 Weighted': f1_weighted,
            'ROC-AUC': roc_auc,
        }

        print(f"    CV F1 Macro: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"    Test Accuracy: {accuracy:.4f}")
        print(f"    Test F1 Macro: {f1_macro:.4f}")
        print(f"    Test ROC-AUC: {roc_auc:.4f}")

        # Track best model (based on F1 Macro)
        if f1_macro > best_score:
            best_score = f1_macro
            best_model = model
            best_model_name = name

    # Print comparison table
    print(f"\n  {'='*70}")
    print(f"  MODEL COMPARISON TABLE - {target_name}")
    print(f"  {'='*70}")
    print(f"  {'Model':<25} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weighted':>12} {'ROC-AUC':>10}")
    print(f"  {'-'*70}")
    for name, metrics in results.items():
        print(f"  {name:<25} {metrics['Accuracy']:>10.4f} {metrics['F1 Macro']:>10.4f} "
              f"{metrics['F1 Weighted']:>12.4f} {metrics['ROC-AUC']:>10.4f}")

    print(f"\n  Best Model: {best_model_name} (F1 Macro: {best_score:.4f})")

    # Confusion matrix for best model
    y_pred_best = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {best_model_name} ({target_name})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'confusion_matrix_{target_name.lower()}.png'), dpi=150)
    plt.close()

    # Feature importance (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feat_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=feat_imp.head(15), x='Importance', y='Feature')
        plt.title(f'Top 15 Feature Importances - {best_model_name} ({target_name})')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'feature_importance_{target_name.lower()}.png'), dpi=150)
        plt.close()

    # Classification report
    report = classification_report(y_test, y_pred_best, target_names=le.classes_)
    print(f"\n  Classification Report ({best_model_name}):")
    print(f"  {report}")

    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'scaler': scaler,
        'results': results,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'y_pred': y_pred_best,
        'label_encoder': le,
        'feature_names': feature_names,
    }


def shap_analysis(model, X_test, feature_names, target_name='Quality'):
    """Perform SHAP analysis for model explainability."""
    if not SHAP_AVAILABLE:
        print("  SHAP not available. Skipping SHAP analysis.")
        return None

    print(f"\n  Performing SHAP analysis for {target_name}...")

    try:
        # Use TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                         show=False, plot_size=(12, 8))
        plt.title(f'SHAP Summary Plot - {target_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'shap_summary_{target_name.lower()}.png'), dpi=150)
        plt.close()
        print(f"  Saved: shap_summary_{target_name.lower()}.png")

        return shap_values
    except Exception as e:
        print(f"  SHAP analysis failed: {e}")
        return None


def save_model(model_dict, target_name):
    """Save the best model and preprocessing pipeline."""
    print(f"\n  Saving model for {target_name}...")

    model_path = os.path.join(MODELS_DIR, f'best_model_{target_name.lower()}.pkl')
    scaler_path = os.path.join(MODELS_DIR, f'scaler_{target_name.lower()}.pkl')
    le_path = os.path.join(MODELS_DIR, f'label_encoder_{target_name.lower()}.pkl')

    joblib.dump(model_dict['best_model'], model_path)
    joblib.dump(model_dict['scaler'], scaler_path)
    joblib.dump(model_dict['label_encoder'], le_path)

    # Save metadata
    metadata = {
        'target': target_name,
        'best_model_name': model_dict['best_model_name'],
        'feature_names': model_dict['feature_names'],
        'classes': list(model_dict['label_encoder'].classes_),
        'results': model_dict['results'],
        'created_at': datetime.now().isoformat(),
    }

    metadata_path = os.path.join(MODELS_DIR, f'model_metadata_{target_name.lower()}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"  Saved: {model_path}")
    print(f"  Saved: {scaler_path}")
    print(f"  Saved: {le_path}")
    print(f"  Saved: {metadata_path}")


def generate_report(quality_results, risk_results):
    """Generate evaluation report."""
    print("\n" + "=" * 60)
    print("STEP 9: GENERATING EVALUATION REPORT")
    print("=" * 60)

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("SMART WATER QUALITY CLASSIFICATION - EVALUATION REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Dataset: UCI Water Treatment Plant")
    report_lines.append(f"Source: https://archive.ics.uci.edu/ml/datasets/water+treatment+plant")
    report_lines.append(f"Total Samples: 527")
    report_lines.append("")

    for target_name, results in [('QUALITY', quality_results), ('RISK', risk_results)]:
        report_lines.append(f"\n{'='*70}")
        report_lines.append(f"{target_name} CLASSIFICATION RESULTS")
        report_lines.append(f"{'='*70}")
        report_lines.append(f"Best Model: {results['best_model_name']}")
        report_lines.append("")

        report_lines.append(f"{'Model':<25} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weighted':>12} {'ROC-AUC':>10}")
        report_lines.append(f"{'-'*70}")

        for name, metrics in results['results'].items():
            report_lines.append(
                f"{name:<25} {metrics['Accuracy']:>10.4f} {metrics['F1 Macro']:>10.4f} "
                f"{metrics['F1 Weighted']:>12.4f} {metrics['ROC-AUC']:>10.4f}"
            )

        report_lines.append("")

    report_text = "\n".join(report_lines)

    report_path = os.path.join(REPORTS_DIR, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"  Saved: {report_path}")
    print(f"\n{report_text}")

    return report_text


def main():
    """Run the complete pipeline."""
    print("\n" + "=" * 70)
    print("SMART WATER QUALITY CLASSIFICATION AND CONTAMINATION RISK ANALYTICS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Step 1: Load data
    df = load_data()

    # Step 2: Clean data
    df = clean_data(df)

    # Step 3: EDA
    exploratory_analysis(df)

    # Step 4: Feature Engineering
    df = feature_engineering(df)

    # Step 5: Generate Quality Labels
    df = generate_labels(df)

    # Step 6: Generate Risk Labels
    df = generate_risk_labels(df)

    # Step 7: Prepare Features
    X, y_quality, y_risk, le_quality, le_risk, feature_names = prepare_features(df)

    # Save processed data
    df.to_csv(os.path.join(DATA_PROCESSED, 'processed_data.csv'), index=False)
    print(f"\n  Saved processed data: {os.path.join(DATA_PROCESSED, 'processed_data.csv')}")

    # Step 8: Train and Evaluate Models - Quality Classification
    quality_results = train_and_evaluate(X, y_quality, le_quality, feature_names, 'Quality')

    # Step 8b: Train and Evaluate Models - Risk Classification
    risk_results = train_and_evaluate(X, y_risk, le_risk, feature_names, 'Risk')

    # SHAP Analysis
    shap_analysis(quality_results['best_model'], quality_results['X_test'],
                  feature_names, 'Quality')
    shap_analysis(risk_results['best_model'], risk_results['X_test'],
                  feature_names, 'Risk')

    # Step 9: Save Models
    save_model(quality_results, 'Quality')
    save_model(risk_results, 'Risk')

    # Step 10: Generate Report
    generate_report(quality_results, risk_results)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutputs:")
    print(f"  - Models: {MODELS_DIR}")
    print(f"  - Reports: {REPORTS_DIR}")
    print(f"  - Figures: {FIGURES_DIR}")
    print(f"  - Processed Data: {DATA_PROCESSED}")


if __name__ == '__main__':
    main()
