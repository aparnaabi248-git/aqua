"""Data preprocessing module for water quality dataset."""
import pandas as pd
import numpy as np


def clean_data(df):
    """Clean the dataset: handle missing values, duplicates, invalid values."""
    # Convert all columns to numeric where possible
    for col in df.columns:
        if col != 'Date':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Parse dates
    df['Date'] = df['Date'].str.replace('D-', '')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y', errors='coerce')

    # Handle missing values - median imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # Remove duplicates
    df = df.drop_duplicates()

    return df


def handle_outliers(df, columns, method='iqr', factor=1.5):
    """Handle outliers using IQR method."""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        df_clean[col] = df_clean[col].clip(lower, upper)
    return df_clean
