"""Data loading module for water quality dataset."""
import os
import pandas as pd
import numpy as np

DATA_RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')

COLUMN_NAMES = [
    'Date', 'Q_E', 'ZN_E', 'PH_E', 'DBO_E', 'DQO_E', 'SS_E', 'SSV_E',
    'SED_E', 'COND_E', 'PH_P', 'DBO_P', 'DQO_P', 'SS_P', 'SSV_P',
    'SED_P', 'COND_P', 'PH_D', 'DBO_D', 'DQO_D', 'SS_D', 'SSV_D',
    'SED_D', 'COND_D', 'PH_S', 'DBO_S', 'DQO_S', 'SS_S', 'SSV_S',
    'SED_S', 'COND_S', 'RD_DBO_S', 'RD_DQO_S', 'RD_SS_S', 'RD_SED_S',
    'RD_CON_S', 'RD_DBO_G', 'RD_DQO_G', 'RD_SS_G'
]


def load_water_quality_data(filepath=None):
    """Load the UCI Water Treatment Plant dataset."""
    if filepath is None:
        filepath = os.path.join(DATA_RAW, 'uci_water_treatment.csv')

    df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)
    df = df.replace('?', np.nan)

    # Convert numeric columns
    for col in df.columns:
        if col != 'Date':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Parse dates
    df['Date'] = df['Date'].str.replace('D-', '')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y', errors='coerce')

    return df
