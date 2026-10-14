"""AquaGuard AI central configuration: theme, parameters, thresholds, features."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
APP_DATA_DIR = os.path.join(BASE_DIR, 'data', 'app_data')
os.makedirs(APP_DATA_DIR, exist_ok=True)

# Exact 35-feature order used when the saved models were trained.
FEATURE_ORDER = [
    'Q_E', 'ZN_E', 'PH_E', 'DBO_E', 'DQO_E', 'SS_E', 'SSV_E',
    'SED_E', 'COND_E', 'PH_P', 'DBO_P', 'DQO_P', 'SS_P', 'SSV_P',
    'SED_P', 'COND_P', 'PH_D', 'DBO_D', 'DQO_D', 'SS_D', 'SSV_D',
    'SED_D', 'COND_D', 'Month', 'DayOfYear',
    'BOD_removal_eff', 'COD_removal_eff', 'SS_removal_eff',
    'COND_removal_eff', 'BOD_COD_ratio', 'SS_SSV_ratio',
    'avg_pH', 'avg_DBO', 'avg_DQO', 'avg_COND',
]

# User-editable input spec: key, label, unit, min, max, default, group, help.
# Defaults are medians of the training data (computed Oct 2026, see statistics.csv).
INPUT_SPEC = [
    # Raw water intake
    {'key': 'Q_E', 'label': 'Flow rate', 'unit': 'm³/day', 'min': 0.0, 'max': 100000.0, 'default': 35000.0, 'group': 'Raw water intake', 'help': 'Incoming flow to the plant.'},
    {'key': 'ZN_E', 'label': 'Zinc', 'unit': 'mg/L', 'min': 0.0, 'max': 20.0, 'default': 1.0, 'group': 'Raw water intake', 'help': 'Zinc concentration in raw water.'},
    {'key': 'PH_E', 'label': 'pH', 'unit': 'pH units', 'min': 0.0, 'max': 14.0, 'default': 7.5, 'group': 'Raw water intake', 'help': 'Acidity/alkalinity. 7 is neutral.'},
    {'key': 'DBO_E', 'label': 'BOD (Biochemical Oxygen Demand)', 'unit': 'mg/L', 'min': 0.0, 'max': 2000.0, 'default': 200.0, 'group': 'Raw water intake', 'help': 'Oxygen needed by microbes to break down organic matter. Higher = more polluted.'},
    {'key': 'DQO_E', 'label': 'COD (Chemical Oxygen Demand)', 'unit': 'mg/L', 'min': 0.0, 'max': 5000.0, 'default': 400.0, 'group': 'Raw water intake', 'help': 'Oxygen needed to chemically oxidise pollutants. Higher = more polluted.'},
    {'key': 'SS_E', 'label': 'Suspended solids', 'unit': 'mg/L', 'min': 0.0, 'max': 5000.0, 'default': 200.0, 'group': 'Raw water intake', 'help': 'Solid particles floating in water.'},
    {'key': 'SSV_E', 'label': 'Volatile suspended solids', 'unit': 'mg/L', 'min': 0.0, 'max': 5000.0, 'default': 150.0, 'group': 'Raw water intake', 'help': 'Organic part of suspended solids.'},
    {'key': 'SED_E', 'label': 'Sediment', 'unit': 'mL/L', 'min': 0.0, 'max': 100.0, 'default': 5.0, 'group': 'Raw water intake', 'help': 'Settleable solids.'},
    {'key': 'COND_E', 'label': 'Conductivity', 'unit': 'µS/cm', 'min': 0.0, 'max': 10000.0, 'default': 1500.0, 'group': 'Raw water intake', 'help': 'Ability to conduct electricity; relates to dissolved salts.'},
    # Primary settler
    {'key': 'PH_P', 'label': 'pH (primary settler)', 'unit': 'pH units', 'min': 0.0, 'max': 14.0, 'default': 7.4, 'group': 'Primary settler', 'help': 'pH after primary settling.'},
    {'key': 'DBO_P', 'label': 'BOD (primary)', 'unit': 'mg/L', 'min': 0.0, 'max': 2000.0, 'default': 150.0, 'group': 'Primary settler', 'help': 'BOD after primary settling.'},
    {'key': 'DQO_P', 'label': 'COD (primary)', 'unit': 'mg/L', 'min': 0.0, 'max': 5000.0, 'default': 350.0, 'group': 'Primary settler', 'help': 'COD after primary settling.'},
    {'key': 'SS_P', 'label': 'Suspended solids (primary)', 'unit': 'mg/L', 'min': 0.0, 'max': 5000.0, 'default': 120.0, 'group': 'Primary settler', 'help': 'Suspended solids after primary settling.'},
    {'key': 'SSV_P', 'label': 'Volatile solids (primary)', 'unit': 'mg/L', 'min': 0.0, 'max': 5000.0, 'default': 90.0, 'group': 'Primary settler', 'help': 'Volatile solids after primary settling.'},
    {'key': 'SED_P', 'label': 'Sediment (primary)', 'unit': 'mL/L', 'min': 0.0, 'max': 100.0, 'default': 3.0, 'group': 'Primary settler', 'help': 'Settleable solids after primary settling.'},
    {'key': 'COND_P', 'label': 'Conductivity (primary)', 'unit': 'µS/cm', 'min': 0.0, 'max': 10000.0, 'default': 1400.0, 'group': 'Primary settler', 'help': 'Conductivity after primary settling.'},
    # Secondary settler
    {'key': 'PH_D', 'label': 'pH (secondary)', 'unit': 'pH units', 'min': 0.0, 'max': 14.0, 'default': 7.4, 'group': 'Secondary settler', 'help': 'pH after secondary (biological) treatment.'},
    {'key': 'DBO_D', 'label': 'BOD (secondary)', 'unit': 'mg/L', 'min': 0.0, 'max': 2000.0, 'default': 80.0, 'group': 'Secondary settler', 'help': 'BOD after secondary treatment.'},
    {'key': 'DQO_D', 'label': 'COD (secondary)', 'unit': 'mg/L', 'min': 0.0, 'max': 5000.0, 'default': 200.0, 'group': 'Secondary settler', 'help': 'COD after secondary treatment.'},
    {'key': 'SS_D', 'label': 'Suspended solids (secondary)', 'unit': 'mg/L', 'min': 0.0, 'max': 5000.0, 'default': 80.0, 'group': 'Secondary settler', 'help': 'Suspended solids after secondary treatment.'},
    {'key': 'SSV_D', 'label': 'Volatile solids (secondary)', 'unit': 'mg/L', 'min': 0.0, 'max': 5000.0, 'default': 60.0, 'group': 'Secondary settler', 'help': 'Volatile solids after secondary treatment.'},
    {'key': 'SED_D', 'label': 'Sediment (secondary)', 'unit': 'mL/L', 'min': 0.0, 'max': 100.0, 'default': 2.0, 'group': 'Secondary settler', 'help': 'Settleable solids after secondary treatment.'},
    {'key': 'COND_D', 'label': 'Conductivity (secondary)', 'unit': 'µS/cm', 'min': 0.0, 'max': 10000.0, 'default': 1300.0, 'group': 'Secondary settler', 'help': 'Conductivity after secondary treatment.'},
]

# Operational limits for treated-water outputs, from CPCB standards as documented
# in DATA_SOURCES.md. These are OPERATIONAL treatment targets, NOT drinking-water
# safety certifications.
OPERATIONAL_LIMITS = {
    'DBO_S': {'label': 'BOD (treated output)', 'unit': 'mg/L', 'warn': 10.0, 'critical': 30.0, 'source': 'CPCB treated-water guideline (DATA_SOURCES.md)'},
    'DQO_S': {'label': 'COD (treated output)', 'unit': 'mg/L', 'warn': 50.0, 'critical': 100.0, 'source': 'CPCB treated-water guideline (DATA_SOURCES.md)'},
    'SS_S': {'label': 'Suspended solids (treated output)', 'unit': 'mg/L', 'warn': 50.0, 'critical': 100.0, 'source': 'CPCB treated-water guideline (DATA_SOURCES.md)'},
    'COND_S': {'label': 'Conductivity (treated output)', 'unit': 'µS/cm', 'warn': 500.0, 'critical': 1500.0, 'source': 'CPCB treated-water guideline (DATA_SOURCES.md)'},
    'PH_S': {'label': 'pH (treated output)', 'unit': 'pH units', 'warn_low': 6.5, 'warn_high': 8.5, 'critical_low': 6.0, 'critical_high': 9.0, 'source': 'CPCB treated-water guideline (DATA_SOURCES.md)'},
}

# Input-stage alert limits (raw water) are NOT established by the dataset, so they
# require operator configuration. Marked explicitly as such.
NEEDS_CONFIGURATION = ['Q_E', 'ZN_E', 'PH_E', 'DBO_E', 'DQO_E', 'SS_E', 'COND_E']

PARAMETER_GLOSSARY = {
    'pH': 'Acidity/alkalinity scale 0–14. 7 is neutral. Treatment target for treated output is 6.5–8.5.',
    'BOD': 'Biochemical Oxygen Demand (mg/L): oxygen microbes need to break down organic matter. Lower is cleaner.',
    'COD': 'Chemical Oxygen Demand (mg/L): oxygen needed to chemically oxidise pollutants. Lower is cleaner.',
    'Suspended solids': 'Solid particles floating in water (mg/L). High values mean cloudy, poorly treated water.',
    'Conductivity': 'Ability to carry electric current (µS/cm); rises with dissolved salts and some pollutants.',
    'Zinc': 'Zinc concentration (mg/L); a metal indicator in this dataset (not a full heavy-metal screen).',
    'Flow rate': 'Volume entering the plant per day. Affects treatment performance.',
}
