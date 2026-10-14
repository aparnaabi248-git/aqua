# AquaGuard AI — Smart Water Quality & Treatment Monitoring System

> Upgraded dashboard built on the project below: sky-blue/dark-blue theme, 7 pages
> (Dashboard, Test a Water Sample, Alerts & Monitoring, Analytics & Trends,
> ML Model Insights, Reports & History, **Government Data**), validated sample
> prediction, alert history, persistent sample history, and genuine CPCB/NWDP
> government-data integration. No new third-party dependencies were added.

## AquaGuard AI quick start (Windows PowerShell)

```powershell
cd "C:\Users\user\Downloads\apar dsa\smart-water-quality"
py -m pip install -r requirements.txt
py -m streamlit run dashboard/app.py
# open http://localhost:8501
```

Run tests:

```powershell
py -m pytest tests/ -q
```

Connect a live sensor (optional): copy `.env.example` to `.env` and set one of
`AQUAGUARD_LIVE_CSV` / `AQUAGUARD_LIVE_JSON` / `AQUAGUARD_LIVE_URL`.

## AquaGuard AI data sources

1. **HISTORICAL** — UCI Water Treatment Plant dataset (1990, single plant, 527 records).
2. **SIMULATED** — reproducible demo readings, always labelled simulated.
3. **GOVERNMENT** — official CPCB surface-water observations via the National Water
   Data Portal (`https://nwdp.nwic.gov.in/`, public CKAN API, no key). Default view:
   Tamil Nadu physical parameters 2021–2025 (401 records, latest observation
   2021-12-13). Cached under `data/app_data/govt_cache/`; refresh validates before
   replacing cache and never fabricates rows. Displayed independently — never fed
   into the UCI-trained models.
4. **LIVE SENSOR** — your configured CSV/JSON/URL only; never faked.

## AquaGuard AI limitations

- UCI labels describe **wastewater-treatment operating condition**, not drinking-water safety.
- Some UCI treated-output columns contain recording quirks (e.g. implausible pH);
  the dashboard flags rather than hides them.
- Government data is periodic (monthly/annual), not real-time; future portal buckets
  (e.g. 2026–2030) may be header-only files with 0 rows and are reported as such.
- Watchlist thresholds on the Government Data page are user-configured, not CPCB standards.
- PDF export needs `fpdf2` (not installed); Markdown/CSV export is built in.

---

# Smart Water Quality Classification and Contamination Risk Analytics

A comprehensive Data Science and Machine Learning project for water quality classification, contamination risk assessment, and interactive analytics using real-world water treatment plant data.

## Project Overview

This project demonstrates the complete Data Science lifecycle using the UCI Water Treatment Plant dataset. It classifies water quality into GOOD/MODERATE/POOR categories, assesses contamination risk as LOW/MEDIUM/HIGH, and provides interactive analytics through a Streamlit dashboard.

## Features

- **Real Government/Public Data**: Uses the UCI Water Treatment Plant dataset
- **Complete DS Lifecycle**: Data collection → Cleaning → EDA → Feature Engineering → ML → Evaluation → Deployment
- **Water Quality Classification**: Multiclass classification (GOOD/MODERATE/POOR)
- **Contamination Risk Analytics**: Risk assessment (LOW/MEDIUM/HIGH)
- **Explainable ML**: SHAP analysis and feature importance
- **Interactive Dashboard**: Streamlit-based analytics platform
- **Model Comparison**: Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Time Series Analysis**: Historical trends and seasonal patterns

## Project Structure

```
smart-water-quality/
│
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned and processed data
│   └── README.md               # Data documentation
│
├── notebooks/
│   └── water_quality_analysis.ipynb
│
├── src/
│   ├── pipeline.py             # Main pipeline script
│   ├── data_loader.py          # Data loading module
│   ├── data_preprocessing.py   # Data cleaning module
│   ├── exploratory_analysis.py # EDA module
│   ├── feature_engineering.py  # Feature engineering module
│   ├── label_generation.py     # Label generation module
│   ├── train_models.py         # Model training module
│   ├── evaluate_models.py      # Model evaluation module
│   ├── risk_analysis.py        # Risk analysis module
│   ├── explainability.py        # Explainability module
│   └── prediction.py           # Prediction module
│
├── models/                     # Saved models and metadata
│
├── dashboard/
│   └── app.py                  # Streamlit dashboard
│
├── reports/
│   ├── model_comparison.csv
│   ├── evaluation_report.txt
│   └── figures/                # Generated visualizations
│
├── requirements.txt
├── README.md
├── DATA_SOURCES.md
├── .gitignore
└── LICENSE
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone or download the project**

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**

   Windows:
   ```bash
   venv\Scripts\activate
   ```

   Linux/Mac:
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Complete Pipeline

```bash
python src/pipeline.py
```

This will:
1. Load and clean the data
2. Perform EDA and generate visualizations
3. Engineer features
4. Generate quality and risk labels
5. Train and compare multiple ML models
6. Perform SHAP analysis
7. Save the best models
8. Generate evaluation reports

### Running the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard provides:
- **Overview**: KPI cards and summary statistics
- **Data Exploration**: Dataset preview, distributions, correlations
- **Water Quality Analytics**: Quality distributions and trends
- **Contamination Risk**: Risk analysis and threshold comparisons
- **ML Model Analysis**: Model comparison and feature importance
- **Sample Prediction**: Interactive prediction form

## Dataset

**Source**: UCI Machine Learning Repository
**URL**: https://archive.ics.uci.edu/ml/datasets/water+treatment+plant
**Records**: 527 samples
**Features**: 39 parameters
**Time Period**: 1990

See [DATA_SOURCES.md](DATA_SOURCES.md) for detailed documentation.

## ML Methodology

### Label Generation

Labels are generated using a composite scoring system based on CPCB (Central Pollution Control Board) standards:

- **GOOD**: Quality Score >= 70
- **MODERATE**: Quality Score 40-69
- **POOR**: Quality Score < 40

### Models Trained

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

### Evaluation Metrics

- Accuracy
- Precision (Macro)
- Recall (Macro)
- F1 Score (Macro & Weighted)
- ROC-AUC
- Cross-validation (5-fold)

## Results

See `reports/evaluation_report.txt` for actual model performance metrics.

## Limitations

- Dataset quality affects model performance
- Single treatment plant location limits generalizability
- Historical data (1990) may not reflect current conditions
- Not all contaminants are represented in the dataset
- ML predictions are data-driven estimates, not guarantees
- Water safety requires laboratory testing and regulatory assessment
- Static dataset cannot be described as real-time

## Future Enhancements

1. IoT water-quality sensors integration
2. Real-time sensor data processing
3. Government API integration (India-WRIS, CPCB, CGWB)
4. GPS-based water-quality monitoring
5. Mobile application
6. Automated alerts system
7. Cloud deployment
8. Time-series forecasting
9. More advanced explainable AI
10. Integration with additional monitoring stations

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Disclaimer

This system is a data-driven water quality classification and risk assessment tool. It does NOT replace laboratory testing or regulatory water safety assessment. Always consult appropriate authorities for official drinking-water safety determination.
