# Data Sources

## Primary Dataset

### UCI Water Treatment Plant Dataset

| Attribute | Details |
|-----------|---------|
| **Dataset Name** | Water Treatment Plant |
| **Organization** | UCI Machine Learning Repository |
| **Source URL** | https://archive.ics.uci.edu/ml/datasets/water+treatment+plant |
| **Download Date** | 2026-10-07 |
| **Number of Records** | 527 |
| **Number of Features** | 39 (including target) |
| **Time Period** | 1990 (daily measurements) |
| **Geographic Coverage** | Single water treatment plant |
| **Data Type** | Historical, Static |
| **License** | CC BY 4.0 (UCI ML Repository) |

### Parameters Available

The dataset contains measurements from different stages of a water treatment plant:

**Input Parameters (Raw Water):**
- Q_E: Flow rate
- ZN_E: Zinc concentration
- PH_E: pH level
- DBO_E: Biological Oxygen Demand (BOD)
- DQO_E: Chemical Oxygen Demand (COD)
- SS_E: Suspended Solids
- SSV_E: Volatile Suspended Solids
- SED_E: Sediment
- COND_E: Electrical Conductivity

**Primary Settler Parameters:**
- PH_P, DBO_P, DQO_P, SS_P, SSV_P, SED_P, COND_P

**Secondary Settler Parameters:**
- PH_D, DBO_D, DQO_D, SS_D, SSV_D, SED_D, COND_D

**Final Output Parameters (Treated Water):**
- PH_S, DBO_S, DQO_S, SS_S, SSV_S, SED_S, COND_S

**Removal Efficiency:**
- RD_DBO_S, RD_DQO_S, RD_SS_S, RD_SED_S, RD_CON_S
- RD_DBO_G, RD_DQO_G, RD_SS_G

### Data Quality Notes

- Missing values are marked with '?' in the original dataset
- Some columns have significant missing data (up to 30%)
- The dataset represents a single treatment plant, limiting geographic generalizability
- Data is from 1990 only, representing historical conditions

### Limitations

1. **Single Location**: Data from one treatment plant only
2. **Historical Data**: From 1990, may not reflect current conditions
3. **Missing Values**: Some parameters have significant missing data
4. **No Contaminants**: Does not include heavy metals (except zinc), pesticides, or microbial indicators
5. **Static Dataset**: Not real-time or near-real-time
6. **Limited Temporal Scope**: Only one year of data

## Label Generation Methodology

### Water Quality Labels (GOOD/MODERATE/POOR)

Labels are generated using a composite scoring system based on CPCB (Central Pollution Control Board) standards:

| Parameter | GOOD | MODERATE | POOR | Weight |
|-----------|------|----------|------|--------|
| BOD (output) | < 10 mg/L | 10-30 mg/L | > 30 mg/L | 25% |
| COD (output) | < 50 mg/L | 50-100 mg/L | > 100 mg/L | 25% |
| pH (output) | 6.5-8.5 | 6.0-9.0 | < 6.0 or > 9.0 | 20% |
| Suspended Solids | < 50 mg/L | 50-100 mg/L | > 100 mg/L | 15% |
| Conductivity | < 500 µS/cm | 500-1500 µS/cm | > 1500 µS/cm | 15% |

**Classification:**
- GOOD: Score >= 70
- MODERATE: Score 40-69
- POOR: Score < 40

### Contamination Risk Labels (LOW/MEDIUM/HIGH)

Risk is calculated based on parameter threshold violations:

| Risk Level | Score Range |
|------------|-------------|
| LOW | 0-15 |
| MEDIUM | 16-35 |
| HIGH | > 35 |

## References

1. UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/
2. CPCB Standards: https://cpcb.nic.in/
3. WHO Drinking Water Guidelines: https://www.who.int/
