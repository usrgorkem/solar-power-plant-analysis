# ☀️ Solar Power Plant Analysis

A complete professional analysis of a solar power plant's 15-minute SCADA dataset (2020).  
Covers data loading, cleaning, statistics, 8 visualisation plots, feature insights, and a written conclusion.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) |
| Plant | Plant 1 |
| Period | May 15, 2020 – June 17, 2020 |
| Interval | 15 minutes |
| Inverters | 22 |
| Peak AC Capacity | ~29,000 kW |
| Files | `Plant_1_Generation_Data.csv` + `Plant_1_Weather_Sensor_Data.csv` |

> **Note:** The dataset is not included in this repository due to size.  
> Download it from Kaggle and place both CSV files inside the `data/` folder.

---

## 📁 Project Structure

```
solar-power-plant-analysis/
├── data/
│   ├── Plant_1_Generation_Data.csv       ← download from Kaggle
│   └── Plant_1_Weather_Sensor_Data.csv   ← download from Kaggle
├── outputs/
│   └── figures/                          ← generated plots saved here
├── solar_power_plant_analysis.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔍 What the Script Does

| Section | Description |
|---|---|
| 1 — Data Loading | Loads both CSV files, prints shape, dtypes, and column descriptions |
| 2 — Cleaning | Parses timestamps, aggregates 22 inverters, merges datasets, flags night-time readings |
| 3 — Statistics | Descriptive stats, capacity factor, performance ratio, peak hour |
| 4 — Visualisation | 7 publication-ready plots (see below) |
| 5 — Feature Insights | Pearson correlations, irradiation bin analysis, top predictors |
| 6 — Key Insights | Written interpretation of all major findings |
| 7 — Conclusion | Summary and suggestions for future work |

### Plots Generated

| # | Plot |
|---|---|
| 01 | Daily Total AC Power Time Series |
| 02 | Pearson Correlation Heatmap |
| 03 | Monthly Average AC Power |
| 04 | Diurnal AC Power Pattern (hour of day) |
| 05 | Irradiation vs AC Power (scatter + regression) |
| 06 | AC Power Distribution by Month (box plot) |
| 07 | Irradiation Distribution (histogram + KDE) |
| 08 | Feature Correlation Ranking (bar chart) |

---

## ⚙️ Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/usrgorkem/solar-power-plant-analysis.git
cd solar-power-plant-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Download from [Kaggle](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) and place both CSV files in `data/`:

```
data/Plant_1_Generation_Data.csv
data/Plant_1_Weather_Sensor_Data.csv
```

### 4. Run the analysis

```bash
# Auto-detects CSVs in data/ folder
python solar_power_plant_analysis.py

# Or specify paths manually
python solar_power_plant_analysis.py \
  --gen path/to/Plant_1_Generation_Data.csv \
  --weather path/to/Plant_1_Weather_Sensor_Data.csv
```

Plots are saved to `outputs/figures/`.

---

## 📈 Sample Results

| Metric | Value |
|---|---|
| Capacity Factor | ~14% |
| Avg Performance Ratio | ~95% |
| Peak Production Hour | 12:00 |
| Total Yield (34 days) | ~1,200,000 kWh |
| Irradiation–Power Correlation | r ≈ 0.97 |

---

## 🛠️ Requirements

- Python 3.10+
- pandas, numpy, matplotlib, seaborn

See `requirements.txt` for pinned versions.

---

## 👤 Author

**Mehmet Görkem User**  
GitHub: [@usrgorkem](https://github.com/usrgorkem)

---

## 📄 License

This project is licensed under the MIT License.
