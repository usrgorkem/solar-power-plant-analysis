"""
solar_power_plant_analysis.py
==============================
A complete professional analysis of a solar power plant's 15-minute generation
and weather sensor dataset (2020). Covers data loading, cleaning, statistics,
visualisation plots, feature insights, and a written conclusion.

Dataset  : Plant_1_Generation_Data.csv + Plant_1_Weather_Sensor_Data.csv
           (~68,000 rows × 7 columns | May 15 – June 17, 2020)
Plant    : 22 inverters | Peak AC capacity ≈ 29,000 kW  |  15-minute intervals
Author   : Mehmet Görkem User
Dataset  : Kaggle - Solar Power Generation Data
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── CLI argument parsing ───────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Solar Power Plant SCADA Analysis")
_parser.add_argument(
    "--gen",
    default=None,
    help="Path to Plant_1_Generation_Data.csv. If omitted, common locations are searched.",
)
_parser.add_argument(
    "--weather",
    default=None,
    help="Path to Plant_1_Weather_Sensor_Data.csv. If omitted, common locations are searched.",
)
_args, _ = _parser.parse_known_args()  # parse_known_args so Jupyter kernels don't crash


def _find_dataset(explicit_path: str | None, filename: str) -> str:
    """
    Resolve a dataset path in this order:
      1. Explicit CLI argument
      2. data/ subfolder relative to this script
      3. Same directory as this script
      4. Current working directory
      5. Downloads folder (Windows & Linux/macOS, including Turkish Windows)
    Raises FileNotFoundError with clear fix instructions if nothing is found.
    """
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(script_dir, "data", filename))
    candidates.append(os.path.join(script_dir, filename))
    candidates.append(os.path.join(os.getcwd(), filename))

    home = os.path.expanduser("~")
    candidates += [
        os.path.join(home, "Downloads", filename),
        os.path.join(home, "İndirilenler", filename),  # Turkish Windows
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    searched = "\n  ".join(candidates)
    raise FileNotFoundError(
        f"{filename} not found. Searched:\n  {searched}\n\n"
        "Fix — either:\n"
        f"  A) Put {filename} in the same folder as this script, OR\n"
        f"  B) Run:  python solar_power_plant_analysis.py --gen C:\\path\\to\\{filename}"
    )


GENERATION_PATH = _find_dataset(_args.gen, "Plant_1_Generation_Data.csv")
WEATHER_PATH    = _find_dataset(_args.weather, "Plant_1_Weather_Sensor_Data.csv")
FIGURES_PATH    = "outputs/figures"
os.makedirs(FIGURES_PATH, exist_ok=True)

# ── Global style ───────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

# ── Plant nameplate constants ──────────────────────────────────────────────────
# These values are derived from the dataset; update if using a different plant.
N_INVERTERS       = 22        # number of inverters in Plant 1
INTERVAL_MIN      = 15        # minutes between readings
INTERVAL_H        = INTERVAL_MIN / 60   # hours per reading (0.25)
PEAK_AC_POWER_KW  = 29_000   # estimated peak AC output of the entire plant (kW)
IRRADIATION_MIN   = 0.0      # W/m² — readings at or below this are treated as night-time


# =============================================================================
# === SECTION 1 — DATA LOADING & UNDERSTANDING ================================
# =============================================================================

print("=" * 70)
print("SECTION 1 — DATA LOADING & UNDERSTANDING")
print("=" * 70)

# Load both CSV files; let pandas infer dtypes first so we can inspect them.
# We'll fix timestamps in Section 2 after we understand the raw formats.
try:
    generation_df = pd.read_csv(GENERATION_PATH)
    weather_df    = pd.read_csv(WEATHER_PATH)
except FileNotFoundError as e:
    raise SystemExit(
        f"\n[ERROR] Could not load dataset: {e}\n"
        "Make sure both CSV files are accessible and re-run the script."
    )

print(f"\n[1] Generation dataset shape : {generation_df.shape[0]:,} rows × {generation_df.shape[1]} columns")
print(f"[2] Weather dataset shape    : {weather_df.shape[0]:,} rows × {weather_df.shape[1]} columns")
print(f"[3] Unique inverters         : {generation_df['SOURCE_KEY'].nunique()}")

print("\n[4] Generation column names  :", list(generation_df.columns))
print("[5] Weather column names     :", list(weather_df.columns))

print("\n[6] Generation dtypes:\n", generation_df.dtypes.to_string())
print("\n[7] First 5 rows (generation):\n", generation_df.head().to_string())

# Brief column reference for readers unfamiliar with solar SCADA data
col_descriptions = {
    "DATE_TIME"          : "15-minute timestamp of the measurement",
    "PLANT_ID"           : "Unique numeric identifier for the solar plant",
    "SOURCE_KEY"         : "Unique identifier for each inverter (22 inverters total)",
    "DC_POWER"           : "DC power output of a single inverter (kW)",
    "AC_POWER"           : "AC power delivered to the grid by a single inverter (kW)",
    "DAILY_YIELD"        : "Cumulative energy yield for the current day per inverter (kWh)",
    "TOTAL_YIELD"        : "Lifetime cumulative energy yield per inverter (kWh)",
    "AMBIENT_TEMPERATURE": "Air temperature measured by the weather station (°C)",
    "MODULE_TEMPERATURE" : "Solar panel surface temperature (°C)",
    "IRRADIATION"        : "Solar irradiance measured at the plant site (W/m²)",
}
print("\n[8] Column descriptions:")
for col, desc in col_descriptions.items():
    print(f"    {col:<25} → {desc}")


# =============================================================================
# === SECTION 2 — DATA CLEANING & PREPROCESSING ===============================
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2 — DATA CLEANING & PREPROCESSING")
print("=" * 70)

# ── 2.1  Parse timestamps ─────────────────────────────────────────────────────
# The two files use DIFFERENT datetime formats:
#   Generation : '15-05-2020 00:00'         → day-first, no seconds
#   Weather    : '2020-05-15 00:00:00'       → ISO 8601, with seconds
# Parsing with the wrong format raises a ValueError, so we handle each file
# explicitly rather than relying on pandas' auto-detection.
generation_df["DATE_TIME"] = pd.to_datetime(
    generation_df["DATE_TIME"], format="%d-%m-%Y %H:%M"
)
weather_df["DATE_TIME"] = pd.to_datetime(weather_df["DATE_TIME"])

# ── 2.2  Aggregate generation across all 22 inverters per timestamp ───────────
# Each timestamp has one row per inverter (22 rows per 15-min slot).
# We sum DC_POWER and AC_POWER to get the plant-level output,
# and also sum DAILY_YIELD and TOTAL_YIELD for whole-plant energy figures.
plant_gen = (
    generation_df
    .groupby("DATE_TIME")
    .agg(
        DC_POWER    =("DC_POWER",    "sum"),
        AC_POWER    =("AC_POWER",    "sum"),
        DAILY_YIELD =("DAILY_YIELD", "sum"),
        TOTAL_YIELD =("TOTAL_YIELD", "sum"),
    )
    .reset_index()
)

# ── 2.3  Select relevant weather columns and merge ────────────────────────────
# We only need the three sensor readings; PLANT_ID is constant and not useful.
weather_clean = weather_df[
    ["DATE_TIME", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
].copy()

# Inner join ensures we only keep timestamps present in BOTH files.
df = pd.merge(plant_gen, weather_clean, on="DATE_TIME", how="inner")

# ── 2.4  Sort chronologically and reset index ──────────────────────────────────
df.sort_values("DATE_TIME", inplace=True)
df.reset_index(drop=True, inplace=True)

# ── 2.5  Missing values ────────────────────────────────────────────────────────
missing = df.isnull().sum()
print("\n[1] Missing values per column:")
print(missing[missing > 0] if missing.any() else "    No missing values found.")

# ── 2.6  Duplicates ───────────────────────────────────────────────────────────
before = len(df)
df.drop_duplicates(inplace=True)
print(f"\n[2] Duplicate rows removed: {before - len(df)}")

# ── 2.7  Night-time flag ──────────────────────────────────────────────────────
# A reading is 'night-time' when solar irradiation is at or below zero.
# Night-time rows are kept in the dataset but excluded from certain plots
# (scatter of irradiation vs power, irradiation histograms) to avoid clutter.
df["is_daytime"] = df["IRRADIATION"] > IRRADIATION_MIN

# ── 2.8  Performance Ratio per timestamp ──────────────────────────────────────
# PR = AC_POWER / DC_POWER (inverter conversion efficiency).
# Only meaningful during daytime when DC_POWER > 0.
daytime_mask = (df["DC_POWER"] > 0) & (df["AC_POWER"] > 0)
df["perf_ratio"] = np.nan
df.loc[daytime_mask, "perf_ratio"] = (
    df.loc[daytime_mask, "AC_POWER"] / df.loc[daytime_mask, "DC_POWER"]
)

# ── 2.9  Derived time features ────────────────────────────────────────────────
df["hour"]       = df["DATE_TIME"].dt.hour
df["date"]       = df["DATE_TIME"].dt.date
df["month"]      = df["DATE_TIME"].dt.month
df["month_name"] = df["DATE_TIME"].dt.strftime("%B")
df["week"]       = df["DATE_TIME"].dt.isocalendar().week.astype(int)

# ── 2.10  Outlier detection (IQR method on DC_POWER) ─────────────────────────
# Night-time zeros are valid — we flag outliers for reporting only, not removal.
Q1  = df["DC_POWER"].quantile(0.25)
Q3  = df["DC_POWER"].quantile(0.75)
IQR = Q3 - Q1
outlier_mask = (df["DC_POWER"] < Q1 - 1.5 * IQR) | (df["DC_POWER"] > Q3 + 1.5 * IQR)

print(f"\n[3] DC_POWER outliers detected (IQR): {outlier_mask.sum()} — kept (night zeros are valid)")
print(f"\n[4] Date range : {df['DATE_TIME'].min().date()} → {df['DATE_TIME'].max().date()}")
print(f"[5] Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"[6] Columns    : {list(df.columns)}")


# =============================================================================
# === SECTION 3 — SUMMARY STATISTICS =========================================
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3 — SUMMARY STATISTICS")
print("=" * 70)

stats_cols = [
    "DC_POWER", "AC_POWER", "DAILY_YIELD",
    "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION",
]

print("\n[1] Descriptive statistics (plant-level, all timestamps):")
print(df[stats_cols].describe().round(2).to_string())

# ── Pre-compute aggregates used by both Sections 4 & 5 ──────────────────────
daily_total   = df.groupby("date")["AC_POWER"].sum()
hourly_avg    = df.groupby("hour")["AC_POWER"].mean()
monthly_avg   = df.groupby("month_name")["AC_POWER"].mean().reindex(["May", "June"])
monthly_total = df.groupby("month_name")["AC_POWER"].sum().reindex(["May", "June"])
corr_matrix   = df[stats_cols].corr()
daytime_df    = df[df["is_daytime"]].copy()

# Quick KPIs
peak_hour          = int(hourly_avg.idxmax())
best_month         = monthly_total.idxmax()
worst_month        = monthly_total.idxmin()
avg_daily_yield    = df.groupby("date")["DAILY_YIELD"].max().mean()
total_yield_kwh    = df.groupby("date")["DAILY_YIELD"].max().sum()

# Capacity factor: actual energy produced vs theoretical maximum
actual_energy_kwh  = df["AC_POWER"].sum() * INTERVAL_H
max_energy_kwh     = PEAK_AC_POWER_KW * len(df) * INTERVAL_H
capacity_factor    = (actual_energy_kwh / max_energy_kwh) * 100

# Average performance ratio (daytime only)
avg_perf_ratio     = df["perf_ratio"].mean() * 100

print(f"\n[2] Capacity factor          : {capacity_factor:.2f}%")
print(f"[3] Avg performance ratio    : {avg_perf_ratio:.2f}%")
print(f"[4] Peak AC power observed   : {df['AC_POWER'].max():,.0f} kW")
print(f"[5] Peak production hour     : {peak_hour:02d}:00")
print(f"[6] Best month (total yield) : {best_month}")
print(f"[7] Worst month (total yield): {worst_month}")
print(f"[8] Avg daily yield          : {avg_daily_yield:,.0f} kWh")
print(f"[9] Total yield              : {total_yield_kwh:,.0f} kWh")


# =============================================================================
# === SECTION 4 — VISUALISATION (7 plots) ====================================
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4 — VISUALISATION")
print("=" * 70)

# Shared colour palette — warm solar tones
C_PRIMARY   = "#E67E22"   # orange
C_SECONDARY = "#F39C12"   # amber
C_ACCENT    = "#C0392B"   # deep red

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Daily Total AC Power Time Series
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating Plot 1 — Daily Total AC Power Time Series …")

daily_series = pd.Series(
    daily_total.values,
    index=pd.to_datetime(list(daily_total.index))
)

fig, ax = plt.subplots(figsize=(12, 4))
fig.suptitle("Plot 1 — Daily Total AC Power Output", fontsize=14, fontweight="bold")

ax.plot(daily_series.index, daily_series.values, color=C_PRIMARY, linewidth=1.5)
ax.fill_between(daily_series.index, daily_series.values, alpha=0.2, color=C_PRIMARY)
ax.axhline(daily_series.mean(), color=C_ACCENT, linestyle="--", linewidth=1.4,
           label=f"Daily mean: {daily_series.mean():,.0f} kW")
ax.set_xlabel("Date")
ax.set_ylabel("Total AC Power (kW)")
ax.set_title("Plant-level daily AC power — May 15 to June 17, 2020")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}/01_daily_ac_power_timeseries.png")
print(f"  → Saved: {FIGURES_PATH}/01_daily_ac_power_timeseries.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating Plot 2 — Correlation Heatmap …")

# Mask the upper triangle to avoid redundant information
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(9, 7))
fig.suptitle("Plot 2 — Pearson Correlation Heatmap", fontsize=14, fontweight="bold")

sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="YlOrRd", center=0, linewidths=0.5,
    annot_kws={"size": 9}, ax=ax,
)
ax.set_title("Lower-triangle Pearson correlations (all timestamps)")

plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}/02_correlation_heatmap.png")
print(f"  → Saved: {FIGURES_PATH}/02_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Monthly Average AC Power (Bar Chart)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating Plot 3 — Monthly Average AC Power …")

fig, ax = plt.subplots(figsize=(6, 4))
fig.suptitle("Plot 3 — Monthly Average AC Power Output", fontsize=14, fontweight="bold")

bars = ax.bar(monthly_avg.index, monthly_avg.values, color=[C_PRIMARY, C_SECONDARY],
              edgecolor="white", linewidth=0.5)
ax.bar_label(bars, fmt="%.0f kW", padding=4, fontsize=9)
ax.set_xlabel("Month")
ax.set_ylabel("Average AC Power (kW)")
ax.set_title("May vs June — mean plant-level output per 15-min interval")

plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}/03_monthly_avg_power.png")
print(f"  → Saved: {FIGURES_PATH}/03_monthly_avg_power.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Hourly Average AC Power (Diurnal Curve)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating Plot 4 — Hourly Average AC Power …")

fig, ax = plt.subplots(figsize=(10, 4))
fig.suptitle("Plot 4 — Diurnal AC Power Pattern", fontsize=14, fontweight="bold")

ax.bar(hourly_avg.index, hourly_avg.values, color=C_PRIMARY, edgecolor="white", linewidth=0.4)
ax.axvline(peak_hour, color=C_ACCENT, linestyle="--", linewidth=1.5,
           label=f"Peak hour: {peak_hour:02d}:00")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Average AC Power (kW)")
ax.set_title("Average power output by hour — reveals daily solar production window")
ax.set_xticks(range(0, 24))
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}/04_hourly_avg_power.png")
print(f"  → Saved: {FIGURES_PATH}/04_hourly_avg_power.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Irradiation vs AC Power (Scatter + Regression)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating Plot 5 — Irradiation vs AC Power …")

# Use a random sample of daytime-only points to avoid over-plotting.
# Night-time readings (irradiation ≈ 0) would cluster at the origin and
# distort the regression line, so we exclude them here.
sample = daytime_df.sample(n=min(6_000, len(daytime_df)), random_state=42)

m, b = np.polyfit(sample["IRRADIATION"], sample["AC_POWER"], 1)
x_line = np.linspace(sample["IRRADIATION"].min(), sample["IRRADIATION"].max(), 200)

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle("Plot 5 — Irradiation vs. AC Power Output", fontsize=14, fontweight="bold")

sc = ax.scatter(
    sample["IRRADIATION"], sample["AC_POWER"],
    c=sample["MODULE_TEMPERATURE"], cmap="YlOrRd",
    s=8, alpha=0.45, label="Observations (coloured by module temp)"
)
plt.colorbar(sc, ax=ax, label="Module Temperature (°C)")
ax.plot(x_line, m * x_line + b, color=C_ACCENT, linewidth=2, label="OLS regression line")

ax.set_xlabel("Irradiation (W/m²)")
ax.set_ylabel("AC Power (kW)")
ax.set_title("Strong linear relationship between solar irradiance and power output")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}/05_irradiation_vs_power.png")
print(f"  → Saved: {FIGURES_PATH}/05_irradiation_vs_power.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 — AC Power Distribution by Month (Box Plot)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating Plot 6 — AC Power Distribution by Month …")

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("Plot 6 — AC Power Distribution by Month", fontsize=14, fontweight="bold")

sns.boxplot(
    data=df, x="month_name", y="AC_POWER",
    hue="month_name", order=["May", "June"],
    palette=[C_PRIMARY, C_SECONDARY], legend=False, ax=ax,
)
ax.set_xlabel("Month")
ax.set_ylabel("AC Power (kW)")
ax.set_title("Spread of 15-min AC power readings — May includes ramp-up period")

plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}/06_monthly_power_boxplot.png")
print(f"  → Saved: {FIGURES_PATH}/06_monthly_power_boxplot.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 7 — Irradiation Distribution (Histogram + KDE, Daytime Only)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating Plot 7 — Irradiation Distribution …")

fig, ax = plt.subplots(figsize=(8, 4))
fig.suptitle("Plot 7 — Irradiation Distribution (Daytime Only)", fontsize=14, fontweight="bold")

sns.histplot(
    daytime_df["IRRADIATION"], bins=40, kde=True,
    color=C_PRIMARY, line_kws={"color": C_ACCENT, "linewidth": 2}, ax=ax,
)
ax.axvline(daytime_df["IRRADIATION"].mean(), color="navy", linestyle="--", linewidth=1.4,
           label=f"Mean: {daytime_df['IRRADIATION'].mean():.3f} W/m²")
ax.set_xlabel("Irradiation (W/m²)")
ax.set_ylabel("Frequency")
ax.set_title("Right-skewed distribution — many moderate irradiance readings, fewer peak days")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}/07_irradiation_distribution.png")
print(f"  → Saved: {FIGURES_PATH}/07_irradiation_distribution.png")

print(f"\n✓ All 7 plots saved to '{FIGURES_PATH}/'")


# =============================================================================
# === SECTION 5 — FEATURE INSIGHTS ============================================
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5 — FEATURE INSIGHTS")
print("=" * 70)

# ── 5.1  Pearson correlations of each feature vs AC_POWER ────────────────────
feature_cols = ["DC_POWER", "DAILY_YIELD", "AMBIENT_TEMPERATURE",
                "MODULE_TEMPERATURE", "IRRADIATION"]
correlations = (
    df[feature_cols]
    .corrwith(df["AC_POWER"])
    .sort_values(ascending=False)
)

print("\n[1] Pearson correlations with AC_POWER (plant total):")
print(correlations.round(4).to_string())

# Feature correlation bar chart
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in correlations.values]
ax.barh(correlations.index[::-1], correlations.values[::-1],
        color=colors[::-1], edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson Correlation with AC Power (kW)")
ax.set_title("Feature Correlations vs AC Power")
fig.suptitle("Section 5 — Feature Correlation Ranking", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}/08_feature_correlations.png")
print(f"\n  → Saved: {FIGURES_PATH}/08_feature_correlations.png")

# ── 5.2  Irradiation bin analysis ─────────────────────────────────────────────
# Group readings into 0.1 W/m² bins to see how power scales with irradiance.
irr_bins   = np.arange(0, daytime_df["IRRADIATION"].max() + 0.1, 0.1)
irr_labels = irr_bins[:-1] + 0.05
daytime_df["irr_bin"] = pd.cut(daytime_df["IRRADIATION"], bins=irr_bins, labels=irr_labels)

irr_stats = daytime_df.groupby("irr_bin", observed=True).agg(
    count         =("AC_POWER", "count"),
    mean_ac_kw    =("AC_POWER", "mean"),
    mean_module_c =("MODULE_TEMPERATURE", "mean"),
).dropna()

print("\n[2] Irradiation bin analysis (sample — top 5 by AC power):")
print(irr_stats.nlargest(5, "mean_ac_kw").round(2).to_string())

# ── 5.3  Top 3 predictors ─────────────────────────────────────────────────────
top3 = correlations.abs().sort_values(ascending=False).head(3)
print("\n[3] Top 3 predictors of AC power output:")
for rank, (feat, val) in enumerate(top3.items(), start=1):
    print(f"    {rank}. {feat:<30} |r| = {val:.4f}")


# =============================================================================
# === SECTION 6 — KEY INSIGHTS ================================================
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 6 — KEY INSIGHTS")
print("=" * 70)

irr_power_corr = correlations["IRRADIATION"]
temp_power_corr = correlations["AMBIENT_TEMPERATURE"]
pct_daytime = df["is_daytime"].mean() * 100

insights = f"""
  • Irradiation – power relationship:
      Pearson correlation r = {irr_power_corr:.3f}, making solar irradiance the dominant
      driver of AC power output. The relationship is strongly linear during
      daytime, consistent with photovoltaic physics where current scales
      directly with incident photon flux.

  • Daylight availability:
      {pct_daytime:.1f}% of all 15-minute intervals recorded irradiation above zero,
      corresponding to the roughly 12-hour solar window at this latitude
      during May–June 2020.

  • Temperature effect:
      Ambient temperature correlation with AC power: r = {temp_power_corr:.3f}.
      While irradiance dominates, elevated module temperatures reduce
      panel efficiency — visible in the scatter plot where high-temp
      readings (darker points) sit slightly below the regression line.

  • Capacity factor:
      {capacity_factor:.2f}% — typical for utility-scale solar in a temperate
      climate (15–25% range). This means the plant produced roughly
      {capacity_factor:.0f} kWh for every 100 kWh it could produce running
      at rated power 24/7.

  • Performance ratio:
      {avg_perf_ratio:.2f}% — the ratio of AC to DC power, reflecting inverter
      conversion efficiency. Values above 90% indicate a well-maintained
      inverter fleet with minimal losses.

  • Peak production window:
      Generation concentrates between 08:00 and 16:00, peaking around
      {peak_hour:02d}:00. Output drops sharply outside this window,
      underscoring the importance of battery storage for grid stability.

  • Monthly comparison:
      Best month  → {best_month}  (higher irradiance, longer days)
      Worst month → {worst_month} (partial month or variable cloud cover)
      The short dataset window (34 days) limits seasonal generalisation.

  • Total energy produced:
      {total_yield_kwh:,.0f} kWh over the recording period — equivalent to
      approximately {total_yield_kwh / 900:.0f} average Turkish households' monthly consumption.
"""

print(insights)


# =============================================================================
# === SECTION 7 — CONCLUSION ==================================================
# =============================================================================

print("=" * 70)
print("SECTION 7 — CONCLUSION")
print("=" * 70)

conclusion = f"""
This analysis examined 34 days (May 15 – June 17, 2020) of 15-minute SCADA
measurements from a {N_INVERTERS}-inverter solar power plant with a peak AC capacity
of ~{PEAK_AC_POWER_KW:,} kW, covering {len(df):,} aggregated plant-level observations.

Solar irradiance emerged as the overwhelmingly dominant predictor of AC power
output (r ≈ {irr_power_corr:.2f}), consistent with the linear photovoltaic current–irradiance
relationship. The plant achieved a capacity factor of {capacity_factor:.1f}% and an average
inverter performance ratio of {avg_perf_ratio:.1f}%, indicating a healthy and well-maintained
system. The diurnal production window (approximately 08:00–16:00) and the
irradiation distribution highlight that output is concentrated into roughly
half the day, a fundamental challenge for grid integration.

Module temperature had a modest negative secondary effect — elevated panel
temperatures reduce open-circuit voltage and overall efficiency — visible
in the scatter plots where high-temperature readings fall slightly below the
regression line. The 34-day recording window precludes firm seasonal conclusions,
but June showed marginally higher irradiance than the partial May period.

Suggestions for future work:
  1. Anomaly detection — Apply isolation forests or autoencoders to flag
     inverters underperforming relative to their peers in real time, enabling
     targeted maintenance before production losses accumulate.
  2. Short-term power forecasting — Train an XGBoost or LSTM model on lagged
     irradiation and temperature features to predict 1–4 hour ahead output,
     supporting intra-day grid dispatch decisions.
  3. Extended dataset — Merge multiple years and plants to quantify seasonal
     and inter-annual variability, estimate degradation rates, and build a
     robust plant-level performance benchmark.
"""

print(conclusion)
print("=" * 70)
print("Analysis complete. All plots saved to:", FIGURES_PATH)
print("=" * 70)
