#!/usr/bin/env python
# coding: utf-8

# ### To what extent do demand, wind, solar, interconnector flows, and system constraints explain and predict short-term price spreads between consecutive I-SEM markets, and how can this predictive power be converted into a robust arbitrage trading strategy?

# ## Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mticker
import datetime as dt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, roc_auc_score, r2_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  GradientBoostingClassifier, GradientBoostingRegressor
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import TimeSeriesSplit

try:
    import lightgbm as lgb
    HAS_LGB = True
    print("LightGBM available")
except ImportError:
    HAS_LGB = False
    print("LightGBM not found")

print("All imports OK")


# ## Checking & Tidying the data

# ### Initial Diagnostics & Duplicates Removal

# In[2]:


# Converting the parquet file to CSV
marketData = pd.read_parquet('MarketData_2022-2026.parquet')
marketData.to_csv('MarketData.csv')


# In[3]:


# Loading the CSV file
data = pd.read_csv('MarketData.csv')
data = data.drop(columns='Unnamed: 0')
data.head(10)


# In[4]:


data.tail(10)


# In[5]:


# Data Shape 
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])


# In[6]:


# Column Data Types
for col, dtype in data.dtypes.items():
    print(col, ":", dtype)


# In[7]:


# Duplicate Check
n_dup_rows = data.duplicated().sum()
print("Duplicated rows:", int(n_dup_rows))


# In[8]:


# Drop duplicate observations
before = len(data)
data = data.drop_duplicates()
after = len(data)
print("Rows removed:", before - after)


# Loaded the dataset and inspected its shape, column names, and data types to understand the structure of market variables (prices, forecasts, etc.). The dataset contains many float-valued market and forecast columns. Full-row duplicates were removed to ensure each remaining observation represents a unique data record.

# ### DateTime Consistency

# In[9]:


# Parsing datetime columns 
data["TradeDate"] = pd.to_datetime(data["TradeDate"], errors="coerce")
data["Time"] = pd.to_datetime(data["Time"], format="%H:%M:%S", errors="coerce").dt.time
data["StartDateTime"] = pd.to_datetime(data["StartDateTime"], errors="coerce")
data = data.copy()

# Delivery Timestamp Column
data["ts"] = pd.to_datetime(data["TradeDate"].astype(str) + " " + data["Time"].astype(str), errors="coerce")


# In[10]:


print("Unparsed delivery timestamps:", data["ts"].isna().sum())

# Sort by delivery time
data = data.sort_values("ts").reset_index(drop=True)

# Delivery-time features
data["hour"] = data["ts"].dt.hour
data["minute"] = data["ts"].dt.minute
data["hod"] = data["hour"] + data["minute"] / 60


# In[11]:


data.head(5)


# In[12]:


# Checking for duplicate timestamps
n_dup_ts = data['ts'].duplicated().sum()
print("Duplicate timestamps:", int(n_dup_ts))

n_full_dup = data.duplicated().sum()
print("Entire duplicate rows:", n_full_dup)


# In[13]:


dup = data[data["ts"].duplicated(keep=False)].sort_values("ts")
dup.tail(10)


# In[14]:


# Columns that are identical across duplicate timestamps
base_cols = [c for c in data.columns if c not in ["ts", "PredictedICFlow"]]     # it was observed that the rows with duplicate timestamps were entirely similar except for 'PedictedICFlow'

base = (
    data
    .groupby("ts", as_index=False)[base_cols]
    .first()
)

# Aggregate PredictedICFlow separately (NaN-safe)
icflow = (
    data
    .groupby("ts", as_index=False)["PredictedICFlow"]
    .mean()
)

data = (base.merge(icflow, on="ts", how="left"))

data = data.copy()

print("Duplicate timestamps:", data["ts"].duplicated().sum())
print("Rows:", len(data))


# In[15]:


# Checking that the dataset contains a continuous half-hour delivery timeline
expected_ts = pd.date_range(
    start=data["ts"].min(),
    end=data["ts"].max(),
    freq="30min"
)

print("Expected half-hours:", len(expected_ts))
print("Actual rows:", len(data))

actual_ts = pd.Index(data["ts"])
missing_ts = expected_ts.difference(actual_ts)

print("Missing half-hours:", len(missing_ts))


# Electricity market analysis is fundamentally time-indexed: each row must map to an unambiguous delivery half-hour. Time features are essential because spreads and volatility are typically intraday-patterned. We parsed `TradeDate` and `Time` into appropriate datetime formats and constructed a single delivery timestamp `ts` using `TradeDate + Time`. Then we sorted observations by `ts` and created delivery-time features (`hour`, `minute`, `hod`) for later analysis. 
# 
# Duplicate timestamps must be resolved because they violate the “one observation per delivery interval” structure expected for modelling. So we checked for duplicate delivery timestamps and investigated them. Some delivery timestamps appear in duplicate pairs. Inspection showed the paired rows are identical across all columns except `PredictedICFlow`. To restore a single observation per delivery interval, we kept one record for all stable columns and aggregated `PredictedICFlow` using a NaN-safe mean. After this step, each `ts` is unique, which makes the dataset consistent for EDA and modelling.

# ### Data Validity Check

# In[16]:


num_cols = data.select_dtypes(include=[np.number]).columns.tolist()

# Define which columns should be non-negative
NONNEG_KEYWORDS = ["WIND", "SOLAR", "DEMAND", "LOAD", "FORECAST", "METER"]

nonneg_cols = [
    c for c in num_cols
    if any(k in c.upper() for k in NONNEG_KEYWORDS)
    and not any(x in c.upper() for x in ["FLOW", "NIV", "PRICE", "SPREAD", "IMBAL"])
]

results = []

for col in nonneg_cols:
    s = pd.to_numeric(data[col], errors="coerce")
    non_missing = s.notna().sum()
    bad = (s < 0).sum()

    results.append({
        "column": col,
        "non_missing_values": int(non_missing),
        "negative_values": int(bad),
        "proportion_negative": round(bad / non_missing if non_missing > 0 else np.nan, 6)
    })

validity_summary = pd.DataFrame(results).sort_values(
    "proportion_negative", ascending=False
)

display(validity_summary)


# In[17]:


neg_sorted = (
    data.loc[data["EirgridActualWind"] < 0, ["ts", "EirgridActualWind"]]
        .sort_values(by="EirgridActualWind")
)

neg_sorted.head(10)


# In[18]:


neg_mask = data["EirgridActualWind"] < 0
print("Negative EirgridActualWind rows:", neg_mask.sum())

# Clip to zero
data.loc[neg_mask, "EirgridActualWind"] = 0


# We performed basic validity checks to ensure values are consistent with the market/physical constraints (e.g., demand and wind should not be negative, while prices and net positions may be negative). Negative values were observed only in EirgridActualWind but in very less proportion (0.000086%). These values were clipped to zero, preserving observations while ensuring physical consistency.

# ### Missing Data Analysis

# In[19]:


# Top missing columns
n = len(data)
data = data.replace(" ", np.nan)
miss = data.isna().sum().sort_values(ascending=False)
miss_pct = (miss / n * 100).round(2)
report = pd.DataFrame({"missing_count": miss, "missing_%": miss_pct})
display(report.head(20))


# In[20]:


# MISSINGNESS HEATMAP 
data0 = data.sort_values('ts')
cols = [
    "ts","PriceDAM","PriceIDA1","PriceIDA2","PriceIDA3","PriceImbalance","AggregatedForecast",
    "EirGridDemandFc_DAM","EirGridDemandFc_IDA1","EirGridDemandFc_IDA2","EirGridDemandFc_IDA3",
    "EirGridWindFc_DAM","EirGridWindFc_IDA1","EirGridWindFc_IDA2","EirGridWindFc_IDA3",
    "ISEMSOLAR_DAM","ISEMSOLAR_IDA1","ISEMSOLAR_IDA2","ISEMSOLAR_IDA3"
    ]
sample = data0[cols].copy()
miss_matrix = sample.isna().astype(int).values
plt.figure(figsize=(5, 7))
cmap = ListedColormap(['#E0E0E0', '#D32F2F'])
plt.imshow(miss_matrix, aspect="auto", interpolation="nearest", cmap=cmap)
plt.yticks([])
plt.xticks(range(len(cols)), cols, rotation=90, fontsize=6)
plt.title("Missingness heatmap")
plt.tight_layout()
plt.show()


# In[21]:


# Must-have columns (should have NO missing values)
must_have = ["ts", "TradeDate", "Time", "hour", "minute", "hod", "EirgridActualDemand", "EirgridActualWind", "NIV_Actual", "TotalPN", 
             "Aggregated Forecast", "EirGridDemandFc_DAM", "EirGridDemandFc_IDA1"]
for c in ["PriceDAM", "PriceIDA1", "PriceImbalance"]:
    if c in data.columns:
        must_have.append(c)
must_have = [c for c in must_have if c in data.columns]

print("\nMust-have columns missing count:")
print(data[must_have].isna().sum()[lambda s: s > 0])

data['ISEMSOLAR_DAM'] = data['ISEMSOLAR_DAM'].fillna(0)
data['ISEMSOLAR_IDA1'] = data['ISEMSOLAR_IDA1'].fillna(0)
data["EirGridWindFc_DAM"] = (
    data["EirGridWindFc_DAM"]
        .fillna(data["EmSys_U_ISEMWIND_DAM"])
        .fillna(data["Meteo_ISEMWIND_DAM"])
        .fillna(0)
)
data["EirGridWindFc_IDA1"] = (
    data["EirGridWindFc_IDA1"]
        .fillna(data["EmSys_U_ISEMWIND_IDA1"])
        .fillna(data["Meteo_ISEMWIND_IDA1"])
        .fillna(0)
)

# Drop the rows with missing values
data1 = data.dropna(subset=must_have)
data1 = data1.sort_values("ts").reset_index(drop=True).copy()
print('Number of rows :', len(data1))


# In[22]:


# Structural missingness check for IDA2/IDA3 
IDA2_WINDOW = (11.0, 23.0)  # 11:00–23:00
IDA3_WINDOW = (17.0, 23.0)  # 17:00–23:00

ida2_should_exist = (data1["hod"] >= IDA2_WINDOW[0]) & (data1["hod"] < IDA2_WINDOW[1])
ida3_should_exist = (data1["hod"] >= IDA3_WINDOW[0]) & (data1["hod"] < IDA3_WINDOW[1])

price_ida1 = "PriceIDA1" if "PriceIDA1" in data1.columns else None
price_ida2 = "PriceIDA2" if "PriceIDA2" in data1.columns else None
price_ida3 = "PriceIDA3" if "PriceIDA3" in data1.columns else None

def structural_check(df, col, should_exist_mask, name):
    if col is None:
        print(f"\n{name}: column not found.")
        return

    inside = df.loc[should_exist_mask, col]
    outside = df.loc[~should_exist_mask, col]

    # Rates
    inside_rate = inside.isna().mean()
    outside_rate = outside.isna().mean()

    # Counts
    inside_missing = inside.isna().sum()
    outside_missing = outside.isna().sum()

    inside_total = inside.shape[0]
    outside_total = outside.shape[0]

    # Non-missing OUTSIDE window (this is suspicious; should usually be 0)
    outside_nonmissing = outside.notna().sum()

    print(f"\n{name} ({col}) structural check:")
    print(f"INSIDE window (should exist):")
    print(f"  total rows      : {inside_total}")
    print(f"  missing count   : {inside_missing}")
    print(f"  missing rate    : {inside_rate:.3f}")
    print(f"OUTSIDE window (expected missing):")
    print(f"  total rows      : {outside_total}")
    print(f"  missing count   : {outside_missing}")
    print(f"  missing rate    : {outside_rate:.3f}")
    print(f"  NON-missing outside (suspicious): {outside_nonmissing}")

if price_ida1:
    all_rows = pd.Series(True, index=data1.index)
    structural_check(data1, price_ida1, all_rows, "IDA1 PRICE")
if price_ida2:
    structural_check(data1, price_ida2, ida2_should_exist, "IDA2 PRICE")
if price_ida3:
    structural_check(data1, price_ida3, ida3_should_exist, "IDA3 PRICE")

structural_check(data1, "EirGridDemandFc_IDA2", ida2_should_exist, "IDA2 DEMAND")
structural_check(data1, "EirGridWindFc_IDA2", ida2_should_exist, "IDA2 WIND")
structural_check(data1, "ISEMSOLAR_IDA2", ida2_should_exist, "IDA2 SOLAR")

structural_check(data1, "EirGridDemandFc_IDA3", ida3_should_exist, "IDA3 DEMAND")
structural_check(data1, "EirGridWindFc_IDA3", ida3_should_exist, "IDA3 WIND")
structural_check(data1, "ISEMSOLAR_IDA3", ida3_should_exist, "IDA3 SOLAR")


# In[23]:


# Solving IDA2 structural missingness
data1.loc[~ida2_should_exist, "ISEMSOLAR_IDA2"] = np.nan 
mask1_ida2_fill = ida2_should_exist & data1["ISEMSOLAR_IDA2"].isna()
data1.loc[mask1_ida2_fill, "ISEMSOLAR_IDA2"] = 0.0

# Solving IDA3 structural missingness
mask2_ida3_fill = ida3_should_exist & data1["ISEMSOLAR_IDA3"].isna()
data1.loc[mask2_ida3_fill, "ISEMSOLAR_IDA3"] = 0.0
mask3_ida3_fill = ida3_should_exist & data1["EirGridWindFc_IDA3"].isna()
data1.loc[mask3_ida3_fill, "EirGridWindFc_IDA3"] = (
    data1.loc[mask3_ida3_fill, "EmSys_U_ISEMWIND_IDA3"]
        .fillna(data1.loc[mask3_ida3_fill, "Meteo_ISEMWIND_IDA3"])
        .fillna(0.0)
)


# In[24]:


# Checking missing half-hours
expected_ts = pd.date_range(
    start=data1["ts"].min(),
    end=data1["ts"].max(),
    freq="30min"
)

missing_ts = expected_ts.difference(pd.Index(data1["ts"]))
print("Missing half-hours after cleaning:", len(missing_ts))
print("First few missing:", list(missing_ts[:5]))


# In[25]:


# MISSINGNESS HEATMAP 
data0 = data1.sort_values('ts')
cols = data0.columns
sample = data0[cols].copy()
miss_matrix = sample.isna().astype(int).values
plt.figure(figsize=(15, 6))
cmap = ListedColormap(['#E0E0E0', '#D32F2F'])
plt.imshow(miss_matrix, aspect="auto", interpolation="nearest", cmap=cmap)
plt.yticks([])
plt.xticks(range(len(cols)), cols, rotation=90, fontsize=6)
plt.title("Missingness heatmap")
plt.tight_layout()
plt.show()


# In[26]:


print('Number of rows :', len(data1))


# The dataset was examined to assess the extent and nature of missing values across all variables. First, column-wise missingness was checked to identify variables with incomplete observations and to distinguish between incidental and systematic gaps. A subset of “must-have” variables was then defined, including the delivery timestamp (ts) and core market indicators such as key prices and fundamental metrics (e.g., demand and wind). Rows lacking any of these essential variables were removed to ensure that subsequent analysis is based on economically meaningful observations. Importantly, missingness was evaluated in the context of market structure. Certain intra-day auction prices (e.g., IDA2 and IDA3) are only defined within specific delivery windows. Therefore, missing values outside those windows were identified as structural missingness rather than data quality issues. Dedicated checks confirmed that IDA price columns were populated within their expected delivery intervals and absent outside them, consistent with market rules. This distinction ensures that structurally valid gaps are not misinterpreted as data errors. 

# ### Outliers Diagnostics

# In[27]:


outlier_cols = [
    c for c in [
        "PriceDAM", "PriceIDA1", "PriceIDA2", "PriceIDA3", "PriceImbalance", "EirGridDemandFc_DAM", "EirGridDemandFc_IDA1", "EirGridDemandFc_IDA2", "EirGridDemandFc_IDA3"
    ] if c in data1.columns
]

def outlier_count(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    min = s.min()
    max = s.max()
    skew = s.skew()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr
    return {
        "minimum": min,
        "maximum": max,
        "skewness": skew,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": ((s < lower) | (s > upper)).sum(),
        "outlier_pct": ((s < lower) | (s > upper)).mean() * 100
    }

report = pd.DataFrame(
    {c: outlier_count(data1[c]) for c in outlier_cols}
).T

print("\nOutlier Report:")
display(report.sort_values("outlier_pct", ascending=False))


# Given the heavy-tailed and skewed nature of electricity prices, a relaxed interquartile threshold (3×IQR) was used for outlier identification. The analysis revealed that extreme observations constitute a small proportion of the dataset, indicating that most variables behave within expected bounds. Crucially, outliers were not automatically removed, as extreme price and imbalance events can be economically meaningful. Instead, this diagnostic step serves to characterise tail behaviour and aid modelling decisions. The low proportion of extreme values suggests that the dataset is not dominated by anomalous spikes, supporting its suitability for further exploratory and predictive analysis.

# ### Final Data Integrity Checklist

# In[28]:


# Ensure strict order of time
data1 = data1.sort_values("ts").reset_index(drop=True)
print((data1["ts"].diff().dropna() < pd.Timedelta(0)).sum())


# In[29]:


# Final Check for Duplicate Timestamps
print(data1["ts"].duplicated().sum())


# In[30]:


# Check for Infinite Values
print(np.isinf(data1.select_dtypes(include=[np.number])).sum().sum())


# In[31]:


data1 = data1.drop(columns=["Time"])


# In[32]:


data1.dtypes.value_counts()


# In[33]:


data1 = data1.reset_index(drop=True)
print(data1.shape)
print("Date range:", data1["ts"].min(), "→", data1["ts"].max())


# In[34]:


for col, dtype in data1.dtypes.items():
    print(col, ":", dtype)


# In[35]:


data1.head(5)


# Following cleaning, validation, and diagnostic checks, a final data integrity review was performed to ensure the dataset meets the structural requirements for analysis. The delivery timestamp (ts) was verified to be unique and strictly ordered, confirming that each row represents a single, well-defined delivery interval. All numeric variables were checked for infinite values and inappropriate data types to maintain a clean data structure suitable for modelling. At this stage, the dataset satisfies the core conditions for time-indexed market analysis: one observation per delivery interval, economically interpretable variable values, consistent timestamps, and essential market indicators. The data is therefore structurally prepared for Exploratory Data Analysis.

# ## Exploratory Data Analysis

# ### Brief Data Summary

# In[36]:


# Univariate & Multivariate Summaries for selected columns
key_cols = [c for c in [
    "PriceDAM","PriceIDA1","PriceIDA2","PriceIDA3","PriceImbalance",
    "EirGridDemandFc_DAM","EirGridDemandFc_IDA1","EirGridDemandFc_IDA2","EirGridDemandFc_IDA3",
    "EirGridWindFc_DAM","EirGridWindFc_IDA1","EirGridWindFc_IDA2","EirGridWindFc_IDA3","NIV_Actual","TotalPN"
] if c in data1.columns]

display(data1[key_cols].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T)

corr = data1[key_cols].corr()
plt.figure(figsize=(10,8))
im = plt.imshow(corr, aspect="auto")

plt.xticks(range(len(key_cols)), key_cols, rotation=45, ha="right")
plt.yticks(range(len(key_cols)), key_cols)

plt.title("Correlation Matrix (Key Variables)")
plt.colorbar()

for i in range(len(corr)):
    for j in range(len(corr)):
        value = corr.iloc[i, j]
        plt.text(
            j, i,
            f"{value:.2f}",
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()


# ### Define & Analyse Necessary Features for Market

# In[37]:


df = data1.copy()
df = df.sort_values("ts").reset_index(drop=True)

# Core price columns 
prices = [c for c in ["PriceDAM","PriceIDA1","PriceIDA2","PriceIDA3","PriceImbalance"] if c in df.columns]
print("Price cols:", prices)

# Define key spreads
if "PriceDAM" in df.columns and "PriceIDA1" in df.columns:
    df["spr_IDA1_DAM"] = df["PriceIDA1"] - df["PriceDAM"]

if "PriceIDA1" in df.columns and "PriceImbalance" in df.columns:
    df["spr_BM_IDA1"] = df["PriceImbalance"] - df["PriceIDA1"]

if "PriceDAM" in df.columns and "PriceImbalance" in df.columns:
    df["spr_BM_DAM"] = df["PriceImbalance"] - df["PriceDAM"]

if "PriceIDA1" in df.columns and "PriceIDA2" in df.columns:
    df["spr_IDA2_IDA1"] = df["PriceIDA2"] - df["PriceIDA1"]

if "PriceIDA2" in df.columns and "PriceIDA3" in df.columns:
    df["spr_IDA3_IDA2"] = df["PriceIDA3"] - df["PriceIDA2"]

if "PriceIDA2" in df.columns and "PriceImbalance" in df.columns:
    df["spr_BM_IDA2"] = df["PriceImbalance"] - df["PriceIDA2"]

if "PriceIDA3" in df.columns and "PriceImbalance" in df.columns:
    df["spr_BM_IDA3"] = df["PriceImbalance"] - df["PriceIDA3"]

if "PriceDAM" in df.columns and "PriceImbalance" in df.columns:
    df["spr_BM_DAM"] = df["PriceImbalance"] - df["PriceDAM"]

# Direction labels 
for s in [c for c in df.columns if c.startswith("spr_")]:
    df[f"dir_{s}"] = np.where(df[s] >= 0, 1, -1)

print("Spread cols:", [c for c in df.columns if c.startswith("spr_")])


# In[38]:


plt.figure(figsize=(5,4))

plt.hist(df['PriceImbalance'], bins=80, alpha=0.5, label='BM')
plt.hist(df['PriceIDA1'], bins=80, alpha=0.5, label='IDA1')
plt.hist(df['PriceDAM'], bins=80, alpha=0.5, label='DAM')

plt.title("Distribution of Market Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()


# In[39]:


plt.figure(figsize=(5,4))

plt.hist(df['spr_BM_DAM'], bins=80, alpha=0.6, label='BM_DAM')
plt.hist(df['spr_BM_IDA1'], bins=80, alpha=0.6, label='BM_IDA1')
plt.hist(df['spr_IDA1_DAM'], bins=80, alpha=0.6, label='IDA1_DAM')

plt.title("Distribution of Market Price Spreads")
plt.xlabel("Price Spread")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()


# To begin, the distributions of market prices and the calculated spreads between consecutive markets were examined. Histograms and summary statistics show that price levels vary considerably over time, with occasional extreme values. However, the spreads between markets are generally centred around zero, meaning that on average there is no guaranteed arbitrage opportunity. Instead, profitable opportunities arise when price spreads temporarily diverge.
# 
# The presence of long tails in the spread distributions indicates that while most delivery intervals show small differences, a small number of intervals exhibit much larger deviations. This suggests that potential profits may be concentrated in specific periods rather than evenly distributed across time. Therefore, understanding when and why these larger spreads occur is central to the our objective.

# In[40]:


# Intraday Price-Spread Profiles
core_spreads = [
    "spr_IDA1_DAM",
    "spr_BM_IDA1",
    "spr_BM_DAM"
]

cascade_spreads = [
    "spr_IDA2_IDA1",
    "spr_IDA3_IDA2",
    "spr_BM_IDA2",
    "spr_BM_IDA3"
]

ticks1 = np.arange(0, 25, 3)
ticks2 = np.arange(11, 25, 2)

plt.figure(figsize=(12,5))
for col in core_spreads:
    if col in df.columns:
        mean_profile = df.groupby("hod")[col].mean()
        plt.plot(mean_profile.index, mean_profile.values, linewidth=2, label=col)
plt.xlabel("Hour of Day")
plt.ylabel("Mean Spread (€/MWh)")
plt.title("Intraday Mean Spread — Core Market Transitions")
plt.legend()
plt.xticks(ticks=ticks1, labels=[f"{int(t):02d}:00" for t in ticks1])
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,5))
for col in cascade_spreads:
    if col in df.columns:
        mean_profile = df.groupby("hod")[col].mean()
        plt.plot(mean_profile.index, mean_profile.values, linewidth=2, label=col)
plt.xlabel("Hour of Day")
plt.ylabel("Mean Spread (€/MWh)")
plt.title("Intraday Mean Spread — Intraday Adjustments")
plt.legend()
plt.xticks(ticks=ticks2, labels=[f"{int(t):02d}:00" for t in ticks2])
plt.tight_layout()
plt.show()


# Spreads were then analysed by hour of day to determine whether arbitrage opportunities follow a predictable daily pattern. The results show that spreads vary systematically throughout the day. In particular, certain hours — typically those associated with higher system demand — exhibit larger average spreads and greater volatility.This finding is consistent with economic intuition: during peak demand periods or when renewable generation is low, system balancing becomes more challenging, leading to greater price differences across markets. These intraday patterns suggest that time-of-day is an important explanatory factor for spread behaviour and may be useful in designing a predictive model.

# In[41]:


# Direction Balance
def direction_balance(spread_col, mask=None):
    d = df[[spread_col]].copy()
    if mask is not None:
        d = d.loc[mask]
    s = pd.to_numeric(d[spread_col], errors="coerce").dropna()
    if s.empty:
        print("No data:", spread_col); return
    out = pd.Series({
        "neg_%": (s < 0).mean(),
        "zero_%": (s == 0).mean(),
        "pos_%": (s > 0).mean(),
        "n": len(s)
    })
    return out

bal = []
for c in [c for c in df.columns if c.startswith("dir_spr_")]:
    bal.append(direction_balance(c))
display(pd.DataFrame(bal, index=[c for c in df.columns if c.startswith("spr_")]))


# The proportion of positive versus negative spreads was analysed to understand whether one direction dominates. Results indicate that spreads are reasonably balanced between positive and negative values. This means that simply always predicting one direction would not yield consistently high accuracy. This balance confirms that spread direction is not trivial to predict and likely depends on underlying system conditions. As a result, modelling efforts must rely on meaningful explanatory variables such as demand, wind generation, and system imbalance rather than frequency alone.

# In[42]:


# Fundamentals vs spreads
fundamentals = [c for c in [
    "EirGridDemandFc_DAM","EirgridActualWind","NIV_Actual","ActualMeterData","TotalPN"
] if c in df.columns]
print("Fundamentals:", fundamentals)

# Net demand = Demand - Wind - Solar
df["NetDemand_DAM"] = df["EirGridDemandFc_DAM"] - df["EirGridWindFc_DAM"] - df["ISEMSOLAR_DAM"] 
df["NetDemand_IDA1"] = df["EirGridDemandFc_IDA1"] - df["EirGridWindFc_IDA1"] - df["ISEMSOLAR_IDA1"] 
df["NetDemand_IDA2"] = df["EirGridDemandFc_IDA2"] - df["EirGridWindFc_IDA2"] - df["ISEMSOLAR_IDA2"] 
df["NetDemand_IDA3"] = df["EirGridDemandFc_IDA3"] - df["EirGridWindFc_IDA3"] - df["ISEMSOLAR_IDA3"] 

for c in [c for c in df.columns if c.startswith("NetDemand_")]:
    print(df[c].describe())


# In[43]:


# Scatterplots 
def scatter(x, y, n=8000, title=None):
    if x not in df.columns or y not in df.columns:
        return

    d = df[[x, y]].copy()
    d[x] = pd.to_numeric(d[x], errors="coerce")
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d.dropna()
    if d.empty:
        return

    if len(d) > n:
        d = d.sample(n, random_state=42)

    xvals = d[x].values
    yvals = d[y].values

    coeffs = np.polyfit(xvals, yvals, 1) 
    poly = np.poly1d(coeffs)

    xs = np.linspace(xvals.min(), xvals.max(), 200)
    ys = poly(xs)

    plt.figure(figsize=(6,5))
    plt.scatter(xvals, yvals, s=6, alpha=0.35)
    plt.plot(xs, ys, linewidth=2, color='red')

    plt.title(title or f"{y} vs {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.show()

scatter("NetDemand_DAM", "spr_IDA1_DAM")
scatter("EirGridDemandFc_DAM", "spr_IDA1_DAM")
scatter("EirGridWindFC_DAM", "spr_IDA1_DAM")
scatter("ISEMSOLAR_DAM", "spr_IDA1_DAM")
scatter("GB DAM Epex", "spr_IDA1_DAM")
scatter("GB DAM N2EX", "spr_IDA1_DAM")
scatter("GB DAM HH Epex", "spr_IDA1_DAM")

scatter("NetDemand_DAM", "spr_BM_DAM")
scatter("EirGridDemandFc_DAM", "spr_BM_DAM")
scatter("EirGridWindFC_DAM", "spr_BM_DAM")
scatter("ISEMSOLAR_DAM", "spr_BM_DAM")
scatter("GB DAM Epex", "spr_BM_DAM")
scatter("GB DAM N2EX", "spr_BM_DAM")
scatter("GB DAM HH Epex", "spr_BM_DAM")

scatter("NetDemand_IDA1", "spr_BM_IDA1")
scatter("EirGridDemandFc_IDA1", "spr_BM_IDA1")
scatter("EirGridWindFC_IDA1", "spr_BM_IDA1")
scatter("ISEMSOLAR_IDA1", "spr_BM_IDA1")
scatter("PredictedICFlow", "spr_BM_IDA1")
scatter("PumpStorage", "spr_BM_IDA1")


# ### Updated Correlation Map

# In[44]:


spread_cols = [c for c in df.columns if c.startswith("spr_")]
nd_cols = [c for c in df.columns if c.startswith("NetDemand_")]
key_cols = [c for c in (prices + spread_cols + fundamentals + nd_cols) if c in df.columns]
key_cols = list(dict.fromkeys(key_cols))  # unique

corr = df[key_cols].corr(numeric_only=True)

plt.figure(figsize=(0.7*len(key_cols)+4, 0.7*len(key_cols)+4))
im = plt.imshow(corr, aspect="auto")
plt.xticks(range(len(key_cols)), key_cols, rotation=45, ha="right")
plt.yticks(range(len(key_cols)), key_cols)
plt.title("Correlation matrix (prices, spreads, fundamentals)")
plt.colorbar()

for i in range(len(key_cols)):
    for j in range(len(key_cols)):
        val = corr.iloc[i, j]
        color = "white" if abs(val) > 0.5 else "black"
        plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=color)

plt.tight_layout()
plt.show()


# A correlation matrix was used to examine overall relationships between prices, spreads, and key system variables. While price levels across markets are strongly correlated — as expected — spreads show weaker and more varied correlations with individual fundamentals. This suggests that no single variable fully explains spread behaviour. Instead, spreads are likely influenced by a combination of factors acting together. This reinforces the need for multivariate modelling rather than relying on a single predictor.

# ### Extended Analysis for Understanding the Data

# #### Seasonality: Are relationships stable over time?

# In[45]:


spread_cols = [c for c in df.columns if c.startswith("spr_")]
df["month"] = df["ts"].dt.to_period("M").astype(str)

monthly_mean = (
    df.groupby("month")[spread_cols]
      .mean(numeric_only=True)
)

plt.figure(figsize=(12,5))
for col in spread_cols:
    if col in monthly_mean.columns:
        plt.plot(monthly_mean.index, monthly_mean[col], label=col)

plt.xticks(rotation=45, ha="right")
plt.title("Monthly Mean of Price Spreads")
plt.xlabel("Month")
plt.ylabel("Spread (€ / MWh)")
plt.legend()
plt.tight_layout()
plt.show()

monthly_std = (
    df.groupby("month")[spread_cols]
      .std(numeric_only=True)
)

plt.figure(figsize=(12,5))
for col in spread_cols:
    if col in monthly_std.columns:
        plt.plot(monthly_std.index, monthly_std[col], label=col)

plt.xticks(rotation=45, ha="right")
plt.title("Monthly Volatility (Std) of Price Spreads")
plt.xlabel("Month")
plt.ylabel("Std Dev (€ / MWh)")
plt.legend()
plt.tight_layout()
plt.show()


# To assess longer-term stability, spreads were analysed by month. The results show that both average spreads and their standard deviation vary over time. Some months exhibit higher variability, indicating more unstable market conditions. This implies that spread behaviour is not constant throughout the year. Periods of elevated volatility may correspond to seasonal demand changes or variations in renewable output. From a trading perspective, this suggests that model performance and risk exposure may differ across time periods.

# #### Analysing the spread extremes

# In[46]:


# Spread Extremes Frequency
spread_cols = [c for c in df.columns if c.startswith("spr_")]

extreme_summary = []

for col in spread_cols:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        continue

    q_low = s.quantile(0.01)
    q_high = s.quantile(0.99)

    extreme_summary.append({
        "spread": col,
        "q01": q_low,
        "q99": q_high,
        "n_extreme_low": (s <= q_low).sum(),
        "n_extreme_high": (s >= q_high).sum(),
        "pct_extreme_total": round(((s <= q_low) | (s >= q_high)).mean(), 4)
    })

extreme_summary = pd.DataFrame(extreme_summary)
display(extreme_summary)


# In[47]:


# Intraday Distribution of Extreme Events
for col in spread_cols:
    s = pd.to_numeric(df[col], errors="coerce")
    q_low = s.quantile(0.01)
    q_high = s.quantile(0.99)

    extreme_mask = (s <= q_low) | (s >= q_high)

    d = df.loc[extreme_mask, ["hod"]].copy()

    if d.empty:
        continue

    hod_counts = d["hod"].value_counts().sort_index()

    plt.figure(figsize=(10,4))
    plt.bar(hod_counts.index, hod_counts.values)
    plt.title(f"Intraday Distribution of Extreme Events: {col}")
    plt.xlabel("Hour of Day")
    plt.ylabel("Count (Top/Bottom 1%)")
    plt.tight_layout()
    plt.show()


# In[48]:


# Monthly Distribution of Extreme Events
df["month"] = df["ts"].dt.to_period("M").astype(str)

for col in spread_cols:
    s = pd.to_numeric(df[col], errors="coerce")
    q_high = s.quantile(0.99)

    extreme_high = df[s >= q_high].groupby("month").size()

    if extreme_high.empty:
        continue

    plt.figure(figsize=(10,4))
    plt.plot(extreme_high.index, extreme_high.values)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Monthly Extreme High Spread Events: {col}")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# Finally, extreme spread events were examined by identifying observations in the top and bottom percentiles. These extreme events occur very rarely but are often concentrated in specific hours or months.
# This analysis indicates that large arbitrage opportunities are not randomly distributed. Instead, they tend to occur during periods of system stress. As a result, any trading strategy must account not only for average spread behaviour but also for the risks and opportunities associated with extreme events.

# #### Conditional Spread Profiles

# In[49]:


# IDA1_DAM & BM_IDA1 & BM_DAM Spread by Wind Quantile
df["Wind_q"] = pd.qcut(df["EirgridActualWind"], q=5, labels=False, duplicates="drop")

grouped = df.groupby("Wind_q")[["spr_IDA1_DAM","spr_BM_IDA1","spr_BM_DAM"]].mean()
display(grouped)

grouped.plot(kind="bar", figsize=(8,4))
plt.title("Average Spread by Wind Quantile")
plt.xlabel("Wind quantile")
plt.ylabel("Mean Spread")
plt.tight_layout()
plt.show()


# In[50]:


# IDA1_DAM & BM_IDA1 & BM_DAM Spread as Function of NetDemand
df["NetD_q_DAM"] = pd.qcut(df["NetDemand_DAM"], q=5, labels=False, duplicates="drop")
df["NetD_q_IDA1"] = pd.qcut(df["NetDemand_IDA1"], q=5, labels=False, duplicates="drop")

grouped2 = df.groupby("NetD_q_DAM")[["spr_IDA1_DAM"]].mean()
grouped3 = df.groupby("NetD_q_IDA1")[["spr_BM_IDA1"]].mean()
grouped4 = df.groupby("NetD_q_DAM")[["spr_BM_DAM"]].mean()

display(grouped2)

grouped2.plot(kind="bar", figsize=(8,4))
plt.title("Average Spread by NetDemand Quantile (spr_IDA1_DAM)")
plt.xlabel("NetDemand quantile")
plt.ylabel("Mean Spread")
plt.tight_layout()
plt.show()

display(grouped3)

grouped3.plot(kind="bar", figsize=(8,4))
plt.title("Average Spread by NetDemand Quantile (spr_BM_IDA1)")
plt.xlabel("NetDemand quantile")
plt.ylabel("Mean Spread")
plt.tight_layout()
plt.show()

display(grouped4)

grouped4.plot(kind="bar", figsize=(8,4))
plt.title("Average Spread by NetDemand Quantile (spr_BM_DAM)")
plt.xlabel("NetDemand quantile")
plt.ylabel("Mean Spread")
plt.tight_layout()
plt.show()


# ## Prediction & Inference

# ### Feature Engineering

# In[51]:


# Cyclical calendar encoding 
df["dow"] = df["ts"].dt.dayofweek
df["month"] = df["ts"].dt.month
df["hour_sin"]  = np.sin(2 * np.pi * df["hod"] / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["hod"] / 24)
df["dow_sin"]   = np.sin(2 * np.pi * df["dow"] / 7)
df["dow_cos"]   = np.cos(2 * np.pi * df["dow"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["is_weekend"]= (df["dow"] >= 5).astype(int)
df["is_winter"] = df["month"].isin([11, 12, 1, 2, 3]).astype(int)
df["is_peak"]   = df["hod"].between(7, 21).astype(int)

# Cross-provider wind forecast consensus & UNCERTAINTY 
# Forecast disagreement (std across providers) is a strong predictor of auction-to-auction price revision 
for sfx in ["DAM","IDA1","IDA2","IDA3"]:
    cands = [c for c in [f"Meteo_ISEMWIND_{sfx}", f"EmSys_C_ISEMWIND_{sfx}",
                          f"EmSys_U_ISEMWIND_{sfx}", f"EirGridWindFc_{sfx}"]
             if c in df.columns]
    if len(cands) >= 2:
        df[f"wind_consensus_{sfx}"]   = df[cands].mean(axis=1)
        df[f"wind_uncertainty_{sfx}"] = df[cands].std(axis=1)   # KEY signal
        df[f"wind_range_{sfx}"]       = df[cands].max(axis=1) - df[cands].min(axis=1)
        if f"EirGridWindFc_{sfx}" in df.columns:
            df[f"wind_eirg_dev_{sfx}"] = df[f"EirGridWindFc_{sfx}"] - df[f"wind_consensus_{sfx}"]

# Demand forecast disagreement 
for sfx in ["DAM","IDA1","IDA2","IDA3"]:
    m,e = f"Meteo_ISEMDEMAND_{sfx}", f"EirGridDemandFc_{sfx}"
    if m in df.columns and e in df.columns:
        df[f"demand_disagreement_{sfx}"] = df[m] - df[e]

# Forecast revision (known at each gate) 
if "EirGridWindFc_IDA1" in df.columns and "EirGridWindFc_DAM" in df.columns:
    df["wind_revision_IDA1_DAM"]  = df["EirGridWindFc_DAM"] - df["EirGridWindFc_IDA1"]
if "EirGridWindFc_IDA2" in df.columns and "EirGridWindFc_IDA1" in df.columns:
    df["wind_revision_IDA2_IDA1"] = df["EirGridWindFc_IDA1"] - df["EirGridWindFc_IDA2"]
if "EirGridWindFc_IDA3" in df.columns and "EirGridWindFc_IDA2" in df.columns:
    df["wind_revision_IDA3_IDA2"] = df["EirGridWindFc_IDA2"] - df["EirGridWindFc_IDA3"]

if "EirGridDemandFc_IDA1" in df.columns and "EirGridDemandFc_DAM" in df.columns:
    df["demand_revision_IDA1_DAM"] = df["EirGridDemandFc_IDA1"] - df["EirGridDemandFc_DAM"]
if "EirGridDemandFc_IDA2" in df.columns and "EirGridDemandFc_IDA1" in df.columns:
    df["demand_revision_IDA2_IDA1"] = df["EirGridDemandFc_IDA1"] - df["EirGridDemandFc_IDA2"]
if "EirGridDemandFc_IDA3" in df.columns and "EirGridDemandFc_IDA2" in df.columns:
    df["demand_revision_IDA3_IDA2"] = df["EirGridDemandFc_IDA2"] - df["EirGridDemandFc_IDA3"]

# Spread columns
SPREAD_COLS = [c for c in df.columns if c.startswith("spr_")]

# TEMPORAL LAG FEATURES 
LAG_BASE = SPREAD_COLS + [c for c in ["PriceDAM","PriceIDA1","PriceIDA2","PriceIDA3","NetDemand_DAM","NetDemand_IDA1","NetDemand_IDA2","NetDemand_IDA3","TotalPN"] if c in df.columns]
for col in LAG_BASE:
    s = df[col]
    df[f"{col}_lag1d"]  = s.shift(48)
    df[f"{col}_lag2d"]  = s.shift(96)
    df[f"{col}_lag7d"]  = s.shift(48*7)
    df[f"{col}_ma7d"]   = s.shift(48).rolling(48*7,  min_periods=24).mean()
    df[f"{col}_std7d"]  = s.shift(48).rolling(48*7,  min_periods=24).std()
    df[f"{col}_ma14d"]  = s.shift(48).rolling(48*14, min_periods=48).mean()

# Renewable Penetration
for sfx in ["DAM","IDA1","IDA2","IDA3"]:
    d,w,s = f"EirGridDemandFc_{sfx}", f"EirGridWindFc_{sfx}", f"ISEMSOLAR_{sfx}"
    if all(c in df.columns for c in [d,w,s]):
        df[f"ren_pct_{sfx}"] = (df[w]+df[s]) / (df[d]+1)

# GB vs Irl price spread
for gb in ["GB DAM Epex","GB DAM N2EX","GB DAM HH Epex"]:
    if gb in df.columns:
        safe = gb.replace(" ","_")
        df[f"GB_vs_IRL_{safe}"] = df[gb] - df["PriceDAM"]

print(f"Feature-engineered shape: {df.shape}")
print(f"Total columns: {df.shape[1]}")


# In[52]:


#  Exclusion sets
ALWAYS_EXCLUDE = {"ts","TradeDate","StartDateTime","Time","PriceImbalance"}
TARGET_PREFIXES = ("spr_","dir_","NetD_q_")

def build_xcols(df, allow_uppers, extra_always=None):
    extra_always = extra_always or []
    time_feats = ["hod","hour","dow","month","is_weekend","is_winter","is_peak","hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos"]
    lag_sfx = ("_lag1d","_lag2d","_lag7d","_ma7d","_std7d","_ma14d")
    utility_kw = ["wind_consensus","wind_uncertainty","wind_range","wind_eirg_dev","demand_disagreement","ren_pct"]
    feats = set()
    for c in df.columns:
        if c in ALWAYS_EXCLUDE: continue
        if any(c.startswith(p) for p in TARGET_PREFIXES): continue
        if c in time_feats or c in extra_always:
            feats.add(c); continue
        if any(c.startswith(kw) for kw in utility_kw):
            x = c
            for kw in utility_kw: x = x.replace(kw,"")
            if any(ap in x.upper() for ap in allow_uppers):
                feats.add(c); continue
        if any(c.endswith(s) for s in lag_sfx):
            base = c
            for s in lag_sfx: base = base.replace(s,"")
            if any(ap in base.upper() for ap in allow_uppers):
                feats.add(c); continue
        if any(ap in c.upper() for ap in allow_uppers):
            feats.add(c)
    return sorted(c for c in feats if c in df.columns)

# Transition definitions 
TRANSITIONS = {
    "DAM→IDA1":  ("spr_IDA1_DAM",
                  ["_DAM","PRICEDAM","GB_DAM","GB DAM","AGGREGATED","LOADFORECAST","_MW","PUMPSTORAGE","TOTALPN","NetDemand_DAM","GB_VS_IRL"],
                  ["PriceDAM","AggregatedForecast","demand_revision_IDA1_DAM","wind_revision_IDA1_DAM"], None),
    "IDA1→BM":   ("spr_BM_IDA1",
                  ["_DAM","_IDA1","PRICEDAM","PRICEIDA1","LOADFORECAST","_MW","PUMPSTORAGE","TOTALPN","NetDemand_IDA1","NetDemand_DAM"],
                  ["PriceDAM","AggregatedForecast","PredictedICFlow","PriceIDA1"], None),
    "IDA1→IDA2": ("spr_IDA2_IDA1",
                  ["_DAM","_IDA1","PRICEDAM","PRICEIDA1","LOADFORECAST","_MW","PUMPSTORAGE","TOTALPN","NetDemand_IDA1","NetDemand_DAM"],
                  ["PriceDAM","AggregatedForecast","PredictedICFlow","PriceIDA1"], "hod >= 11"),
    "IDA2→IDA3": ("spr_IDA3_IDA2",
                  ["_DAM","_IDA1","_IDA2","PRICEDAM","PRICEIDA1","PRICEIDA2","LOADFORECAST","_MW","PUMPSTORAGE","TOTALPN","NetDemand_IDA2","NetDemand_IDA1","NetDemand_DAM"],
                  ["PriceDAM","PriceIDA1","PriceIDA2","PostIDA1Flow","AggregatedForecast"], "hod >= 17"),
    "DAM→BM":    ("spr_BM_DAM",
                  ["_DAM","PRICEDAM","GB_DAM","GB DAM","AGGREGATED","LOADFORECAST","_MW","PUMPSTORAGE","TOTALPN","NetDemand_DAM","GB_VS_IRL"],
                  ["PriceDAM","AggregatedForecast","demand_revision_IDA1_DAM","wind_revision_IDA1_DAM"], None),
    "IDA2→BM":   ("spr_BM_IDA2",
                  ["_DAM","_IDA1","_IDA2","PRICEDAM","PRICEIDA1","PRICEIDA2","LOADFORECAST","_MW","PUMPSTORAGE","TOTALPN","NetDemand_IDA2","NetDemand_IDA1","NetDemand_DAM"],
                  ["PriceDAM","PriceIDA1","PriceIDA2","PostIDA1Flow","AggregatedForecast"], "hod >= 11"),
    "IDA3→BM":   ("spr_BM_IDA3",
                  ["_DAM","_IDA1","_IDA2","_IDA3","PRICEDAM","PRICEIDA1","PRICEIDA2","PRICEIDA3","LOADFORECAST","_MW","PUMPSTORAGE","TOTALPN","NetDemand_IDA3","NetDemand_IDA2","NetDemand_IDA1","NetDemand_DAM"],
                  ["PriceDAM","PriceIDA1","PriceIDA2","PriceIDA3","PostIDA1Flow","PostIDA2Flow","AggregatedForecast"], "hod >= 17"),
}

for name,(spr,allow,extra,win) in TRANSITIONS.items():
    n = len(build_xcols(df, [a.upper() for a in allow], extra))
    exists = spr in df.columns and df[spr].notna().sum() > 500
    print(f"  {name:12s}: spread={spr}  features={n:3d}  available={exists}")


# In[53]:


print(build_xcols(df, ["_DAM","_IDA1","_IDA2","_IDA3","PRICEDAM","PRICEIDA1","PRICEIDA2","PRICEIDA3","LOADFORECAST","_MW","PUMPSTORAGE","TOTALPN","NetDemand_IDA3","NetDemand_IDA2","NetDemand_IDA1","NetDemand_DAM"],
                  ["PriceDAM","PriceIDA1","PriceIDA2","PriceIDA3","PostIDA1Flow","PostIDA2Flow","AggregatedForecast"]))


# ### Defining & Running Baseline Models

# In[54]:


def make_pre(X):
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scl", StandardScaler())])
    return ColumnTransformer([("n", pipe, list(X.columns))], remainder="drop")

def make_clf():
    if HAS_LGB:
        return lgb.LGBMClassifier(
            n_estimators=800, learning_rate=0.03, max_depth=6, num_leaves=63,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)
    return GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.04, max_depth=5,
        min_samples_leaf=25, subsample=0.8, random_state=42)

def make_reg():
    if HAS_LGB:
        return lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.03, max_depth=6, num_leaves=63,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)
    return GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.04, max_depth=5,
        min_samples_leaf=25, subsample=0.8, random_state=42)

LAG_BURNIN = 48 * 14  # 14-day lag warm-up period

def build_ds(df, spr_col, xcols, win_expr=None):
    mask = df.index >= LAG_BURNIN
    if win_expr:
        mask &= df.eval(win_expr)
    mask &= df[spr_col].notna()
    mask &= np.sign(df[spr_col]) != 0
    d    = df.loc[mask, xcols].copy().reset_index(drop=True)
    yc   = np.sign(df.loc[mask, spr_col]).astype(int).reset_index(drop=True)
    yr   = df.loc[mask, spr_col].astype(float).reset_index(drop=True)
    orig = df.index[mask].to_numpy()
    return d, yc, yr, orig

def split(d, yc, yr, orig, frac=0.20):
    n = len(d); cut = int(n*(1-frac))
    return (d.iloc[:cut].copy(), d.iloc[cut:].copy(),
            yc.iloc[:cut].copy(), yc.iloc[cut:].copy(),
            yr.iloc[:cut].copy(), yr.iloc[cut:].copy(),
            orig[:cut], orig[cut:])

print("Pipeline definitions ready.")


# In[55]:


RESULTS = {}

for name, (spr, allow, extra, win) in TRANSITIONS.items():
    if spr not in df.columns or df[spr].notna().sum() < 500:
        print(f"  Skipping {name} — insufficient data")
        continue

    xcols = build_xcols(df, [a.upper() for a in allow], extra)
    xcols = [c for c in xcols if c in df.columns]
    d, yc, yr, orig = build_ds(df, spr, xcols, win)

    Xtr,Xte,yc_tr,yc_te,yr_tr,yr_te,oi_tr,oi_te = split(d,yc,yr,orig)

    print(f"\n{'═'*60}")
    print(f"  {name}  |  train={len(Xtr):,}  test={len(Xte):,}  features={len(xcols)}")
    print(f"  Class balance +1={(yc_tr==1).mean():.1%}  -1={(yc_tr==-1).mean():.1%}")

    # Classifier
    clf = Pipeline([("pre", make_pre(Xtr)), ("m", make_clf())])
    clf.fit(Xtr, yc_tr)
    yhat = clf.predict(Xte)
    acc  = accuracy_score(yc_te, yhat)
    try:
        proba  = clf.predict_proba(Xte)
        up_idx = list(clf.classes_).index(1)
        auc    = roc_auc_score((yc_te==1).astype(int), proba[:,up_idx])
    except:
        auc = float("nan")
    print(f"  [CLF] accuracy={acc:.4f}  auc={auc:.4f}")
    print(classification_report(yc_te, yhat, digits=4))

    # Regressor
    reg = Pipeline([("pre", make_pre(Xtr)), ("m", make_reg())])
    reg.fit(Xtr, yr_tr)
    yr_pred = reg.predict(Xte)
    mae = mean_absolute_error(yr_te, yr_pred)
    r2  = r2_score(yr_te, yr_pred)
    print(f"  [REG] MAE={mae:.4f}  R²={r2:.4f}")

    # Walk-forward CV (5 folds)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_accs = []
    for tr_i, te_i in tscv.split(d):
        p = Pipeline([("pre", make_pre(d.iloc[tr_i])), ("m", make_clf())])
        p.fit(d.iloc[tr_i], yc.iloc[tr_i])
        cv_accs.append(accuracy_score(yc.iloc[te_i], p.predict(d.iloc[te_i])))
    print(f"  [WF-CV 5-fold] {[round(a,4) for a in cv_accs]}  "
          f"mean={np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")

    RESULTS[name] = dict(
        tag=name, spr=spr, clf=clf, reg=reg,
        Xte=Xte, yc_te=yc_te, yr_te=yr_te, oi_te=oi_te,
        acc=acc, auc=auc, mae=mae, r2=r2,
        cv_mean=np.mean(cv_accs), cv_std=np.std(cv_accs)
    )

print("\nAll models trained.")


# In[56]:


n = min(len(RESULTS), 4)
fig, axes = plt.subplots(1, n, figsize=(5*n, 6))
if n == 1: axes = [axes]

for ax, (name, res) in zip(axes, list(RESULTS.items())[:n]):
    step = res["clf"].named_steps["m"]
    if not hasattr(step, "feature_importances_"):
        ax.set_title(f"{name}\n(no importances)"); continue
    imp   = step.feature_importances_
    feats = res["Xte"].columns.tolist()
    idx   = np.argsort(imp)[::-1][:15]
    ax.barh([feats[i] for i in idx][::-1], imp[idx][::-1], color="steelblue")
    ax.set_title(name, fontsize=9); ax.tick_params(axis="y", labelsize=6)
    ax.set_xlabel("Importance", fontsize=8)

plt.suptitle("Top 15 Feature Importances", fontsize=11, y=1.01)
plt.tight_layout(); plt.show()


# ### Trading Strategy Evaluation

# In[57]:


def evaluate_pnl(res, df, conf_thresholds=(0.0, 0.05, 0.10, 0.15, 0.20),
                  max_vol=10.0):
    spread_true = df.loc[res["oi_te"], res["spr"]].to_numpy(dtype=float)
    Xte         = res["Xte"]

    proba  = res["clf"].predict_proba(Xte)
    up_idx = list(res["clf"].classes_).index(1)
    p_up   = proba[:, up_idx]
    conf   = np.abs(p_up - 0.5)
    pred_m = res["reg"].predict(Xte)
    # Magnitude-proportional volume: vol ∝ |predicted spread| / median|predicted spread|
    p25 = max(np.percentile(np.abs(pred_m), 25), 0.5)
    vol_raw = np.clip(np.abs(pred_m) / p25, 0.1, max_vol)

    rows = []
    for thresh in conf_thresholds:
        sign_p = np.sign(p_up - 0.5).copy()
        sign_p[conf <= thresh] = 0
        traded   = sign_p != 0
        correct  = (sign_p == np.sign(spread_true)) & traded
        pnl_mwh  = np.where(correct, np.abs(spread_true), -np.abs(spread_true))
        pnl_eur  = np.where(traded, pnl_mwh * vol_raw, 0.0)
        vol_exec = np.where(traded, vol_raw, 0.0)
        t_vol = vol_exec.sum(); t_pnl = pnl_eur.sum()
        epm   = t_pnl / t_vol if t_vol > 0 else np.nan
        hr    = correct[traded].mean() if traded.any() else np.nan
        s_ratio = (pnl_eur[traded].mean() / (pnl_eur[traded].std() + 1e-9)
                   if traded.sum() > 1 else np.nan)
        rows.append({"transition":res["tag"], "conf_thresh":thresh,
                     "n_trades":int(traded.sum()), "hit_rate":round(hr,4),
                     "eur_per_mwh":round(epm,4), "total_pnl_eur":round(t_pnl,2),
                     "sharpe_proxy":round(s_ratio,4)})
    return pd.DataFrame(rows)

all_trading = pd.concat([evaluate_pnl(r, df) for r in RESULTS.values()], ignore_index=True)

print("\nTrading results at all confidence thresholds:")
print(all_trading.to_string(index=False))


# In[58]:


# Best threshold per transition
best = (all_trading
        .sort_values("eur_per_mwh", ascending=False)
        .drop_duplicates("transition")
        .reset_index(drop=True))

metrics = pd.DataFrame([{
    "transition": r["tag"],
    "test_accuracy": round(r["acc"],4),
    "roc_auc":       round(r["auc"],4),
    "cv_accuracy":   round(r["cv_mean"],4),
    "cv_std":        round(r["cv_std"],4),
    "reg_r2":        round(r["r2"],4),
    "reg_mae":       round(r["mae"],4),
} for r in RESULTS.values()])

LEADERBOARD = best.merge(metrics, on="transition")
LEADERBOARD = LEADERBOARD.sort_values("eur_per_mwh", ascending=False).reset_index(drop=True)

print("\n" + "="*80)
print("FINAL LEADERBOARD (sorted by EUR/MWh profit)")
print("="*80)
print(LEADERBOARD.to_string(index=False))

w_profit = LEADERBOARD.iloc[0]
w_acc    = LEADERBOARD.sort_values("test_accuracy", ascending=False).iloc[0]
print(f"\n>>> BEST PROFIT: {w_profit['transition']}  →  {w_profit['eur_per_mwh']:.4f} €/MWh"
      f"  (hit_rate={w_profit['hit_rate']:.2%}  n_trades={int(w_profit['n_trades'])})")
print(f">>> BEST ACCURACY: {w_acc['transition']}  →  {w_acc['test_accuracy']:.4f}"
      f"  (AUC={w_acc['roc_auc']:.4f}  CV={w_acc['cv_accuracy']:.4f}±{w_acc['cv_std']:.4f})")


# In[59]:


# Confusion Matrix
best_transition = w_acc["transition"]   # Use the transition with highest EUR/MWh
print(f"Creating confusion matrix for: {best_transition}")

res = RESULTS[best_transition]
yc_true = res["yc_te"]
yc_pred = res["clf"].predict(res["Xte"])

cm = confusion_matrix(yc_true, yc_pred, labels=[-1, 1])
cm_pct = cm.astype('float') / cm.sum() * 100  # percentage of total
cm_row_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # row percentages

fig, ax = plt.subplots(figsize=(8, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Spread ↓ (−1)", "Spread ↑ (+1)"]
)

disp.plot(ax=ax, cmap="Blues", values_format='d', colorbar=True)

# Add percentage annotations to each cell
for i in range(2):
    for j in range(2):
        count_text = cm[i, j]
        row_pct = cm_row_pct[i, j]
        ax.text(j, i + 0.25, f"({row_pct:.1f}% of row)",
                ha="center", va="center", 
                fontsize=9, color="darkblue" if cm[i,j] > cm.max()/2 else "black")

accuracy = res["acc"]
tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
recall_down = tn / (tn + fp) if (tn + fp) > 0 else 0
recall_up = tp / (tp + fn) if (tp + fn) > 0 else 0

ax.set_title(
    f"Confusion Matrix — {best_transition}\n"
    f"Accuracy: {accuracy:.2%}  |  Recall ↓: {recall_down:.1%}  |  Recall ↑: {recall_up:.1%}",
    fontsize=12, fontweight='bold', pad=15
)
ax.set_xlabel("Predicted Direction", fontsize=11, fontweight='bold')
ax.set_ylabel("Actual Direction", fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\n Interpretation:")
print(f"  • True Negatives (TN): {tn:,} — Correctly predicted downward spreads")
print(f"  • False Positives (FP): {fp:,} — Predicted up, actually down (Type I error)")
print(f"  • False Negatives (FN): {fn:,} — Predicted down, actually up (Type II error)")
print(f"  • True Positives (TP): {tp:,} — Correctly predicted upward spreads")
print(f"\n  • Total test samples: {cm.sum():,.0f}")
print(f"  • Overall accuracy: {accuracy:.2%}")


# ### Visualisations

# In[60]:


# Fig 1: Cumulative P&L 
fig, ax = plt.subplots(figsize=(14, 5))
cols = plt.cm.tab10.colors
BEST_THRESH = 0.20  

for i, (name, res) in enumerate(RESULTS.items()):
    spread_true = df.loc[res["oi_te"], res["spr"]].to_numpy(dtype=float)
    proba = res["clf"].predict_proba(res["Xte"])
    up_i  = list(res["clf"].classes_).index(1)
    p_up  = proba[:, up_i]; conf = np.abs(p_up - 0.5)
    pred_m = res["reg"].predict(res["Xte"])
    p25    = max(np.percentile(np.abs(pred_m), 25), 0.5)
    vol    = np.clip(np.abs(pred_m)/p25, 0.1, 10.0)
    sign_p = np.sign(p_up - 0.5); sign_p[conf <= BEST_THRESH] = 0
    correct= (sign_p == np.sign(spread_true)) & (sign_p != 0)
    pnl    = np.where(sign_p!=0,
                      np.where(correct, np.abs(spread_true), -np.abs(spread_true))*vol, 0)
    ts_v   = df.loc[res["oi_te"], "ts"].values
    cum    = np.cumsum(pnl)
    ax.plot(ts_v, cum, lw=1.8, color=cols[i%10],
            label=f"{name}  final=€{cum[-1]:,.0f}")

ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_title(f"Cumulative P&L — All Transitions (conf_threshold={BEST_THRESH}, magnitude sizing)", fontsize=11)
ax.set_xlabel("Date"); ax.set_ylabel("Cumulative P&L (€)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"€{x:,.0f}"))
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# Fig 2: Accuracy vs EUR/MWh bubble chart 
plt.figure(figsize=(12, 6))
ax = plt.gca()
size_scale = np.sqrt(LEADERBOARD["n_trades"])
size_scale = 40 + 220 * (size_scale - size_scale.min()) / (size_scale.max() - size_scale.min())
plot_df = LEADERBOARD.sort_values("n_trades")

for i, (_, row) in enumerate(plot_df.iterrows()):
    ax.scatter(
        row["test_accuracy"],
        row["eur_per_mwh"],
        s=size_scale.iloc[i],
        alpha=0.75,
        edgecolor="black",
        linewidth=0.6,
        color=cols[i % len(cols)],
        zorder=3
    )
    ax.annotate(
        row["transition"],
        (row["test_accuracy"], row["eur_per_mwh"]),
        fontsize=9,
        xytext=(6, 6),
        textcoords="offset points",
        weight="semibold"
    )

ax.axhline(0, color="red", lw=1.2, ls="--", alpha=0.7)
ax.set_xlabel("Test Accuracy", fontsize=12, weight="semibold")
ax.set_ylabel("EUR/MWh", fontsize=12, weight="semibold")
ax.set_title(
    "Model Performance: Accuracy vs Profitability\n(Bubble size ∝ number of trades)",
    fontsize=14,
    weight="bold",
    pad=15
)
ax.set_xlim(LEADERBOARD["test_accuracy"].min() - 0.01,
            LEADERBOARD["test_accuracy"].max() + 0.01)

ax.set_ylim(bottom=min(-1, LEADERBOARD["eur_per_mwh"].min() - 1))
ax.grid(alpha=0.25, linestyle="--", zorder=0)
ax.legend().remove()
plt.tight_layout()
plt.show()

# Fig 3: EUR/MWh sensitivity to confidence threshold 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name, res in RESULTS.items():
    t = all_trading[all_trading["transition"]==name]
    axes[0].plot(t["conf_thresh"], t["eur_per_mwh"], marker="o", label=name)
    axes[1].plot(t["conf_thresh"], t["n_trades"],    marker="s", label=name)
for ax, ti, yi in zip(axes,
    ["EUR/MWh vs Confidence Threshold", "N Trades vs Confidence Threshold"],
    ["EUR/MWh","N Trades"]):
    ax.set_title(ti); ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel(yi); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# Fig 4: Spread distributions 
n = len([s for s in TRANSITIONS if s in RESULTS])
fig, axes = plt.subplots(2, 4, figsize=(18, 8)); axes = axes.flatten()
for i, (name, (spr,_,_,_)) in enumerate(TRANSITIONS.items()):
    if i >= len(axes) or spr not in df.columns: continue
    s = df[spr].dropna()
    axes[i].hist(s, bins=80, color="steelblue", alpha=0.7, edgecolor="none")
    axes[i].axvline(0, color="red", lw=1.2, ls="--")
    axes[i].set_xlim(s.quantile(0.01), s.quantile(0.99))
    axes[i].set_title(f"{name}\nμ={s.mean():.1f}  σ={s.std():.1f}  |μ|={s.abs().mean():.1f}", fontsize=9)
    axes[i].set_xlabel("€/MWh")
for j in range(i+1, len(axes)): axes[j].set_visible(False)
plt.suptitle("Spread Distributions (1st–99th percentile)", fontsize=12, y=1.01)
plt.tight_layout(); plt.show()


# In[61]:


# DAM → BM only
cols = plt.cm.tab10.colors
BEST_THRESH = 0.20

# 1) find the DAM→BM result key
dam_bm_key = None
for k in RESULTS.keys():
    kk = str(k).upper().replace(" ", "")
    kk = kk.replace("→", "->")
    if ("DAM" in kk) and ("BM" in kk) and ("DAM->BM" in kk or "DAM-BM" in kk or "DAM_BM" in kk or ("DAM" in kk and "BM" in kk)):
        dam_bm_key = k
        break

if dam_bm_key is None:
    raise KeyError(f"Couldn't find a DAM→BM transition key in RESULTS. Keys are: {list(RESULTS.keys())[:10]} ...")

res = RESULTS[dam_bm_key]

# 2) compute P&L for that transition
spread_true = df.loc[res["oi_te"], res["spr"]].to_numpy(dtype=float)

proba = res["clf"].predict_proba(res["Xte"])
up_i  = list(res["clf"].classes_).index(1)
p_up  = proba[:, up_i]
conf  = np.abs(p_up - 0.5)

pred_m = res["reg"].predict(res["Xte"])
p25    = max(np.percentile(np.abs(pred_m), 25), 0.5)
vol    = np.clip(np.abs(pred_m) / p25, 0.1, 10.0)

sign_p = np.sign(p_up - 0.5)
sign_p[conf <= BEST_THRESH] = 0

correct = (sign_p == np.sign(spread_true)) & (sign_p != 0)
pnl     = np.where(
    sign_p != 0,
    np.where(correct, np.abs(spread_true), -np.abs(spread_true)) * vol,
    0
)

ts_v = df.loc[res["oi_te"], "ts"].values
cum  = np.cumsum(pnl)

# 3) plot
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(ts_v, cum, lw=2.0, color=cols[0],
        label=f"{dam_bm_key}  final=€{cum[-1]:,.0f}")

ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_title(f"Cumulative P&L — {dam_bm_key} (conf_threshold={BEST_THRESH}, magnitude sizing)", fontsize=11)
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative P&L (€)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()


# In[62]:


# DAM → BM Drawdown
BEST_THRESH = 0.20

dam_bm_key = None
for k in RESULTS.keys():
    kk = str(k).upper().replace(" ", "").replace("→", "->")
    if "DAM" in kk and "BM" in kk:
        dam_bm_key = k
        break

if dam_bm_key is None:
    raise KeyError("DAM→BM transition not found in RESULTS")

res = RESULTS[dam_bm_key]

spread_true = df.loc[res["oi_te"], res["spr"]].to_numpy(dtype=float)

proba = res["clf"].predict_proba(res["Xte"])
up_i  = list(res["clf"].classes_).index(1)
p_up  = proba[:, up_i]
conf  = np.abs(p_up - 0.5)

pred_m = res["reg"].predict(res["Xte"])
p25    = max(np.percentile(np.abs(pred_m), 25), 0.5)
vol    = np.clip(np.abs(pred_m) / p25, 0.1, 10.0)

sign_p = np.sign(p_up - 0.5)
sign_p[conf <= BEST_THRESH] = 0

correct = (sign_p == np.sign(spread_true)) & (sign_p != 0)

pnl = np.where(
    sign_p != 0,
    np.where(correct, np.abs(spread_true), -np.abs(spread_true)) * vol,
    0
)

test_dam_bm = (
    df.loc[res["oi_te"], ["ts"]]
      .assign(pnl=pnl)
      .sort_values("ts")
      .reset_index(drop=True)
)

test_dam_bm["cum_pnl"] = test_dam_bm["pnl"].cumsum()
test_dam_bm["cum_max"] = test_dam_bm["cum_pnl"].cummax()
test_dam_bm["drawdown"] = test_dam_bm["cum_pnl"] - test_dam_bm["cum_max"]

max_drawdown_dam_bm = test_dam_bm["drawdown"].min()
print(f"Max Drawdown (DAM→BM): €{max_drawdown_dam_bm:,.2f}")

plt.figure(figsize=(10, 3))
plt.plot(test_dam_bm["ts"], test_dam_bm["drawdown"], color="red", lw=1.5)
plt.axhline(0, color="black", linewidth=1)
plt.title("Drawdown (€) — DAM → BM")
plt.ylabel("Drawdown (€)")
plt.xlabel("Date")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

