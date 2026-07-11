# 1. Import thư viện
### Import Dependencies

```python
# ── Data & Math ──────────────────────────────────────────────────
import pandas as pd
import numpy as np

# ── Visualization ─────────────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns

# ── ML & Preprocessing ────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.inspection import permutation_importance

# ── Clustering ────────────────────────────────────────────────────
from dtaidistance import dtw as dtw_lib
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

# ── Utilities ─────────────────────────────────────────────────────
import joblib
import json, os, warnings
warnings.filterwarnings('ignore')
```

***
# 2. Data Pipeline
### Pipeline 3 Level — Tiền xử lý Chi Tiết

Dữ liệu COVID-19 trải qua **3 mức tiền xử lý** trước khi vào mô hình:

#### **LEVEL 1: Data Exploration & Column Selection**
**File:** `0_data_exploration.ipynb` + `1_data_selection.ipynb`

```python
# ─── Input ───────────────────────────────────────────────────
df_raw = pd.read_csv('covid_data.csv')
print(f"Shape gốc: {df_raw.shape}")  # (570,532, 67)
print(f"Thời gian: {df_raw['date'].min()} → {df_raw['date'].max()}")
print(f"Quốc gia: {df_raw['country'].nunique()}")  # 239 countries

# ─── Phân tích cột ────────────────────────────────────────────
missing_rate = df_raw.isnull().sum() / len(df_raw)
print("Top 10 cột thiếu nhiều nhất:")
print(missing_rate.nlargest(10))
# Output:
# people_fully_vaccinated           0.832  ← 83.2% missing
# people_vaccinated                 0.801  ← 80.1% missing
# excess_mortality_cumulative_excess 0.924
# ...

# ─── Lọc cột ──────────────────────────────────────────────────
# Tiêu chí: Giữ cột có < 80% missing
good_threshold = 0.80
good_cols = missing_rate[missing_rate < good_threshold].index.tolist()
df_level1 = df_raw[good_cols].copy()

print(f"\\n=== OUTPUT LEVEL 1 ===")
print(f"Shape sau lọc cột: {df_level1.shape}")  # (570,532, 45)
print(f"Cột loại bỏ: {len(df_raw.columns) - len(good_cols)}")  # 22 cột

# Các cột được giữ (mẫu):
# date, country, population, total_cases, total_deaths,
# new_cases_smoothed, new_deaths_smoothed, 
# new_people_vaccinated_smoothed_per_hundred,
# stringency_index, ...
```

**Output Level 1:**
```
✅ Shape: (570,532 dòng, 45 cột)
✅ Missing rate: 1.2% - 78.5% (trong ngưỡng)
✅ Thời gian: 2020-01-01 → 2026-02-22
✅ Quốc gia: 239 quốc gia
❌ Loại bỏ: 22 cột (missing > 80%)
```

---

#### **LEVEL 2: Data Cleaning & Normalization**
**File:** `2_data_processing.ipynb`

```python
# ─── Input từ Level 1 ────────────────────────────────────────
df_level1 = pd.read_csv('covid_data_stage1.csv')  # (570,532, 45)

# ─── A. Xử lý Missing Values ──────────────────────────────────

# 1. Interpolation: Dùng liner interpolation cho gaps nhỏ
df_level2 = df_level1.groupby('country').apply(
    lambda group: group.interpolate(method='linear', limit=7)
    # Tối đa fill 7 ngày liên tiếp
)

# 2. Forward/Backward Fill: Cho các missing đầu/cuối
df_level2 = df_level2.groupby('country', group_keys=False).apply(
    lambda g: g.fillna(method='ffill').fillna(method='bfill')
)

missing_before = df_level1.isnull().sum().sum()  # ~15,000 cells
missing_after = df_level2.isnull().sum().sum()   # ~200 cells (NaN không phục hồi được)
print(f"Missing values: {missing_before} → {missing_after}")
# → 98.7% missing được phục hồi

# ─── B. Rolling Average Smooth (7-day) ────────────────────────
smoothing_cols = [
    'new_cases', 'new_deaths', 'new_people_vaccinated'
]

for col in smoothing_cols:
    if col in df_level2.columns:
        # Per-country rolling average (tránh cross-country leak)
        df_level2[f'{col}_custom_smoothed'] = (
            df_level2.groupby('country')[col]
            .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        )

# ─── C. Outlier Handling (IQR method) ────────────────────────
for col in smoothing_cols:
    if f'{col}_custom_smoothed' in df_level2.columns:
        smooth_col = f'{col}_custom_smoothed'
        
        def remove_outliers(group):
            Q1 = group[smooth_col].quantile(0.25)
            Q3 = group[smooth_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers thay vì drop (keep linearity)
            group[smooth_col] = group[smooth_col].clip(lower_bound, upper_bound)
            return group
        
        df_level2 = df_level2.groupby('country', group_keys=False).apply(remove_outliers)

# ─── D. Per-Capita Normalization ──────────────────────────────
pop_col = 'population'
for col in ['new_cases_custom_smoothed', 'new_deaths_smoothed']:
    # Per-capita: cases per 100,000 population
    df_level2[f'{col}_per_100k'] = (
        (df_level2[col] / df_level2[pop_col]) * 100_000
    ).fillna(0)

print(f"\\n=== OUTPUT LEVEL 2 ===")
print(f"Shape: {df_level2.shape}")  # (570,532, 67) - thêm smoothed columns
print(f"Missing: {df_level2.isnull().sum().sum()}")  # ~200 cells
print(f"Date range: {df_level2['date'].min()} → {df_level2['date'].max()}")

# Lưu output
df_level2.to_csv('covid_data_stage2.csv', index=False)
```

**Output Level 2:**
```
✅ Shape: (570,532 dòng, 67 cột)
✅ Missing: 200 cells / 38M → 0.0005% (xem như bằng 0)
✅ Smoothed: new_cases_custom_smoothed, new_deaths_smoothed_7d
✅ Per-capita: Per 100,000 dân
✅ Outliers: Capped (không drop dòng)
✅ Output: covid_data_stage2.csv ← **Dùng cho các mô hình**
```

---

#### **LEVEL 3: Model-Level Preprocessing**
**File:** `3_model_*.ipynb` (trong mỗi mô hình)

Mỗi mô hình có preprocessing riêng tùy theo loại task:

```python
# ═══════════════════════════════════════════════════════════════
# 3A. CLUSTERING PREPROCESSING (3_model_clustering_v2.ipynb)
# ═══════════════════════════════════════════════════════════════

# Input: covid_data_stage2.csv (570,532 × 67)

# Step 1: Chọn 40 quốc gia cố định (diverse, reproducible)
COUNTRIES_40 = ['Vietnam', 'Thailand', ..., 'Australia']
# Lọc: available in df → 40 quốc gia
# Lọc: pivot missing < 5% → 39 quốc gia (loại 1 quốc gia)

# Step 2: Pivot long → wide (mỗi cột = 1 quốc gia chuỗi thời gian)
pivot_df = df.pivot_table(
    index='date', columns='country', 
    values='new_cases_custom_smoothed'
)
# Shape: (2245 ngày, 39 quốc gia)

# Step 3: Min-Max normalization per-country (tránh scale bias)
def normalize_minmax(series):
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - s_min) / (s_max - s_min)

normalized_df = pivot_df.apply(normalize_minmax, axis=0)
# Output: (2245, 39) giá trị trong [0, 1]

# Step 4: Compute DTW distance matrix
dtw_distances = np.zeros((39, 39))
for i in range(39):
    for j in range(i+1, 39):
        dist = dtw.distance(
            normalized_df.iloc[:, i].values,
            normalized_df.iloc[:, j].values,
            window=225  # 10% của 2245 ngày
        )
        dtw_distances[i, j] = dist
        dtw_distances[j, i] = dist

# Output: 39×39 distance matrix → Hierarchical clustering
# Result: cluster_assignment.csv (39 quốc gia → 8 clusters)


# ═══════════════════════════════════════════════════════════════
# 3B. RANDOM FOREST PREPROCESSING (3_model_random_forest.ipynb)
# ═══════════════════════════════════════════════════════════════

# Input: covid_data_stage2.csv (529,000+ rows)

# Step 1: Target Variable Creation (shift -7 days)
df['future_cases'] = df.groupby('country')['new_cases_custom_smoothed'].shift(-7)
df['future_ratio'] = df['future_cases'] / df['new_cases_custom_smoothed'].replace(0, np.nan)
df['target'] = 'Stable'
df.loc[df['future_ratio'] >= 1.15, 'target'] = 'Increase'
df.loc[df['future_ratio'] <= 0.85, 'target'] = 'Decrease'
# Fix data leakage:
dynamic_low = df['population'] / 1_000_000 * 1.0  # 1 case per M people
low_mask = (df['new_cases_custom_smoothed'] <= dynamic_low) & \
           (df['future_cases'] <= dynamic_low)
df.loc[low_mask, 'target'] = 'Stable'
# Remove rows without future data
df = df.dropna(subset=['future_cases'])

# Step 2: Feature Engineering (18 features)
# Lag group (4)
df['lag1'] = df.groupby('country')['new_cases_custom_smoothed'].shift(1)
df['lag7'] = df.groupby('country')['new_cases_custom_smoothed'].shift(7)
df['lag14'] = df.groupby('country')['new_cases_custom_smoothed'].shift(14)
df['lag_deaths_7'] = df.groupby('country')['new_deaths_smoothed'].shift(7)

# Trend group (4)
df['growth_rate_daily'] = df['new_cases_custom_smoothed'] / df['lag1'].replace(0, np.nan) - 1
df['growth_rate_weekly'] = df['new_cases_custom_smoothed'] / df['lag7'].replace(0, np.nan) - 1
df['acceleration'] = df['growth_rate_daily'] - df.groupby('country')['growth_rate_daily'].shift(1)
df['rolling_std'] = df.groupby('country')['new_cases_custom_smoothed'].transform(
    lambda x: x.rolling(7, min_periods=3).std()
)

# Intervention, Time, Contextual groups... (10 more)

SELECTED_FEATURES = [
    'lag1', 'lag7', 'lag14', 'lag_deaths_7',
    'growth_rate_daily', 'growth_rate_weekly', 'acceleration', 'rolling_std',
    'vaccination_coverage', 'stringency_index', 'vax_phase_encoded',
    'day_of_week', 'month', 'is_weekend', 'days_since_first_case',
    'cfr', 'attack_rate', 'reproduction_proxy',
]

# Clean NaN from shifts
df_model = df.dropna(subset=SELECTED_FEATURES + ['target']).copy()
# Output: 291,085 rows × 20 columns

# Step 3: Time-based Split
train = df_model[df_model['date'] < '2021-10-01']     # 108,814 rows
val = df_model[(df_model['date'] >= '2021-10-01') &   #  18,858 rows
               (df_model['date'] < '2022-01-01')]
test = df_model[(df_model['date'] >= '2022-01-01') &  #  71,678 rows
                (df_model['date'] <= '2022-12-31')]

X_train = train[SELECTED_FEATURES]
X_val = val[SELECTED_FEATURES]
X_test = test[SELECTED_FEATURES]
y_train = train['target']
y_val = val['target']
y_test = test['target']

# Step 4: StandardScaler (fit only on train!)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)  # Fit
X_val_sc = scaler.transform(X_val)          # Transform only
X_test_sc = scaler.transform(X_test)        # Transform only

print(f"=== OUTPUT LEVEL 3 (RF) ===")
print(f"Train: {X_train_sc.shape}")   # (108814, 18)
print(f"Val:   {X_val_sc.shape}")     #  (18858, 18)
print(f"Test:  {X_test_sc.shape}")    #  (71678, 18)
print(f"Label distribution: Stable {(y_train=='Stable').mean():.1%}, "
      f"Decrease {(y_train=='Decrease').mean():.1%}, "
      f"Increase {(y_train=='Increase').mean():.1%}")


# ═══════════════════════════════════════════════════════════════
# 3C. HYBRID PREPROCESSING (4_model_hybrid_v2.ipynb)
# ═══════════════════════════════════════════════════════════════

# Same as RF steps 1-3, PLUS:

# Step 4a: Load cluster assignment
cluster_df = pd.read_csv('cluster_assignment.csv')
cluster_map = dict(zip(cluster_df['country'], cluster_df['cluster']))
df['dtw_cluster'] = df['country'].map(cluster_map).fillna(-1).astype(int)

# Step 4b: One-hot encode cluster
df_ohe = pd.get_dummies(
    df[['dtw_cluster']],
    columns=['dtw_cluster'],
    prefix='cluster',
    drop_first=False  # Keep all clusters
)
CLUSTER_COLS = [c for c in df_ohe.columns if c != 'cluster_-1']  # ~8 columns

# Combine features
SELECTED_FEATURES_HYBRID = SELECTED_FEATURES + CLUSTER_COLS  # 18 + 8 = 26

# Rest of preprocessing same as RF...
# Output: X_train_sc, X_val_sc, X_test_sc (shape: n × 26)

print(f"=== OUTPUT LEVEL 3 (Hybrid) ===")
print(f"Base features: {len(SELECTED_FEATURES)}")
print(f"Cluster OHE: {len(CLUSTER_COLS)}")
print(f"Total features: {len(SELECTED_FEATURES_HYBRID)}")
```

**Output Level 3:**

| Mô hình | Features | Train | Val | Test | Scaling |
|---------|----------|-------|-----|------|---------|
| **Clustering** | Không (DTW distance matrix) | (2245, 2245) distance matrix | — | — | Min-Max per-country |
| **RF đơn** | 18 features | (108K, 18) | (18K, 18) | (71K, 18) | StandardScaler |
| **Hybrid** | 18 + 8 OHE | (108K, 26) | (18K, 26) | (71K, 26) | StandardScaler |

---
# 3. Phân tích Clusting Code __ DTW + BIRCH
# 4. Phân tích thuật toán ARIMA
# 5. Kết quả
