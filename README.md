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
### File: `3_model_clustering_v2.ipynb`

#### Bước 1: Chọn quốc gia và Pivot dữ liệu

```python
CLUSTER_FEATURE = 'new_cases_custom_smoothed'

# [Fix 1] 40 quốc gia CỐ ĐỊNH — đa dạng địa lý, không random
# → reproducibility: mỗi lần chạy ra cùng kết quả
COUNTRIES_40 = [
    # Châu Á — Đông Nam Á & Đông Á
    'Vietnam', 'Thailand', 'Indonesia', 'Philippines', 'Malaysia',
    'South Korea', 'Japan', 'China', 'Taiwan',
    # Châu Á — Nam Á & Tây Á
    'India', 'Bangladesh', 'Pakistan', 'Iran', 'Turkey',
    # Châu Âu — Tây & Bắc
    'United Kingdom', 'Germany', 'France', 'Italy', 'Spain',
    'Netherlands', 'Sweden', 'Norway',
    # Châu Âu — Đông
    'Russia', 'Poland', 'Ukraine',
    # Châu Mỹ — Bắc & Nam
    'United States', 'Canada', 'Mexico',
    'Brazil', 'Argentina', 'Colombia', 'Chile', 'Peru',
    # Châu Phi
    'South Africa', 'Nigeria', 'Egypt', 'Kenya', 'Morocco',
    # Châu Đại Dương
    'Australia', 'New Zealand',
]

# Lọc quốc gia thực sự có trong dữ liệu
available = df['country'].unique().tolist()
COUNTRIES  = [c for c in COUNTRIES_40 if c in available]
missing    = [c for c in COUNTRIES_40 if c not in available]
print(f"Sử dụng {len(COUNTRIES)} quốc gia (bỏ {len(missing)}: {missing})")

df_cluster = df[df['country'].isin(COUNTRIES)].copy()

# Long → Wide: mỗi quốc gia = 1 cột chuỗi thời gian
pivot_df = df_cluster.pivot_table(
    index='date', columns='country', values=CLUSTER_FEATURE
)
# Giữ chỉ quốc gia có đủ dữ liệu (missing < 5%)
good_cols = pivot_df.columns[pivot_df.isnull().mean() < 0.05].tolist()
pivot_df  = pivot_df[good_cols].fillna(0)
COUNTRIES = good_cols
```

**Output thực tế:**

```
Sử dụng 40 quốc gia (bỏ 0 không có trong dữ liệu: [])
Shape wide format : (2245, 39)
Giai đoạn         : 2020-01-01 → 2026-02-22
Missing sau lọc   : 0.0000
```

**Lý do dùng fixed list thay vì random:**

| Tiêu chí        | Giải thích                                                           |
| --------------- | -------------------------------------------------------------------- |
| Reproducibility | Mỗi lần chạy ra cùng 39 quốc gia → so sánh được                      |
| Đa dạng địa lý  | Phủ 6 khu vực: Đông Nam Á, Đông Á, Nam Á, Châu Âu, Châu Mỹ, Châu Phi |
| Đa dạng quy mô  | Từ 450K dân (New Zealand) đến 1.4B (China)                           |
| Không singleton | Mỗi cluster có ít nhất 3 quốc gia → Silhouette tính được             |

#### Bước 2: Chuẩn hóa Min-Max per-country

```python
def normalize_minmax(series):
    """
    Min-Max normalization: đưa chuỗi về [0, 1].
    Xử lý edge case: nếu min == max (chuỗi phẳng) → trả về toàn 0.
    """
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - s_min) / (s_max - s_min)

normalized_df = pivot_df.apply(normalize_minmax, axis=0).fillna(0)
series_list   = [normalized_df[c].values for c in COUNTRIES]
```

**Output thực tế:**

```
Số quốc gia  : 40
Độ dài chuỗi : 2245 ngày
Min / Max sau chuẩn hóa: 0.000 / 1.000
```

**Tại sao Min-Max không StandardScaler?**

| Phương pháp        | Công thức                        | Phù hợp khi                          |
| ------------------ | -------------------------------- | ------------------------------------ |
| **Min-Max**        | (x - min) / (max - min) → \[0,1] | So sánh *hình dạng* curve ← **CHỌN** |
| **StandardScaler** | (x - mean) / std → N(0,1)        | Phân phối chuẩn, so sánh *mức độ*    |

→ **Min-Max tốt hơn cho DTW**: Mỹ (345,000 ca/ngày) không "xa" Việt Nam (8,000 ca/ngày) nếu shape curve giống nhau.

#### Bước 3: Chọn DTW window tối ưu — \[Fix 2]

```python
def compute_dtw_matrix(series_list, countries, window=60, verbose=True):
    """
    Tính DTW distance matrix n×n.
    window: Sakoe-Chiba constraint — giới hạn warping tối đa.
    Normalize về [0,1] để so sánh giữa các window khác nhau.
    """
    n = len(series_list)
    D = np.zeros((n, n))
    if verbose:
        print(f"Tính DTW (window={window}) cho {n} quốc gia ({n*(n-1)//2} cặp)...")
    for i in range(n):
        for j in range(i+1, n):
            d = dtw_lib.distance(
                series_list[i].astype(np.float64),
                series_list[j].astype(np.float64),
                window=window  # Sakoe-Chiba constraint
            )
            D[i, j] = D[j, i] = d
    mx = D.max()
    return D / mx if mx > 0 else D  # normalize [0,1]

# [Fix 2] So sánh 3 mức window — chọn theo Cophenetic Correlation
results_window = {}
for w in [30, 60, 90]:
    D    = compute_dtw_matrix(series_list, COUNTRIES, window=w, verbose=False)
    cond = squareform(D)
    Z    = linkage(cond, method='complete')
    cop, _ = cophenet(Z, cond)
    results_window[w] = {'matrix': D, 'linkage': Z, 'cophenetic': cop}
    print(f"window={w:3d}: Cophenetic = {cop:.4f}")

best_window = max(results_window, key=lambda w: results_window[w]['cophenetic'])
print(f"\n→ Chọn window = {best_window} ngày (Cophenetic cao nhất)")

dtw_matrix = results_window[best_window]['matrix']
Z_complete = results_window[best_window]['linkage']

# So sánh Complete vs Ward Linkage
Z_ward    = linkage(squareform(dtw_matrix), method='ward')
cop_c, _ = cophenet(Z_complete, squareform(dtw_matrix))
cop_w, _ = cophenet(Z_ward,     squareform(dtw_matrix))
print(f"  Complete : {cop_c:.4f}")
print(f"  Ward     : {cop_w:.4f}")
print(f"  → Sử dụng: {'Complete' if cop_c >= cop_w else 'Ward'}")
```

**Output thực tế:**

```
window= 30: Cophenetic = 0.7213
window= 60: Cophenetic = 0.7285  ← Cao nhất
window= 90: Cophenetic = 0.5366

→ Chọn window = 60 ngày (Cophenetic cao nhất)

Linkage so sánh (window=60):
  Complete : 0.7285
  Ward     : 0.5796
  → Sử dụng: Complete
```

**DTW vs Euclidean Distance:**

| Metric        | Ưu điểm                                        | Nhược điểm            | Phù hợp khi                      |
| ------------- | ---------------------------------------------- | --------------------- | -------------------------------- |
| **DTW**       | Bỏ qua time shift, capture sequence similarity | Chậm O(n²m)           | Series lệch thời gian ← **CHỌN** |
| **Euclidean** | Nhanh O(mn)                                    | Nhạy cảm với time lag | Series đồng bộ                   |

→ Đỉnh dịch Mỹ tháng 9/2021 và Nhật tháng 8/2021 lệch 4 tuần → DTW nhận ra chúng giống nhau, Euclidean không thể.

**Cophenetic Correlation Coefficient** đo mức độ dendrogram phản ánh đúng ma trận distance gốc (1.0 = hoàn hảo). Window=60 cao hơn window=30 vì dịch bệnh giữa các khu vực địa lý thường lệch 1-2 tháng.

#### Bước 4: Chọn số cụm tối ưu bằng Silhouette — \[Fix 3]

```python
print("=== Silhouette Score theo số cụm k ===")
sil_scores = {}
for k in range(2, 7):
    labels = fcluster(Z_complete, k, criterion='maxclust')
    # Kiểm tra không có singleton (cụm chỉ 1 phần tử)
    counts       = pd.Series(labels).value_counts()
    has_singleton = (counts == 1).any()
    if has_singleton or len(np.unique(labels)) < k:
        sil_scores[k] = -1
        print(f"  k={k}: Có singleton cluster — bỏ qua")
        continue
    # metric='precomputed' vì dùng DTW distance matrix sẵn có
    score = silhouette_score(dtw_matrix, labels, metric='precomputed')
    sil_scores[k] = score
    note = " ← Cao nhất" if score == max(v for v in sil_scores.values() if v > -1) else ""
    print(f"  k={k}:  {score:.4f}  {note}")

best_k     = max(sil_scores, key=lambda k: sil_scores[k])
N_CLUSTERS = best_k
print(f"\n→ Chọn k = {best_k} (Silhouette = {sil_scores[best_k]:.4f})")
```

**Output thực tế:**

```
=== Silhouette Score theo số cụm k ===
  k=2:  0.3489
  k=3:  0.3554  ← Cao nhất
  k=4:  0.2357
  k=5:  0.3080
  k=6: Có singleton cluster — bỏ qua

→ Chọn k = 3 (Silhouette = 0.3554)
```

**Silhouette Score ∈ \[-1, 1]:**

| Giá trị | Ý nghĩa                                            |
| ------- | -------------------------------------------------- |
| Gần 1   | Điểm trong cluster gần nhau, xa cluster khác (tốt) |
| Gần 0   | Điểm nằm ở biên giới giữa hai cluster              |
| Âm      | Điểm bị phân nhầm cluster                          |

→ 0.3554 là kết quả **tốt cho dữ liệu thực tế** — COVID là hiện tượng liên tục, không có ranh giới cluster tuyệt đối.

#### Bước 5: Dendrogram và gán nhãn cluster

```python
cluster_labels_arr = fcluster(Z_complete, N_CLUSTERS, criterion='maxclust')

cluster_result = pd.DataFrame({
    'country': COUNTRIES,
    'cluster': cluster_labels_arr
}).sort_values('cluster').reset_index(drop=True)

# Visualize dendrogram
fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(
    Z_complete,
    labels=COUNTRIES,
    leaf_rotation=45,
    leaf_font_size=9,
    color_threshold=dtw_matrix.max() * 0.6,  # ngưỡng màu cluster
    ax=ax
)
ax.set_ylabel('DTW Distance (normalized)')
ax.set_title(f'Dendrogram — Hierarchical Clustering (DTW window={best_window}, k={N_CLUSTERS})',
             fontweight='bold')
plt.savefig('../models_and_results/dendrogram.png', dpi=150)
```

**Output thực tế — 3 clusters:**

```
Cluster 1 "Làn sóng nhỏ"   (27 nước): Argentina, Australia, Bangladesh, Brazil,
                             Canada, Chile, Colombia, Egypt, India, Iran,
                             Indonesia, Italy, Kenya, Mexico, Morocco, Nigeria,
                             Norway, Pakistan, Peru, Philippines, Poland, Russia,
                             South Africa, Sweden, Turkey, UK, USA
Cluster 2 "Kiểm soát tốt"  (3 nước) : China, Japan, South Korea
Cluster 3 "Bùng phát vừa"  (9 nước) : France, Germany, Malaysia, Netherlands,
                             New Zealand, Spain, Thailand, Ukraine, Vietnam
```

#### Bước 6: Phân tích và đặt tên cluster — \[Fix 4]

```python
df_cluster['cfr'] = df_cluster['total_deaths'] / df_cluster['total_cases'].replace(0, np.nan)
cluster_result_full = pd.merge(df_cluster, cluster_result, on='country')

cluster_stats = cluster_result_full.groupby('cluster').agg(
    n_countries     = ('country',                   'nunique'),
    cases_mean      = ('new_cases_custom_smoothed', 'mean'),
    deaths_mean     = ('new_deaths_smoothed',        'mean'),
    stringency_mean = ('stringency_index',           'mean'),
    cfr_mean        = ('cfr',                        'mean'),
).round(4)

# [Fix 4] Đặt tên semantic dựa trên cases_mean (thấp → cao)
sorted_clusters  = cluster_stats['cases_mean'].sort_values().index.tolist()
semantic_names   = [
    'Cluster Kiểm soát tốt',  # cases thấp nhất
    'Cluster Làn sóng nhỏ',
    'Cluster Bùng phát vừa',
    'Cluster Bùng phát lớn',  # cases cao nhất
]
cluster_name_map = {}
for rank, cid in enumerate(sorted_clusters):
    name = semantic_names[rank] if rank < len(semantic_names) else f'Cluster {cid}'
    cluster_name_map[cid] = name

cluster_result['cluster_name'] = cluster_result['cluster'].map(cluster_name_map)

# Lưu để Hybrid model đọc lại
cluster_result.to_csv('../models_and_results/cluster_assignment.csv', index=False)
```

**Output thực tế — thống kê 3 clusters:**

```
         n_countries  cases_mean  deaths_mean  stringency_mean  cfr_mean
cluster
1 "Làn sóng nhỏ"    27   3748.75      84.72         32.63       0.0234
2 "Kiểm soát tốt"    3   1935.94      34.62         41.16       0.0104
3 "Bùng phát vừa"    9   5500.75      35.53         27.29       1.7273
```

**Phân tích ý nghĩa clusters:**

| Cluster           | Profile                                                               | Lý giải                                              |
| ----------------- | --------------------------------------------------------------------- | ---------------------------------------------------- |
| **Kiểm soát tốt** | Cases thấp nhất (1,936), stringency cao nhất (41.2), CFR thấp (0.010) | Zero-COVID policy — China, Japan, South Korea        |
| **Làn sóng nhỏ**  | Cases trung bình (3,749), deaths cao nhất (84.7)                      | Pattern nhiều làn sóng nhỏ liên tiếp                 |
| **Bùng phát vừa** | Cases cao nhất (5,501) nhưng deaths thấp (35.5)                       | Mở cửa muộn + Omicron peak — Việt Nam thuộc nhóm này |

#### Bước 7 & 8: Cluster Profiles và DTW Heatmap

```python
# Cluster Profiles — pattern dịch trung bình mỗi cluster
colors = ['#1D9E75', '#378ADD', '#EF9F27']  # xanh lá, xanh dương, cam

fig, axes = plt.subplots(1, n_clust, figsize=(4*n_clust, 5), sharey=True)
for ax, (k, color) in zip(axes, zip(unique_clusters, colors)):
    members = cluster_result[cluster_result['cluster']==k]['country'].tolist()
    avail   = [c for c in members if c in normalized_df.columns]
    arr     = np.array([normalized_df[c].values for c in avail])
    mean_s  = arr.mean(axis=0)
    std_s   = arr.std(axis=0)

    for s in arr:
        ax.plot(s, color=color, lw=0.8, alpha=0.15)  # individual curves (mờ)
    ax.plot(mean_s, color=color, lw=2.5, label='Mean')  # mean curve (đậm)
    ax.fill_between(range(len(mean_s)),
                    np.maximum(mean_s - std_s, 0), mean_s + std_s,
                    alpha=0.15, color=color)   # ±1 std band
    if vax_idx:
        ax.axvline(vax_idx, color='gray', lw=1.5,
                   linestyle='--', alpha=0.7)  # vaccine milestone

# DTW Heatmap — sắp xếp theo cluster để thấy block structure
order    = cluster_result.sort_values('cluster')['country'].tolist()
idx      = [COUNTRIES.index(c) for c in order if c in COUNTRIES]
D_sorted = dtw_matrix[np.ix_(idx, idx)]

sns.heatmap(
    pd.DataFrame(D_sorted, index=order, columns=order),
    annot=False, cmap='RdYlGn_r', ax=ax,
    cbar_kws={'label': 'DTW Distance (normalized)'}
)
```

**Output thực tế:**

```
=== TÓM TẮT ===
Window tối ưu    : 60 ngày
Cophenetic Corr. : 0.7285
Số cụm tối ưu k  : 3 (Silhouette = 0.3554)
Đã lưu: cluster_assignment.csv, cluster_profiles.png,
        dtw_heatmap.png, dendrogram.png
```

***
# 4. Phân tích thuật toán ARIMA
## 4.1. Tiền xử lý dữ liệu
Quá trình tiền xử lý dữ liệu đóng vai trò quan trọng trong việc đảm bảo chất lượng dữ liệu đầu vào cho các mô hình học máy. Trong bài toán phân tích dữ liệu COVID-19 theo chuỗi thời gian, dữ liệu ban đầu được thu thập từ nhiều nguồn, có thể tồn tại giá trị thiếu, nhiễu và không đồng nhất. Một **pipeline tiền xử lý gồm 8 bước** đã được xây dựng nhằm chuẩn hóa và làm giàu dữ liệu.

---

### Bước 1: Đọc và kết hợp dữ liệu
Hai file dữ liệu gốc `covid_data.csv` và `column_final_data.csv` được đọc. Hai cột `ICU` và `hospital` được trích xuất từ tập dữ liệu gốc và merge vào tập dữ liệu chính theo khóa `country` và `date`.

> 📷 **Hình 2.1:** *Code đọc và kết hợp dữ liệu
```python
# =================================================================
# 1. Đọc dữ liệu
# =================================================================
df_final = pd.read_csv("column_final_data.csv", parse_dates=['date'])
df_raw   = pd.read_csv("covid_data.csv", parse_dates=['date'])

# =================================================================
# 2. Chọn 2 cột cần lấy từ file gốc
# =================================================================
cols_needed = [
    'country',
    'date',
    'icu_patients_per_million',
    'hosp_patients_per_million'
]

df_extra = df_raw[cols_needed]

# =================================================================
# 3. Merge vào dataset đã xử lý
# =================================================================
df_merged = pd.merge(
    df_final,
    df_extra,
    on=['country', 'date'],
    how='left'
)

# =================================================================
# 4. Kiểm tra
# =================================================================
print("Sau khi merge:", df_merged.shape)
print(df_merged[['icu_patients_per_million', 'hosp_patients_per_million']].isnull().sum())
```

---

### Bước 2: Xử lý giá trị thiếu (Missing Values)
Các giá trị thiếu ở cột `ICU` và `hospital` được xử lý theo từng quốc gia bằng phương pháp **Forward Fill (ffill)**, và điền bằng `0` nếu không có giá trị trước đó.

> 📷 **Hình 2.2:** *Code xử lý giá trị thiếu*
 ```python
# =================================================================
# 5. Xử lý missing (rất quan trọng)
# =================================================================

# forward fill theo từng quốc gia
for col in ['icu_patients_per_million', 'hosp_patients_per_million']:
    df_merged[col] = df_merged.groupby('country')[col].ffill()
    df_merged[col] = df_merged[col].fillna(0)
```

---

### Bước 3: Lọc và chuẩn hóa theo quốc gia
Loại bỏ các dòng không thuộc về quốc gia cụ thể, tiến hành lọc ra 7 quốc gia tiêu biểu phục vụ nghiên cứu: **Mỹ, Trung Quốc, Ấn Độ, Brazil, Anh, Việt Nam và Nam Phi**.

> 📷 **Hình 2.3:** *Code lọc 7 quốc gia*
```python
# =================================================================
# # 2. CẤU HÌNH
# =================================================================
INPUT_FILE = "column_final_data_full.csv"
OUTPUT_FILE = "preprocessed_data.csv"

COUNTRIES = [
    'United States', 'China', 'India', 'Brazil',
    'United Kingdom', 'Vietnam', 'South Africa'
]

# =================================================================
# # 3. HÀM TIỀN XỬ LÝ
# =================================================================
def preprocess_data(file_path):
    print("📁 Đang đọc dữ liệu...")
    df = pd.read_csv(file_path, parse_dates=['date'])
    
    print("Kích thước ban đầu:", df.shape)
    
    # -------------------------------------------------------------
    # # 1. Xóa dữ liệu không phải quốc gia
    # -------------------------------------------------------------
    df = df.dropna(subset=['continent'])
    
    # -------------------------------------------------------------
    # # 2. Lọc quốc gia
    # -------------------------------------------------------------
    df = df[df['country'].isin(COUNTRIES)]
```
---

### Bước 4: Sắp xếp theo thời gian
Sắp xếp dữ liệu theo thứ tự ưu tiên `country` và `date` để đảm bảo tính đúng đắn của chuỗi thời gian, tránh gây sai lệch khi tính toán các đặc trưng trễ (lag) hoặc cửa sổ trượt (rolling).

> 📷 **Hình 2.4:** *Code sắp xếp thời gian*
```python
df = df.sort_values(by=['country','date']).
```
---

### Bước 5: Xử lý giá trị âm
Các cột dữ liệu dạng số ca mắc mới (`new_`) được kiểm tra và thay thế toàn bộ các giá trị âm (nếu có) bằng `0`.

> 📷 **Hình 2.5:** *Code xử lý giá trị âm*
```python
# -------------------------------------------------------------
    # # 4. Xử lý giá trị âm (new_ columns)
    # -------------------------------------------------------------
    new_cols = [col for col in df.columns if 'new_' in col]
    
    for col in new_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            df.loc[df[col] < 0, col] = 0
```
---

### Bước 6: Xử lý thiếu theo nhóm biến
Quy tắc xử lý giá trị thiếu (`NaN`) được áp dụng linh hoạt theo từng nhóm thuộc tính:
* **Cột tích lũy (`total_`, `people_`):** Áp dụng `forward fill` $\rightarrow$ các giá trị còn lại điền `0`.
* **Cột theo ngày (`new_`):** Điền trực tiếp bằng `0`.
* **Cột tĩnh nhân khẩu học (`population`, `GDP`,...):** Điền bằng giá trị `median` (trung vị) theo từng quốc gia $\rightarrow$ các phần còn lại điền `median` toàn bộ tập dữ liệu.
* **Các cột chỉ số khác (`stringency`, `R`, `positive_rate`):** Áp dụng `forward fill` $\rightarrow$ các ô trống còn lại điền `median`.

> 📷 **Hình 2.6:** *Code xử lý thiếu theo nhóm*
```python
# -------------------------------------------------------------
    # # 5. Xử lý missing values
    # -------------------------------------------------------------
    
    # # A. Cột tích lũy
    cumulative_cols = [col for col in df.columns if 'total_' in col or col.startswith('people_')]
    for col in cumulative_cols:
        df[col] = df.groupby('country')[col].ffill()
        df[col] = df[col].fillna(0)
        
    # # B. Cột daily
    for col in new_cols:
        df[col] = df[col].fillna(0)
        
    # # C. Cột tĩnh
    static_cols = ['population', 'population_density', 'gdp_per_capita']
    for col in static_cols:
        if col in df.columns:
            df[col] = df.groupby('country')[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df[col].median())
            
    # # D. Các cột khác
    other_cols = ['stringency_index', 'reproduction_rate', 'positive_rate', 'tests_per_case']
    for col in other_cols:
        if col in df.columns:
            df[col] = df.groupby('country')[col].ffill()
            df[col] = df[col].fillna(df[col].median())
            
    print("Kích thước sau xử lý:", df.shape)
    
    return df
```
---

### Bước 7: Feature Engineering (Tạo đặc trưng mới)
Tiến hành làm giàu tập dữ liệu phục vụ mô hình:
* Làm mượt dữ liệu 7 ngày (`rolling mean`) cho các cột: `new_cases`, `new_deaths`, `new_cases_per_million`.
* Tạo snapshot cuối kỳ: Tỷ lệ tử vong (`CFR`), Logarit của GDP (`GDP log`).
* Phân chia dòng thời gian: Giai đoạn trước và sau khi có vaccine (`Pre/Post-vaccine`).
* Tạo các đặc trưng trễ (`lag1`, `lag7`, `lag14`) và độ lệch chuẩn trượt 7 ngày (`rolling std 7 ngày`).

> 📷 **Hình 2.7:** *Code tạo đặc trưng mới*
```python
df = df.sort_values(['country', 'date']).reset_index(drop=True)  # sắp xếp dữ liệu đảm bảo dữ liệu theo đúng thời gian

# — 3.1 Smoothed columns ————————————————————————————————————  # làm mượt dữ liệu
for col in ['new_cases', 'new_deaths', 'new_cases_per_million']:
    smooth_col = f'{col}_smoothed_7d'                          # tính trung bình 7 ngày
    if smooth_col not in df.columns:
        df[smooth_col] = (
            df.groupby('country')[col]
            .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        )
        
print("✅ Smoothed cols:", [c for c in df.columns if 'smoothed_7d' in c])
```
---

### Bước 8: Lưu dữ liệu
File dữ liệu sau khi làm sạch hoàn chỉnh được xuất ra thành file `covid_analysis_ready.csv` để sẵn sàng đưa vào huấn luyện các mô hình học máy
## 4.2. Phân loại giai đoạn dịch COVID-19 bằng thuật toán ARIMA
### 4.2.1. Khái niệm về chuỗi thời gian dừng (Stationary)
Một chuỗi thời gian được gọi là **dừng** (*stationary*) khi các tính chất thống kê của nó — bao gồm giá trị trung bình, phương sai và cấu trúc tự tương quan — không thay đổi theo thời gian. Đây là điều kiện tiên quyết để áp dụng mô hình ARIMA, bởi các tham số của mô hình chỉ có ý nghĩa ổn định khi chuỗi dữ liệu thỏa mãn tính dừng. 

Trong thực tế, nhiều chuỗi thời gian (như dữ liệu tiêm chủng tích lũy) thường có xu hướng tăng dần theo thời gian, do đó cần được biến đổi trước khi đưa vào mô hình.

Để kiểm định tính dừng, nghiên cứu sử dụng đồng thời hai kiểm định thống kê bổ sung cho nhau:

* **Kiểm định ADF (Augmented Dickey-Fuller):** 
  * Giả thuyết $H_0$: Chuỗi có nghiệm đơn vị (*non-stationary*).
  * Quy tắc quyết định: Nếu $p\text{-value} < 0.05$, bác bỏ $H_0 \rightarrow$ Kết luận chuỗi là **dừng**.

* **Kiểm định KPSS (Kwiatkowski-Phillips-Schmidt-Shin):** 
  * Giả thuyết $H_0$: Chuỗi là dừng (*stationary*).
  * Quy tắc quyết định: Nếu $p\text{-value} > 0.05$, không bác bỏ $H_0 \rightarrow$ Kết luận chuỗi là **dừng**.

> **Lưu ý:** Kết luận về tính dừng chỉ được xác nhận chắc chắn khi **cả hai kiểm định đồng thuận**, tránh trường hợp kết quả mâu thuẫn do đặc thù của từng phương pháp.

---

### 4.2.2. Cấu trúc mô hình ARIMA(p, d, q)

Mô hình ARIMA bao gồm ba thành phần chính, được ký hiệu bằng bộ tham số $(p, d, q)$:

#### 1. Thành phần AR — AutoRegressive (Tự hồi quy) — Bậc $p$
Thành phần AR mô tả mối quan hệ tuyến tính giữa giá trị hiện tại và $p$ giá trị quá khứ của chuỗi. 

Mô hình $\text{AR}(p)$ có dạng:
$$Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \dots + \phi_p Y_{t-p} + \varepsilon_t$$

* Trong đó: $\phi_1, \phi_2, \dots, \phi_p$ là các hệ số tự hồi quy, $c$ là hằng số, và $\varepsilon_t$ là nhiễu trắng (*white noise*).
* **Bậc $p$** được xác định dựa trên đồ thị **PACF** (*Partial Autocorrelation Function*) — bậc $p$ tương ứng với độ trễ (*lag*) cuối cùng còn có ý nghĩa thống kê trên PACF.

#### 2. Thành phần I — Integrated (Tích hợp sai phân) — Bậc $d$
Thành phần I thể hiện số lần lấy sai phân cần thiết để chuyển chuỗi không dừng thành chuỗi dừng. 

Phép lấy sai phân bậc 1 được định nghĩa là:
$$\Delta Y_t = Y_t - Y_{t-1}$$

Trong nghiên cứu này, chuỗi `people_vaccinated_per_hundred` là chuỗi tích lũy tăng dần, do đó cần lấy sai phân bậc 1 ($d = 1$) để chuyển về chuỗi biểu diễn tốc độ tiêm chủng hàng tuần. Bậc $d$ được xác định thông qua kết quả kiểm định ADF và KPSS trên chuỗi gốc cùng các chuỗi sai phân.

#### 3. Thành phần MA — Moving Average (Trung bình trượt) — Bậc $q$
Thành phần MA mô tả sự phụ thuộc của giá trị hiện tại vào $q$ sai số ngẫu nhiên trong quá khứ. 

Mô hình $\text{MA}(q)$ có dạng:
$$Y_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + \dots + \theta_q \varepsilon_{t-q}$$

* Trong đó: $\theta_1, \theta_2, \dots, \theta_q$ là các hệ số trung bình trượt và $\mu$ là giá trị trung bình của chuỗi.
* **Bậc $q$** được xác định dựa trên đồ thị **ACF** (*Autocorrelation Function*) — bậc $q$ tương ứng với độ trễ (*lag*) cuối cùng còn có ý nghĩa thống kê trên ACF.

---

#### Mô hình ARIMA(p, d, q) tổng quát
$$
\Delta^d Y_t = c + \sum_{i=1}^{p} \phi_i \Delta^d Y_{t-i} + \varepsilon_t + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j}
$$
### Chuẩn bị dữ liệu
Sau khi đã tiến hành tiền xử lý dữ liệu chung cho tập dữ liệu ban đầu là covid_data.csv thì tiếp theo chúng ta sẽ tiến hành xử lý cho mô hình ARIMA.
#### Bước A: Weekly Resampling 
Bước này được dùng để chuyển dữ liệu COVID từ dạng theo ngày (daily) sang dạng theo tuần (weekly) bằng phương pháp resample("W-MON"). Mục đích chính là giảm nhiễu dữ liệu, giảm số lượng bản ghi và giúp quá trình phân tích hoặc huấn luyện mô hình Machine Learning hiệu quả hơn. Trong quá trình resample, các nhóm cột sẽ được xử lý khác nhau: những cột mang tính cố định như dân số, GDP hay tuổi trung vị sẽ lấy giá trị đầu tuần (first()); các cột mang tính tích luỹ như tổng ca nhiễm, tổng ca tử vong hay số người tiêm vaccine sẽ lấy giá trị cuối tuần (last()); còn các dữ liệu thay đổi hằng ngày sẽ được lấy trung bình theo tuần (mean()). Sau khi gom dữ liệu, chương trình tiếp tục kiểm tra và điền các giá trị bị thiếu bằng phương pháp forward-fill và backward-fill để đảm bảo dữ liệu hoàn chỉnh trước khi đưa vào phân tích hoặc xây dựng mô hình.
```python
print("\n" + "=" * 60)
print("BƯỚC 6 — Weekly Resampling (W-MON)")
print("=" * 60)

RESAMPLE_FIRST = safe_cols([
    "code", "continent", "population", "population_density",
    "median_age", "life_expectancy", "gdp_per_capita",
    "extreme_poverty", "diabetes_prevalence", "hospital_beds_per_thousand",
], df)

RESAMPLE_LAST = safe_cols([
    "total_cases", "total_cases_per_million",
    "total_deaths", "total_deaths_per_million",
    "people_vaccinated", "people_vaccinated_per_hundred",
    "people_fully_vaccinated", "people_fully_vaccinated_per_hundred",
], df)

already_assigned = set(RESAMPLE_FIRST + RESAMPLE_LAST + ["country", "date"])
RESAMPLE_MEAN = [
    c for c in df.columns
    if c not in already_assigned
    and df[c].dtype in [np.float64, np.int64, float, int]
]

print(f"\n  first (static)   : {len(RESAMPLE_FIRST)} cột")
print(f"  last  (luỹ kế)   : {len(RESAMPLE_LAST)} cột")
print(f"  mean  (hàng ngày): {len(RESAMPLE_MEAN)} cột")

def resample_country(group):
    group = group.set_index("date")
    agg_first = group[RESAMPLE_FIRST].resample("W-MON").first() if RESAMPLE_FIRST else pd.DataFrame()
    agg_last  = group[RESAMPLE_LAST].resample("W-MON").last()   if RESAMPLE_LAST  else pd.DataFrame()
    agg_mean  = group[RESAMPLE_MEAN].resample("W-MON").mean()   if RESAMPLE_MEAN  else pd.DataFrame()
    combined  = pd.concat([agg_first, agg_last, agg_mean], axis=1)
    combined["country"] = group["country"].iloc[0]
    return combined.reset_index().rename(columns={"date": "week_start"})

print("\n  Đang resample...")
weekly_parts = []
for country in countries:
    grp = df[df["country"] == country].copy()
    weekly_parts.append(resample_country(grp))

df_weekly = pd.concat(weekly_parts, ignore_index=True)

cols_order = ["country", "week_start"] + [
    c for c in df_weekly.columns if c not in ("country", "week_start")
]
df_weekly = df_weekly[cols_order]

print(f"\n  Daily  : {df.shape[0]:,} hàng × {df.shape[1]} cột")
print(f"  Weekly : {df_weekly.shape[0]:,} hàng × {df_weekly.shape[1]} cột")
print(f"  Range  : {df_weekly['week_start'].min().date()} → {df_weekly['week_start'].max().date()}")

print("\n  Số tuần mỗi quốc gia:")
for country in countries:
    n = len(df_weekly[df_weekly["country"] == country])
    print(f"    {country:<20} {n:>4} tuần")

# Fill missing phát sinh sau resample
missing_after = df_weekly.isnull().sum()
missing_after = missing_after[missing_after > 0]
if missing_after.empty:
    print("\n  ✅ Không có missing sau resample!")
else:
    print(f"\n  ⚠️  {len(missing_after)} cột missing sau resample — forward-fill lại:")
    df_weekly[missing_after.index.tolist()] = (
        df_weekly.groupby("country")[missing_after.index.tolist()]
        .transform(lambda x: x.ffill().bfill())
    )
    print("  ✅ Đã fill xong!")
```
Kết quả thu được
```python
============================================================
BƯỚC 6 — Weekly Resampling (W-MON)
============================================================

  first (static)   : 10 cột
  last  (luỹ kế)   : 8 cột
  mean  (hàng ngày): 8 cột

  Đang resample...

  Daily  : 15,715 hàng × 28 cột
  Weekly : 2,247 hàng × 28 cột
  Range  : 2020-01-06 → 2026-02-23

  Số tuần mỗi quốc gia:
    Vietnam              321 tuần
    United States        321 tuần
    China                321 tuần
    United Kingdom       321 tuần
    Brazil               321 tuần
    India                321 tuần
    South Africa         321 tuần

  ✅ Không có missing sau resample!
```
#### Bước B: Data Transformation (Differencing)
Bước này được sử dụng để biến đổi dữ liệu bằng phương pháp First-order Differencing nhằm chuyển dữ liệu tích luỹ về dạng thay đổi theo thời gian. Cột mục tiêu TARGET_COL là dữ liệu tích luỹ và bị giới hạn trong khoảng từ 0 đến 100 nên không phù hợp để áp dụng phép biến đổi log. Vì vậy, chương trình sử dụng diff(1) để tính hiệu giữa giá trị tuần hiện tại và tuần trước, từ đó tạo ra cột mới biểu diễn tốc độ thay đổi theo tuần, ví dụ như số phần trăm người được tiêm vaccine tăng thêm mỗi tuần. Việc này giúp chuỗi dữ liệu trở nên ổn định hơn, giảm xu hướng tăng dần liên tục của dữ liệu tích luỹ và hỗ trợ các mô hình phân tích chuỗi thời gian hoặc Machine Learning học hiệu quả hơn. Sau khi tính sai phân, các giá trị bị thiếu ở tuần đầu tiên sẽ được thay bằng 0, rồi chương trình in ra các thống kê như giá trị nhỏ nhất, lớn nhất, trung bình và độ lệch chuẩn cho từng quốc gia để kiểm tra đặc điểm của dữ liệu sau biến đổi.
```python
print("\n" + "=" * 60)
print("BƯỚC 7 — Data Transformation (First-order Differencing)")
print("=" * 60)
print(f"""
  Target : '{TARGET_COL}'
  Loại   : luỹ kế, giới hạn [0, 100]
  Lý do  : Log transform không phù hợp vì chuỗi bị chặn tại 100
  Cách   : diff(1) → tốc độ tiêm chủng mỗi tuần
""")

DIFF_COL  = f"{TARGET_COL}_diff1"
DIFF2_COL = f"{TARGET_COL}_diff2"

for country in countries:
    mask = df_weekly["country"] == country
    df_weekly.loc[mask, DIFF_COL] = df_weekly.loc[mask, TARGET_COL].diff(1)

df_weekly[DIFF_COL] = df_weekly.groupby("country")[DIFF_COL].transform(
    lambda x: x.fillna(0)
)

print(f"  {'Quốc gia':<20} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8}")
print("  " + "-" * 56)
for country in countries:
    sub = df_weekly[df_weekly["country"] == country][DIFF_COL]
    print(f"  {country:<20} {sub.min():>8.3f} {sub.max():>8.3f} {sub.mean():>8.3f} {sub.std():>8.3f}")
```
Kết quả thu được
```python
============================================================
BƯỚC 7 — Data Transformation (First-order Differencing)
============================================================

  Target : 'people_vaccinated_per_hundred'
  Loại   : luỹ kế, giới hạn [0, 100]
  Lý do  : Log transform không phù hợp vì chuỗi bị chặn tại 100
  Cách   : diff(1) → tốc độ tiêm chủng mỗi tuần

  Quốc gia                 Min      Max     Mean      Std
  --------------------------------------------------------
  Vietnam                0.000    7.429    0.283    0.953
  United States          0.000    4.109    0.246    0.636
  China                  0.000   43.644    0.288    3.043
  United Kingdom         0.000    5.116    0.246    0.778
  Brazil                 0.000    4.167    0.281    0.731
  India                  0.000    2.959    0.225    0.546
  South Africa           0.000    2.366    0.121    0.321
```
#### Bước C- Kiểm tra stationary (ADF + KPSS)
Bước này được dùng để kiểm tra tính stationary (tính dừng) của chuỗi dữ liệu thời gian sau khi đã biến đổi bằng sai phân. Một chuỗi stationary là chuỗi có đặc điểm thống kê ổn định theo thời gian như trung bình và phương sai không thay đổi, đây là điều kiện rất quan trọng đối với nhiều mô hình phân tích chuỗi thời gian và dự báo. Chương trình sử dụng hai kiểm định phổ biến là ADF (Augmented Dickey-Fuller) và KPSS để đánh giá. Với ADF, nếu p-value < 0.05 thì chuỗi được xem là stationary; còn với KPSS, nếu p-value > 0.05 thì chuỗi cũng được xem là stationary. Kết luận chỉ được chấp nhận khi cả hai kiểm định đồng thuận. Chương trình sẽ kiểm tra cả chuỗi gốc và chuỗi sau sai phân bậc 1 (diff(1)), sau đó in ra các giá trị thống kê và kết luận cho từng quốc gia. Nếu sau diff(1) mà chuỗi vẫn chưa stationary, hệ thống sẽ tự động tiếp tục thực hiện sai phân bậc 2 (diff(2)) rồi kiểm tra lại để đảm bảo dữ liệu đạt trạng thái phù hợp trước khi đưa vào mô hình dự báo hoặc Machine Learning.
```python
print("\n" + "=" * 60)
print("BƯỚC 8 — Kiểm tra Stationarity")
print("=" * 60)
print("""
  ADF  : H0 = có unit root → p < 0.05 → Stationary ✅
  KPSS : H0 = stationary   → p > 0.05 → Stationary ✅
  Kết luận chắc chắn khi CẢ HAI đồng thuận.
""")

stat_results = []

for country in countries:
    sub = df_weekly[df_weekly["country"] == country]
    stat_results.append(test_stationarity(sub[TARGET_COL], "original", country))
    stat_results.append(test_stationarity(sub[DIFF_COL],   "diff(1)",  country))

print(f"  {'Quốc gia':<20} {'Chuỗi':<12} {'ADF p':>8} {'ADF':>12} {'KPSS p':>8} {'KPSS':>12}  Kết luận")
print("  " + "-" * 100)
for r in stat_results:
    adf_p_str  = f"{r['adf_p']:.4f}"  if r['adf_p']  is not None else "  N/A  "
    kpss_p_str = f"{r['kpss_p']:.4f}" if r['kpss_p'] is not None else "  N/A  "
    print(
        f"  {r['country']:<20} {r['series']:<12} "
        f"{adf_p_str:>8} {r['adf_stat']:>12} "
        f"{kpss_p_str:>8} {r['kpss_stat']:>12}  {r['conclusion']}"
    )

# Tự động thêm diff(2) nếu cần
needs_diff2 = [
    r["country"] for r in stat_results
    if r["series"] == "diff(1)" and "NON-STATIONARY" in r["conclusion"]
]

if not needs_diff2:
    print("\n  ✅ Tất cả stationary sau diff(1)")
else:
    print(f"\n  ⚠️  {len(needs_diff2)} quốc gia cần diff(2): {needs_diff2}")
    for country in needs_diff2:
        mask = df_weekly["country"] == country
        df_weekly.loc[mask, DIFF2_COL] = df_weekly.loc[mask, DIFF_COL].diff(1)
        df_weekly[DIFF2_COL] = df_weekly.groupby("country")[DIFF2_COL].transform(
            lambda x: x.fillna(0)
        )
    print(f"\n  Kết quả sau diff(2):")
    print(f"  {'Quốc gia':<20} {'ADF p':>8} {'KPSS p':>8}  Kết luận")
    print("  " + "-" * 60)
    for country in needs_diff2:
        sub = df_weekly[df_weekly["country"] == country]
        r2  = test_stationarity(sub[DIFF2_COL], "diff(2)", country)
        print(f"  {country:<20} {r2['adf_p']:>8.4f} {r2['kpss_p']:>8.4f}  {r2['conclusion']}")
```
Kết quả thu được 
```python
============================================================
BƯỚC 8 — Kiểm tra Stationarity
============================================================

  ADF  : H0 = có unit root → p < 0.05 → Stationary ✅
  KPSS : H0 = stationary   → p > 0.05 → Stationary ✅
  Kết luận chắc chắn khi CẢ HAI đồng thuận.

  Quốc gia             Chuỗi           ADF p           ADF   KPSS p         KPSS  Kết luận
  ----------------------------------------------------------------------------------------------------
  Vietnam              original       0.3786   Non-stat ❌   0.0100   Non-stat ❌  ❌ NON-STATIONARY
  Vietnam              diff(1)        0.0621   Non-stat ❌   0.0998     Stat ✅  ⚠️ KẾT QUẢ MÂU THUẪN — cần xem xét thêm
  United States        original       0.1183   Non-stat ❌   0.0100   Non-stat ❌  ❌ NON-STATIONARY
  United States        diff(1)        0.0992   Non-stat ❌   0.0232   Non-stat ❌  ❌ NON-STATIONARY
  China                original       0.2854   Non-stat ❌   0.0100   Non-stat ❌  ❌ NON-STATIONARY
  China                diff(1)        0.0535   Non-stat ❌   0.0992     Stat ✅  ⚠️ KẾT QUẢ MÂU THUẪN — cần xem xét thêm
  United Kingdom       original       0.0898   Non-stat ❌   0.0100   Non-stat ❌  ❌ NON-STATIONARY
  United Kingdom       diff(1)        0.1155   Non-stat ❌   0.0307   Non-stat ❌  ❌ NON-STATIONARY
  Brazil               original       0.1930   Non-stat ❌   0.0100   Non-stat ❌  ❌ NON-STATIONARY
  Brazil               diff(1)        0.2475   Non-stat ❌   0.0360   Non-stat ❌  ❌ NON-STATIONARY
  India                original       0.1131   Non-stat ❌   0.0100   Non-stat ❌  ❌ NON-STATIONARY
  India                diff(1)        0.4388   Non-stat ❌   0.0188   Non-stat ❌  ❌ NON-STATIONARY
  South Africa         original       0.2818   Non-stat ❌   0.0100   Non-stat ❌  ❌ NON-STATIONARY
  South Africa         diff(1)        0.2086   Non-stat ❌   0.0384   Non-stat ❌  ❌ NON-STATIONARY

  ⚠️  5 quốc gia cần diff(2): ['United States', 'United Kingdom', 'Brazil', 'India', 'South Africa']

  Kết quả sau diff(2):
  Quốc gia                ADF p    KPSS p  Kết luận
  ------------------------------------------------------------
  United States          0.0000    0.1000  ✅ STATIONARY
  United Kingdom         0.0000    0.1000  ✅ STATIONARY
  Brazil                 0.0000    0.1000  ✅ STATIONARY
  India                  0.0000    0.1000  ✅ STATIONARY
  South Africa           0.0000    0.1000  ✅ STATIONARY
```
#### Bước D: Phân rã chuỗi thời gian (Seasonal Decompose)
Bước này được dùng để phân rã chuỗi thời gian nhằm tách dữ liệu thành các thành phần chính:

Xu hướng (Trend)

Mùa vụ (Seasonal)

Nhiễu (Residual)

Chương trình sử dụng mô hình cộng (additive) với chu kỳ mùa vụ là 52 tuần để phân tích theo năm. Hàm seasonal_decompose() được áp dụng để tính toán các tham số: trend_strength, seasonal_strength và residual_std. Kết quả được trực quan hóa thành biểu đồ 4 thành phần và lưu dưới dạng file ảnh cho từng quốc gia.
```python
print("\n" + "=" * 60)
print("BƯỚC 9 — Phân rã chuỗi thời gian (Seasonal Decompose)")
print("=" * 60)
print("  Model: additive | Period: 52 tuần\n")

decomp_summary = {}

for country in countries:
    sub = df_weekly[df_weekly["country"] == country].copy()
    sub = sub.set_index("week_start")[TARGET_COL].dropna()

    if len(sub) < 104:
        print(f"  ⚠️  {country:<20} — quá ít dữ liệu ({len(sub)} tuần), bỏ qua")
        decomp_summary[country] = {"trend_strength": 0, "seasonal_strength": 0, "residual_std": 0}
        continue

    try:
        decomp = seasonal_decompose(sub, model="additive", period=52, extrapolate_trend="freq")

        seasonal_strength = np.std(decomp.seasonal) / (
            np.std(decomp.seasonal) + np.std(decomp.resid.dropna())
        )
        trend_strength = 1 - np.var(decomp.resid.dropna()) / np.var(
            decomp.trend.dropna() + decomp.resid.dropna()
        )
        residual_std = np.std(decomp.resid.dropna())

        decomp_summary[country] = {
            "trend_strength":    round(trend_strength, 3),
            "seasonal_strength": round(seasonal_strength, 3),
            "residual_std":      round(residual_std, 3),
        }

        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        fig.suptitle(f"Decomposition — {country}", fontsize=13, fontweight="bold")

        axes[0].plot(sub.index, sub.values, color="#2563EB", linewidth=1.2)
        axes[0].set_ylabel("Original"); axes[0].grid(True, alpha=0.3)

        axes[1].plot(decomp.trend.index, decomp.trend.values, color="#16A34A", linewidth=1.5)
        axes[1].set_ylabel("Trend"); axes[1].grid(True, alpha=0.3)
        axes[1].annotate(f"Trend strength: {trend_strength:.2f}",
                         xy=(0.02, 0.85), xycoords="axes fraction",
                         fontsize=9, color="#16A34A", fontweight="bold")

        axes[2].plot(decomp.seasonal.index, decomp.seasonal.values, color="#D97706", linewidth=1.0)
        axes[2].set_ylabel("Seasonal"); axes[2].grid(True, alpha=0.3)
        axes[2].annotate(f"Seasonal strength: {seasonal_strength:.2f}",
                         xy=(0.02, 0.85), xycoords="axes fraction",
                         fontsize=9, color="#D97706", fontweight="bold")

        axes[3].plot(decomp.resid.index, decomp.resid.values, color="#DC2626", linewidth=0.8, alpha=0.8)
        axes[3].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[3].set_ylabel("Residual"); axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"decomposition_{country.replace(' ', '_')}.png"
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close()

        strength_label = "mạnh" if seasonal_strength > 0.4 else "yếu"
        print(f"  {country:<20} trend={trend_strength:.2f} | seasonal={seasonal_strength:.2f} ({strength_label}) → 💾 {fname}")

    except Exception as e:
        print(f"  ⚠️  {country:<20} — lỗi: {e}")
        decomp_summary[country] = {"trend_strength": 0, "seasonal_strength": 0, "residual_std": 0}
```
Kết quả thu được
```python
============================================================
BƯỚC 9 — Phân rã chuỗi thời gian (Seasonal Decompose)
============================================================
  Model: additive | Period: 52 tuần

  Vietnam              trend=0.98 | seasonal=0.30 (yếu) → 💾 decomposition_Vietnam.png
  United States        trend=0.96 | seasonal=0.24 (yếu) → 💾 decomposition_United_States.png
  China                trend=0.96 | seasonal=0.20 (yếu) → 💾 decomposition_China.png
  United Kingdom       trend=0.94 | seasonal=0.25 (yếu) → 💾 decomposition_United_Kingdom.png
  Brazil               trend=0.97 | seasonal=0.12 (yếu) → 💾 decomposition_Brazil.png
  India                trend=0.98 | seasonal=0.19 (yếu) → 💾 decomposition_India.png
  South Africa         trend=0.99 | seasonal=0.24 (yếu) → 💾 decomposition_South_Africa.png
```
#### Bước E: ACF / PACF Plot
Bước này dùng đồ thị ACF (Autocorrelation Function) và PACF (Partial Autocorrelation Function) để hỗ trợ xác định tham số đầu vào cho mô hình ARIMA:
ACF: Gợi ý bậc $q$ cho thành phần MA (Moving Average).
PACF: Gợi ý bậc $p$ cho thành phần AR (AutoRegressive).
Chương trình sử dụng chuỗi đã qua biến đổi sai phân, dựa trên khoảng tin cậy 95% để tìm ra các độ trễ (lag) có ý nghĩa thống kê và tự động gợi ý giá trị khởi tạo cho $(p, q)$.
```python
print("\n" + "=" * 60)
print("BƯỚC 10 — ACF / PACF (xác định p, q)")
print("=" * 60)

acf_pacf_summary = {}

for country in countries:
    sub    = df_weekly[df_weekly["country"] == country].copy()
    series = sub[DIFF_COL].dropna()

    if len(series) < 30:
        print(f"  ⚠️  {country:<20} — quá ít dữ liệu, bỏ qua")
        acf_pacf_summary[country] = {"suggest_p": 1, "suggest_q": 1}
        continue

    lags = min(40, len(series) // 2 - 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"ACF & PACF — {country}", fontsize=12, fontweight="bold")

    plot_acf( series, ax=axes[0], lags=lags, alpha=0.05, color="#2563EB")
    plot_pacf(series, ax=axes[1], lags=lags, alpha=0.05, color="#DC2626", method="ywm")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Lag (tuần)")

    acf_vals,  acf_confint  = acf( series, nlags=lags, alpha=0.05)
    pacf_vals, pacf_confint = pacf(series, nlags=lags, alpha=0.05, method="ywm")

    sig_acf  = [i for i in range(1, len(acf_vals))
                if acf_vals[i] > acf_confint[i, 1] - acf_vals[i]
                or acf_vals[i] < acf_confint[i, 0] - acf_vals[i]]
    sig_pacf = [i for i in range(1, len(pacf_vals))
                if pacf_vals[i] > pacf_confint[i, 1] - pacf_vals[i]
                or pacf_vals[i] < pacf_confint[i, 0] - pacf_vals[i]]

    suggest_q = sig_acf[0]  if sig_acf  else 1
    suggest_p = sig_pacf[0] if sig_pacf else 1

    acf_pacf_summary[country] = {"suggest_p": suggest_p, "suggest_q": suggest_q}

    axes[0].set_title(f"ACF  → gợi ý q = {suggest_q}", fontsize=10)
    axes[1].set_title(f"PACF → gợi ý p = {suggest_p}", fontsize=10)

    plt.tight_layout()
    fname = f"acf_pacf_{country.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"  {country:<20} gợi ý p={suggest_p}, q={suggest_q}  → 💾 {fname}")
```
Kết quả thu được
```python
============================================================
BƯỚC 10 — ACF / PACF (xác định p, q)
============================================================
  Vietnam              gợi ý p=1, q=1  → 💾 acf_pacf_Vietnam.png
  United States        gợi ý p=1, q=1  → 💾 acf_pacf_United_States.png
  China                gợi ý p=11, q=11 → 💾 acf_pacf_China.png
  United Kingdom       gợi ý p=1, q=1  → 💾 acf_pacf_United_Kingdom.png
  Brazil               gợi ý p=1, q=1  → 💾 acf_pacf_Brazil.png
  India                gợi ý p=1, q=1  → 💾 acf_pacf_India.png
  South Africa         gợi ý p=1, q=1  → 💾 acf_pacf_South_Africa.png
```
#### Bước F: Xác định d và chuỗi INPUT ARIMA
Trong bước này, chương trình xác định bậc sai phân $d$ thích hợp dựa trên kết quả kiểm định tính dừng thu được ở bước trước:
Nếu chuỗi gốc đạt dừng $\rightarrow d = 0$ (dùng chuỗi gốc).
Nếu dừng sau sai phân bậc 1 $\rightarrow d = 1$ (dùng cột diff1).
Nếu chưa dừng sau sai phân bậc 1 $\rightarrow d = 2$ (dùng cột diff2).
Lưu ý: Các đặc trưng thời gian như year, month, quarter, week_of_year được tạo thêm chỉ nhằm mục đích phân tích khám phá (EDA) và trực quan hóa, không được đưa trực tiếp vào mô hình ARIMA đơn biến.
```python
print("\n" + "=" * 60)
print("BƯỚC 11 — Xác định d và chuỗi input ARIMA")
print("=" * 60)

# Thêm datetime features (metadata / EDA, KHÔNG đưa vào ARIMA)
df_weekly["year"]         = df_weekly["week_start"].dt.year
df_weekly["month"]        = df_weekly["week_start"].dt.month
df_weekly["quarter"]      = df_weekly["week_start"].dt.quarter
df_weekly["week_of_year"] = df_weekly["week_start"].dt.isocalendar().week.astype(int)
df_weekly["is_q1"]        = (df_weekly["quarter"] == 1).astype(int)
df_weekly["is_q4"]        = (df_weekly["quarter"] == 4).astype(int)

print("  ⚠️  DateTime features (year/month/quarter/...) chỉ dùng EDA, KHÔNG đưa vào ARIMA")

arima_input_col = {}
print(f"\n  {'Quốc gia':<20} {'d':>3}  Chuỗi input ARIMA")
print("  " + "-" * 55)

for country in countries:
    non_stat_orig = any(
        r["conclusion"] == "❌ NON-STATIONARY"
        for r in stat_results if r["country"] == country and r["series"] == "original"
    )
    stat_d1 = any(
        r["conclusion"] == "✅ STATIONARY"
        for r in stat_results if r["country"] == country and r["series"] == "diff(1)"
    )

    if not non_stat_orig:
        d, col = 0, TARGET_COL
    elif stat_d1:
        d, col = 1, DIFF_COL
    else:
        d, col = 2, DIFF2_COL if DIFF2_COL in df_weekly.columns else DIFF_COL

    arima_input_col[country] = {"d": d, "col": col}
    print(f"  {country:<20} {d:>3}  '{col}'")
```
Kết quả thu được
```python
============================================================
BƯỚC 11 — Xác định d và chuỗi input ARIMA
============================================================
  ⚠️  DateTime features (year/month/quarter/...) chỉ dùng EDA, KHÔNG đưa vào ARIMA

  Quốc gia               d  Chuỗi input ARIMA
  -------------------------------------------------------
  Vietnam                2  'people_vaccinated_per_hundred_diff2'
  United States          2  'people_vaccinated_per_hundred_diff2'
  China                  2  'people_vaccinated_per_hundred_diff2'
  United Kingdom         2  'people_vaccinated_per_hundred_diff2'
  Brazil                 2  'people_vaccinated_per_hundred_diff2'
  India                  2  'people_vaccinated_per_hundred_diff2'
  South Africa           2  'people_vaccinated_per_hundred_diff2'
```
#### Bước G: Train/Test Split (theo thời gian)
Bước này được dùng để chia dữ liệu thành hai tập Train và Test để đánh giá mô hình.

Thay vì phân chia ngẫu nhiên, hệ thống áp dụng phương pháp Time-based Split (phân chia theo thứ tự thời gian) nhằm giữ nguyên thứ tự chuỗi và tuyệt đối tránh hiện tượng Data Leakage. Đối với từng quốc gia, TEST_WEEKS (52 tuần cuối cùng) được giữ lại làm tập kiểm tra, phần dữ liệu còn lại được dùng để huấn luyện.
```python
print("\n" + "=" * 60)
print(f"BƯỚC 12 — Train/Test Split (test = {TEST_WEEKS} tuần cuối)")
print("=" * 60)
print("  ⚠️  Time-based split — KHÔNG random (tránh data leakage)\n")

split_summary = {}

for country in countries:
    sub       = df_weekly[df_weekly["country"] == country].copy().reset_index(drop=True)
    input_col = arima_input_col[country]["col"]
    series    = sub[input_col].dropna()

    if len(series) <= TEST_WEEKS:
        print(f"  ⚠️  {country:<20} — quá ít dữ liệu")
        continue

    n_train = len(series) - TEST_WEEKS
    split_summary[country] = {
        "train_start": sub["week_start"].iloc[0].date(),
        "train_end":   sub["week_start"].iloc[n_train - 1].date(),
        "test_start":  sub["week_start"].iloc[n_train].date(),
        "test_end":    sub["week_start"].iloc[-1].date(),
        "n_train":     n_train,
        "n_test":      TEST_WEEKS,
    }
    print(
        f"  {country:<20} "
        f"Train: {n_train:>4} tuần ({sub['week_start'].iloc[0].date()} → {sub['week_start'].iloc[n_train-1].date()})  |  "
        f"Test: {TEST_WEEKS:>4} tuần ({sub['week_start'].iloc[n_train].date()} → {sub['week_start'].iloc[-1].date()})"
    )

df_weekly["split"] = "train"
for country, info in split_summary.items():
    mask_test = (
        (df_weekly["country"] == country) &
        (df_weekly["week_start"] >= pd.Timestamp(info["test_start"]))
    )
    df_weekly.loc[mask_test, "split"] = "test"

print(f"\n  Tổng Train : {(df_weekly['split']=='train').sum():,}")
print(f"  Tổng Test  : {(df_weekly['split']=='test').sum():,}")
```
Kết quả thu được
```python
============================================================
BƯỚC 12 — Train/Test Split (test = 52 tuần cuối)
============================================================
  ⚠️  Time-based split — KHÔNG random (tránh data leakage)

  Vietnam              Train:  269 tuần (2020-01-06 → 2025-02-24)  |  Test:   52 tuần (2025-03-03 → 2026-02-23)
  United States        Train:  269 tuần (2020-01-06 → 2025-02-24)  |  Test:   52 tuần (2025-03-03 → 2026-02-23)
  China                Train:  269 tuần (2020-01-06 → 2025-02-24)  |  Test:   52 tuần (2025-03-03 → 2026-02-23)
  United Kingdom       Train:  269 tuần (2020-01-06 → 2025-02-24)  |  Test:   52 tuần (2025-03-03 → 2026-02-23)
  Brazil               Train:  269 tuần (2020-01-06 → 2025-02-24)  |  Test:   52 tuần (2025-03-03 → 2026-02-23)
  India                Train:  269 tuần (2020-01-06 → 2025-02-24)  |  Test:   52 tuần (2025-03-03 → 2026-02-23)
  South Africa         Train:  269 tuần (2020-01-06 → 2025-02-24)  |  Test:   52 tuần (2025-03-03 → 2026-02-23)

  Tổng Train : 1,883
  Tổng Test  : 364
```
#### Bước H: Grid Search (p,d,q) – Tiêu chí AIC
Bước này được dùng để tìm bộ tham số tối ưu cho mô hình ARIMA bằng phương pháp Grid Search. Cụ thể, chương trình sẽ thử nhiều tổ hợp khác nhau của hai tham số p và q trong các khoảng giá trị đã định trước (P_RANGE và Q_RANGE), trong khi giá trị d đã được xác định ở bước trước dựa trên kiểm tra stationary. Với mỗi quốc gia, hệ thống lấy tập dữ liệu train rồi huấn luyện nhiều mô hình ARIMA khác nhau tương ứng với từng cặp (p, q). Sau khi huấn luyện, mô hình sẽ được đánh giá bằng các chỉ số AIC và BIC, trong đó AIC được sử dụng làm tiêu chí chính để chọn mô hình tốt nhất. AIC càng nhỏ thì mô hình càng phù hợp với dữ liệu mà vẫn tránh được hiện tượng quá phức tạp (overfitting). Chương trình sẽ lưu lại bộ tham số (p, d, q) có AIC thấp nhất và in ra Top 5 mô hình tốt nhất cho từng quốc gia để dễ so sánh. Nếu dữ liệu huấn luyện quá ít, hệ thống sẽ dùng mặc định p=1 và q=1. Mục đích chính của bước này là tự động tìm ra cấu hình ARIMA phù hợp nhất cho từng chuỗi thời gian, giúp cải thiện độ chính xác của mô hình dự báo.
```python
print("\n" + "=" * 65)
print("BƯỚC 13 — Grid Search tham số ARIMA")
print(f"          p ∈ {list(P_RANGE)}, q ∈ {list(Q_RANGE)} | tiêu chí: AIC")
print("=" * 65)

best_params = {}

for country in countries:
    sub     = df_weekly[df_weekly["country"] == country]
    col     = arima_input_col[country]["col"]
    d       = arima_input_col[country]["d"]
    train_s = sub[sub["split"] == "train"][col].dropna().values

    if len(train_s) < WF_MIN_TRAIN_WEEKS:
        print(f"  ⚠️  {country:<20} — ít dữ liệu, dùng p=1, q=1")
        best_params[country] = {"p": 1, "d": d, "q": 1, "aic": None}
        continue

    best_aic     = np.inf
    best_pq      = (1, 1)
    results_grid = []
    for p, q in itertools.product(P_RANGE, Q_RANGE):
        if p == 0 and q == 0:
            continue
        try:
            fit = ARIMA(train_s, order=(p, d, q)).fit()
            results_grid.append((p, q, round(fit.aic, 2), round(fit.bic, 2)))
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_pq  = (p, q)
        except Exception:
            continue

    best_params[country] = {
        "p": best_pq[0], "d": d, "q": best_pq[1],
        "aic": round(best_aic, 2)
    }

    results_grid.sort(key=lambda x: x[2])
    print(f"\n  {country} — Top 5 AIC:")
    print(f"    {'(p,d,q)':<12} {'AIC':>10} {'BIC':>10}")
    for row in results_grid[:5]:
        star = " ← best" if (row[0], row[1]) == best_pq else ""
        print(f"    ({row[0]},{d},{row[1]}){'':<6} {row[2]:>10.2f} {row[3]:>10.2f}{star}")

```
Kết quả thu được
```python
BƯỚC 13 — Grid Search tham số ARIMA
          p ∈ [0, 1, 2, 3], q ∈ [0, 1, 2, 3] | tiêu chí: AIC
Vietnam — Top 5 AIC:
    (p,d,q)             AIC        BIC
    (0,2,1)         -6069.42   -6062.24 ← best
    (1,2,0)         -6069.42   -6062.24
    (0,2,2)         -6067.42   -6056.66
    (1,2,1)         -6067.42   -6056.66
    (2,2,0)         -6067.42   -6056.66

United States — Top 5 AIC:
    (p,d,q)             AIC        BIC
    (1,2,2)          -225.18    -210.83 ← best
    (2,2,2)          -223.69    -205.75
    (0,2,3)          -223.47    -209.12
    (0,2,2)          -212.19    -201.42
    (1,2,3)          -208.18    -190.24

China — Top 5 AIC:
    (p,d,q)             AIC        BIC
    (0,2,1)         -6069.42   -6062.24 ← best
    (1,2,0)         -6069.42   -6062.24
    (0,2,2)         -6067.42   -6056.66
    (1,2,1)         -6067.42   -6056.66
    (2,2,0)         -6067.42   -6056.66
United Kingdom — Top 5 AIC:
    (p,d,q)             AIC        BIC
    (0,2,2)           185.31     196.07 ← best
    (0,2,3)           186.27     200.62
    (1,2,2)           186.34     200.69
    (2,2,2)           188.22     206.16
    (1,2,3)           189.16     207.10
Brazil — Top 5 AIC:
    (p,d,q)             AIC        BIC
    (0,2,2)           -72.94     -62.18 ← best
    (1,2,2)           -71.69     -57.34
    (0,2,3)           -71.66     -57.31
    (1,2,3)           -69.02     -51.08
    (3,2,2)           -67.26     -45.74
India — Top 5 AIC:
    (p,d,q)             AIC        BIC
    (2,2,3)           -42.81     -21.29 ← best
    (3,2,3)           -40.49     -15.38
    (0,2,3)           -39.64     -25.29
    (1,2,2)           -39.20     -24.85
    (1,2,3)           -38.03     -20.09
South Africa — Top 5 AIC:
    (p,d,q)             AIC        BIC
    (0,2,3)           -86.89     -72.54 ← best
    (1,2,3)           -85.27     -67.33
    (3,2,2)           -83.41     -61.89
    (2,2,2)           -75.14     -57.20
    (3,2,3)           -70.46     -45.35

```
#### Bước I: Walk-Forward Validation
Bước này được dùng để đánh giá mô hình ARIMA bằng phương pháp Walk-Forward Validation với cơ chế Expanding Window. Đây là cách kiểm tra rất phổ biến trong bài toán dự báo chuỗi thời gian vì nó mô phỏng đúng tình huống thực tế: tại mỗi thời điểm, mô hình chỉ được phép học từ dữ liệu quá khứ và dự đoán cho tương lai, tuyệt đối không sử dụng dữ liệu phía sau để tránh data leakage. Với mỗi quốc gia, chương trình lấy bộ tham số ARIMA tối ưu (p, d, q) đã tìm được ở bước trước, sau đó bắt đầu từ tập train ban đầu và liên tục mở rộng cửa sổ dữ liệu huấn luyện theo thời gian. Ở mỗi bước t, mô hình sẽ được train trên dữ liệu từ đầu chuỗi đến thời điểm t-1, rồi dự đoán cho tuần t. Quá trình này được lặp lại cho đến hết tập test để thu được toàn bộ giá trị dự đoán và giá trị thực tế. Sau cùng, chương trình tính các chỉ số đánh giá gồm MAE, RMSE và MAPE nhằm đo mức độ sai số của mô hình. Nếu mô hình ARIMA gặp lỗi trong quá trình dự báo, hệ thống sẽ sử dụng phương pháp dự đoán đơn giản (naive forecast) bằng cách lấy giá trị cuối cùng của chuỗi train làm dự báo. Mục đích chính của bước này là đánh giá khả năng dự báo thực tế của mô hình ARIMA một cách khách quan và sát với điều kiện triển khai ngoài thực tế.
```python
print("\n" + "=" * 65)
print("BƯỚC 14 — Walk-Forward Validation (Expanding Window)")
print("=" * 65)
print("""
  Tại mỗi bước t: train trên [0..t-1], predict bước t.
  Mô phỏng dự báo thực tế — không dùng dữ liệu tương lai.
""")

wf_results = {}

for country in countries:
    sub     = df_weekly[df_weekly["country"] == country].copy().reset_index(drop=True)
    col     = arima_input_col[country]["col"]
    p, d, q = best_params[country]["p"], best_params[country]["d"], best_params[country]["q"]
    series  = sub[col].dropna().values
    dates   = sub["week_start"].values

    if len(series) <= TEST_WEEKS + WF_MIN_TRAIN_WEEKS:
        print(f"  ⚠️  {country:<20} — không đủ dữ liệu walk-forward")
        continue

    n_train         = len(series) - TEST_WEEKS
    actuals, preds  = [], []

    for t in range(n_train, len(series)):
        train_window = series[:t]
        try:
            pred = ARIMA(train_window, order=(p, d, q)).fit().forecast(steps=1)[0]
        except Exception:
            pred = train_window[-1]   # fallback: naive
        actuals.append(series[t])
        preds.append(pred)

    mae_val  = mean_absolute_error(actuals, preds)
    rmse_val = np.sqrt(mean_squared_error(actuals, preds))
    mape_val = mape(actuals, preds)

    wf_results[country] = {
        "mae":     round(mae_val, 4),
        "rmse":    round(rmse_val, 4),
        "mape":    round(mape_val, 2),
        "actuals": actuals,
        "preds":   preds,
        "dates":   dates[n_train:]
    }
    print(f"  {country:<20} MAE={mae_val:.4f} | RMSE={rmse_val:.4f} | MAPE={mape_val:.2f}%")

```
