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
Một chuỗi thời gian được gọi là dừng (stationary) khi các tính chất thống kê của nó — bao gồm giá trị trung bình, phương sai và cấu trúc tự tương quan — không thay đổi theo thời gian. Đây là điều kiện tiên quyết để áp dụng mô hình ARIMA, bởi các tham số của mô hình chỉ có ý nghĩa ổn định khi chuỗi dữ liệu thỏa mãn tính dừng. Trong thực tế, nhiều chuỗi thời gian như dữ liệu tiêm chủng tích lũy thường có xu hướng tăng dần theo thời gian, do đó cần được biến đổi trước khi đưa vào mô hình.
Để kiểm định tính dừng, nghiên cứu sử dụng đồng thời hai kiểm định thống kê bổ sung cho nhau:
 * •	Kiểm định ADF (Augmented Dickey-Fuller): Giả thuyết H₀ cho rằng chuỗi có nghiệm đơn vị (non-stationary). Nếu p-value < 0.05, bác bỏ H₀ và kết luận chuỗi là dừng.
 * •	Kiểm định KPSS (Kwiatkowski-Phillips-Schmidt-Shin): Giả thuyết H₀ cho rằng chuỗi là dừng. Nếu p-value > 0.05, không bác bỏ H₀ và kết luận chuỗi là dừng.
# 5. Kết quả
