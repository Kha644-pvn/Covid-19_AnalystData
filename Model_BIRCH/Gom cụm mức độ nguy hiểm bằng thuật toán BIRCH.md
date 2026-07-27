## 2.3.3.1. Giới thiệu và lý thuyết thuật toán

**BIRCH** (*Balanced Iterative Reducing and Clustering using Hierarchies*) sử dụng **CF Tree** (*Clustering Feature Tree*) để nén thông tin, trong đó mỗi nút lưu ba thành phần `(N, LS, SS)`. Thuật toán này rất phù hợp với dữ liệu lớn nhờ tốc độ xử lý nhanh và khả năng mở rộng tốt.

---

## 2.3.3.2. Chuẩn bị dữ liệu

```python
import pandas as pd
import numpy as np
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ====================== 1. ĐỌC DỮ LIỆU ======================
df = pd.read_csv('covid_data_stage2.csv')
df['date'] = pd.to_datetime(df['date'])

# ==================== PHẦN BẠN CHỈ CẦN SỬA Ở ĐÂY ====================
COUNTRIES = [
    'Vietnam', 'United States', 'Brazil', 'United Kingdom', 'India',
    'South Africa', 'China', 'France', 'Germany', 'Italy', 'Spain'
]

# Lọc dữ liệu
df_cluster = df[df["country"].isin(COUNTRIES)].copy()

print(f"Đang phân tích {len(COUNTRIES)} quốc gia | Shape dữ liệu: {df_cluster.shape}")
```

---

## 2.3.3.3. Xây dựng mô hình

```python
# ====================== 2. FEATURE ENGINEERING ======================
def compute_risk_features(group):
    group = group.sort_values('date').reset_index(drop=True)

    # ==================== MỨC ĐỘ LÂY LAN ====================
    mean_cases = group['new_cases_custom_smoothed'].mean()
    max_cases = group['new_cases_custom_smoothed'].max()
    total_cases_per_mil = group['total_cases_per_million_custom_smoothed'].max()
    peak_ratio = max_cases / (mean_cases + 1e-8)

    # ==================== MỨC ĐỘ NGHIÊM TRỌNG ====================
    mean_deaths = group['new_deaths_custom_smoothed'].mean()
    max_deaths = group['new_deaths_custom_smoothed'].max()
    mean_cfr = (
        group['new_deaths_custom_smoothed'].sum() /
        (group['new_cases_custom_smoothed'].sum() + 1e-8)
    )

    # ==================== KHẢ NĂNG KIỂM SOÁT ====================
    mean_stringency = group['stringency_index'].mean()
    mean_R = group['reproduction_rate'].mean()

    # ==================== XU HƯỚNG PHÁT TRIỂN ====================
    max_R = group['reproduction_rate'].max()
    days_R_gt1_ratio = (group['reproduction_rate'] > 1).mean()

    # Xu hướng slope
    if len(group) > 14:
        slope = np.polyfit(range(len(group)), group['new_cases_custom_smoothed'], 1)[0]
    else:
        slope = 0

    # Tỷ lệ thay đổi sau vaccine
    post_vaccine_mean = (
        group[group.get('post_vaccine_phase', 0) == 1]['new_cases_custom_smoothed'].mean()
        if 'post_vaccine_phase' in group.columns else 0
    )

    return pd.Series({
        'mean_cases_norm': mean_cases,
        'max_cases_norm': max_cases,
        'total_cases_per_mil': total_cases_per_mil,
        'peak_ratio': peak_ratio,
        'mean_deaths_norm': mean_deaths,
        'max_deaths_norm': max_deaths,
        'mean_cfr': mean_cfr,
        'mean_stringency': mean_stringency,
        'mean_R': mean_R,
        'max_R': max_R,
        'days_R_gt1_ratio': days_R_gt1_ratio,
        'cases_trend_slope': slope,
        'post_vaccine_mean_cases': post_vaccine_mean
    })

# Áp dụng cho tất cả quốc gia
feature_df = df_cluster.groupby('country').apply(compute_risk_features).reset_index()

# ====================== 3. BIRCH CLUSTERING ======================
feature_cols = [col for col in feature_df.columns if col != 'country']
X = feature_df[feature_cols].fillna(0).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Giảm chiều (rất quan trọng khi nhiều quốc gia)
pca = PCA(n_components=min(8, len(feature_cols)))
X_pca = pca.fit_transform(X_scaled)

# ====================== 4. ÁP DỤNG BIRCH ======================
birch = Birch(threshold=0.7, branching_factor=30, n_clusters=4)
feature_df['cluster'] = birch.fit_predict(X_pca) + 1
```

**Hình 2.19: Code xây dựng mô hình BIRCH**

---

## 2.3.3.4. Phân tích quy trình

Quy trình xây dựng mô hình BIRCH trong đoạn code trên là một pipeline **feature-based clustering** được thiết kế nhằm gom cụm các quốc gia theo mức độ nguy hiểm của dịch COVID-19. Thay vì sử dụng trực tiếp chuỗi thời gian dài như DTW, quy trình này chuyển đổi dữ liệu chuỗi thời gian thành một bảng đặc trưng (*tabular data*) ngắn gọn, sau đó áp dụng thuật toán BIRCH để phân cụm. Toàn bộ quy trình được chia thành 4 bước chính như sau:

### Bước 1: Feature Engineering

Hàm `compute_risk_features` được áp dụng cho từng nhóm quốc gia thông qua `groupby('country')`. Hàm này nhận vào toàn bộ dữ liệu thời gian của một quốc gia, sau đó tính toán 13 đặc trưng quan trọng đại diện cho 4 khía cạnh nguy hiểm:

* **Mức độ lây lan:**
  trung bình ca nhiễm (`mean_cases_norm`), đỉnh ca nhiễm (`max_cases_norm`), tổng ca trên triệu dân (`total_cases_per_mil`), và độ nhọn của dịch (`peak_ratio`).

* **Mức độ nghiêm trọng:**
  trung bình tử vong (`mean_deaths_norm`), tử vong cao nhất (`max_deaths_norm`), và tỷ lệ tử vong trung bình (`mean_cfr`).

* **Khả năng kiểm soát:**
  mức độ nghiêm ngặt của chính sách (`mean_stringency`).

* **Xu hướng phát triển:**
  hệ số lây nhiễm trung bình (`mean_R`), cao nhất (`max_R`), tỷ lệ ngày `R > 1` (`days_R_gt1_ratio`), độ dốc xu hướng ca nhiễm (`cases_trend_slope`), và mức ca nhiễm trung bình sau vaccine (`post_vaccine_mean_cases`).

Kết quả sau bước này là bảng `feature_df`, trong đó mỗi quốc gia chỉ còn một dòng với 13 cột đặc trưng số. Nhờ đó, dữ liệu chuỗi thời gian dài đã được biến đổi thành dữ liệu bảng phù hợp cho clustering.

### Bước 2: Chuẩn bị dữ liệu

Chương trình tách ra danh sách các cột đặc trưng (`feature_cols`), chuyển thành ma trận NumPy `X`, và xử lý giá trị thiếu bằng `fillna(0)`. Sau đó, dữ liệu được chuẩn hóa bằng `StandardScaler` để đưa tất cả đặc trưng về cùng một thang đo (`mean = 0`, `std = 1`), tránh hiện tượng một số đặc trưng có giá trị lớn chi phối kết quả clustering.

Vì số lượng đặc trưng khá nhiều (13 cột), bước giảm chiều bằng PCA (`n_components = min(8, len(feature_cols))`) được thực hiện để giữ lại các thành phần chính quan trọng nhất, giảm nhiễu và tăng tốc độ tính toán cho BIRCH.

### Bước 3: Huấn luyện BIRCH

Mô hình BIRCH được khởi tạo với các tham số:

* `threshold = 0.7`
* `branching_factor = 30`
* `n_clusters = 4`

Sau đó, mô hình được huấn luyện trên ma trận đã giảm chiều (`X_pca`) thông qua phương thức `fit_predict`. Kết quả là mỗi quốc gia được gán một nhãn cluster (từ 1 đến 4) và được thêm vào cột `cluster` của `feature_df`.

---

## 2.3.3.5. Kết quả

### Kết quả phân cụm BIRCH

```text
           country  cluster
0           Brazil        1
1            China        1
5            Italy        1
4            India        1
6     South Africa        2
8   United Kingdom        2
10         Vietnam        2
9    United States        3
7            Spain        4
3          Germany        4
2           France        4
```

Mô hình BIRCH đã gom 11 quốc gia thành 4 cụm dựa trên mức độ nguy hiểm tổng hợp (*lây lan, nghiêm trọng, khả năng kiểm soát, xu hướng*).

| Cluster   | Quốc gia trong cụm                    | Ý nghĩa                                       |
| --------- | ------------------------------------- | --------------------------------------------- |
| Cluster 1 | Brazil, China, Italy, India           | Nguy hiểm cao – Lây lan mạnh & tích lũy lớn   |
| Cluster 2 | South Africa, United Kingdom, Vietnam | Nguy hiểm trung bình thấp – Kiểm soát tốt hơn |
| Cluster 3 | United States                         | Nguy hiểm rất cao – Siêu dịch                 |
| Cluster 4 | Spain, Germany, France                | Nguy hiểm cao – Peak mạnh & châu Âu điển hình |

### Diễn giải từng cụm

* **Cluster 1 (Brazil, China, Italy, India):**
  Đây là nhóm có tổng ca nhiễm tích lũy rất cao (`total_cases_per_mil ≈ 183.573`) và mức ca nhiễm trung bình cao. Tuy nhiên, tỷ lệ tử vong (`mean_cfr`) tương đối thấp. Nhóm này đại diện cho các quốc gia có dịch bùng phát mạnh về quy mô nhưng khả năng kiểm soát (`stringency`) và xu hướng `R` không quá cực đoan.

* **Cluster 2 (South Africa, UK, Vietnam):**
  Đây là cụm an toàn nhất trong 4 cụm. Các chỉ số lây lan (`mean_cases_norm`, `max_cases_norm`) và tử vong đều thấp nhất. Tỷ lệ ngày `R > 1` cũng thấp. Việt Nam nằm trong cụm này cho thấy mô hình đánh giá Việt Nam thuộc nhóm kiểm soát tốt, lây lan hạn chế so với các nước khác.

* **Cluster 3 (United States):**
  Cụm đơn lẻ nhưng cực kỳ nổi bật. Mỹ có mức ca nhiễm trung bình cực cao (`mean_cases_norm ≈ 33.101`), tổng ca nhiễm khổng lồ, và `post_vaccine_mean_cases` vẫn rất cao. Đây là cụm đại diện cho siêu dịch, quy mô lây lan vượt trội hoàn toàn so với các cụm còn lại.

* **Cluster 4 (Spain, Germany, France):**
  Nhóm các nước châu Âu có `peak_ratio` cao nhất (`18.359`), nghĩa là dịch có những đợt bùng phát rất mạnh và đột ngột. Tổng ca nhiễm trên triệu dân cao thứ hai (`452.362`). Tử vong và `R` cũng ở mức cao. Đây là cụm điển hình cho các nước phát triển châu Âu có hệ thống y tế tốt nhưng vẫn chịu áp lực dịch mạnh.

---

## Heatmap “Trung bình các đặc trưng theo Cluster”

* **Màu sắc:** Càng đỏ → giá trị càng cao (nguy hiểm cao). Càng xanh dương → giá trị càng thấp.

### Quan sát nổi bật

* Cột `total_cases_per_mil` và `post_vaccine_mean_cases` có sự phân biệt rất rõ giữa các cụm.
* **Cluster 3 (Mỹ)** nổi bật đỏ rực ở `mean_cases_norm (33.101)` và `post_vaccine_mean_cases (32.365)` → cho thấy Mỹ có quy mô ca nhiễm cực lớn ngay cả sau vaccine.
* **Cluster 4** có màu đỏ đậm ở `peak_ratio (18.359)` → dịch có đỉnh rất cao, bùng nổ mạnh.
* **Cluster 2** hầu hết là màu xanh nhạt/xanh dương → các chỉ số nguy hiểm đều thấp nhất.
* `mean_cfr` (tỷ lệ tử vong) khá thấp ở tất cả cụm, cho thấy vaccine và y tế đã làm giảm đáng kể tỷ lệ chết.

---

## 2.3.3.6. Nhận xét

Mô hình BIRCH hoạt động khá tốt trong việc phân nhóm các quốc gia theo mức độ nguy hiểm tổng hợp. Điểm mạnh lớn nhất là:

* Chuyển hóa thành công từ dữ liệu chuỗi thời gian dài → bảng đặc trưng ngắn gọn, dễ diễn giải.
* Phân biệt rõ ràng giữa các nhóm: Mỹ đứng một mình (*siêu dịch*), châu Âu có peak mạnh, nhóm Việt Nam–UK–Nam Phi kiểm soát tốt hơn, và nhóm Brazil–Ấn Độ–Trung Quốc–Italy có quy mô lớn nhưng CFR thấp.
* Khả năng mở rộng tốt khi thêm nhiều quốc gia.

### Hạn chế

* `mean_cases_norm` đang chi phối quá mạnh (như thấy trong bar chart), làm một số đặc trưng khác như CFR, `R`, `stringency` bị lu mờ.
* Cluster 3 chỉ có 1 quốc gia → có thể xem là outlier.
* Chưa có trọng số cho 4 nhóm nguy hiểm (*lây lan, tử vong, kiểm soát, xu hướng*).

