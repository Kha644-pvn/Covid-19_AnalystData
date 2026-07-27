## Dynamic Time Warping (DTW)

### DTW là gì?

**Dynamic Time Warping (DTW)** là thuật toán đo khoảng cách giữa hai chuỗi thời gian bằng cách cho phép “kéo giãn” thời gian (*warping*) để căn chỉnh (*align*) các pattern tương tự nhau.
DTW không yêu cầu hai chuỗi phải có cùng độ dài hay xảy ra đồng thời.

### Tại sao dùng DTW thay vì Euclidean distance?

**Minh họa trực quan:**
Quốc gia A có đỉnh dịch ở tháng 3, trong khi Quốc gia B có đỉnh dịch ở tháng 5 nhưng có pattern tương tự.

* **Euclidean distance** sẽ đánh giá hai chuỗi này là “xa” vì so sánh điểm-điểm theo thời gian.
* **DTW distance** sẽ đánh giá là “gần” vì tìm được một alignment tối ưu cho hai pattern tương tự nhau.

### Ưu điểm của DTW

* **Time shift invariance:** Không bị ảnh hưởng bởi độ trễ về thời gian.
* **Non-linear alignment:** Align được các pattern với tốc độ diễn tiến khác nhau.
* **Shape similarity:** Tập trung vào hình dạng của curve, không phụ thuộc magnitude.
* **Robust to noise:** Ít nhạy cảm với outliers hơn Euclidean.

### Cách tính DTW distance

DTW tạo một ma trận khoảng cách giữa tất cả các cặp điểm của hai chuỗi, sau đó tìm đường đi (*warping path*) có tổng khoảng cách nhỏ nhất từ `(0,0)` đến `(m,n)`.

**DTW distance** = tổng khoảng cách trên đường đi tối ưu đó.

```python
from dtaidistance import dtw
from dtaidistance import dtw_ndim
import numpy as np

# Lấy danh sách quốc gia và chuỗi đã chuẩn hóa
countries = normalized_df.columns.tolist()
series_list = [normalized_df[c].dropna().values for c in countries]

# Tính ma trận khoảng cách DTW (n_countries × n_countries)
n = len(countries)
dtw_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        dist = dtw.distance(series_list[i], series_list[j],
                           window=30)  # Sakoe-Chiba window constraint
        dtw_matrix[i, j] = dist
        dtw_matrix[j, i] = dist

print(f'DTW distance matrix shape: {dtw_matrix.shape}')
print(f'Ví dụ: DTW(Vietnam, Thailand) = {dtw_matrix[countries.index("Vietnam"), countries.index("Thailand")]:.4f}')
```

### Constraints trong DTW

* **Window constraint (Sakoe-Chiba band):** Giới hạn warping để tránh align quá xa trong thời gian, đảm bảo alignment có tính thực tế.
* **Slope constraint:** Đảm bảo tính đơn điệu (*monotonicity*) của warping path.

### Output

Kết quả cuối cùng là **ma trận đối xứng DTW distance** có kích thước **`n_countries × n_countries`**.

