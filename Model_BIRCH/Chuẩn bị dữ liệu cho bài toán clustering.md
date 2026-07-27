## Chuyển đổi cấu trúc dữ liệu

Bài toán gom cụm cần chuyển đổi từ **long format** (mỗi dòng là một bản ghi `country-date-value`) sang **wide format**, trong đó mỗi quốc gia được biểu diễn như một chuỗi thời gian duy nhất.

* **Input:** Long format – `(country, date, value)`
* **Output:** Wide format – mỗi quốc gia là một vector chuỗi thời gian

## Chọn feature để clustering

Feature chính được chọn là **`new_cases_smoothed_7d`** vì:

1. Đã được làm mượt → loại bỏ noise ngày.
2. Phản ánh xu hướng chính của dịch.
3. Có tính comparability tốt giữa các quốc gia.

Các feature thay thế có thể xem xét:

* `growth_rate` (pattern tăng trưởng)
* `death_rate` (mức nghiêm trọng)
* **Multi-variate** kết hợp nhiều features

## Xử lý độ dài chuỗi khác nhau

Các quốc gia bắt đầu dịch ở thời điểm khác nhau dẫn đến chuỗi có độ dài không đồng đều. Có 3 giải pháp:

* **Giải pháp 1:** Align theo ngày *"first 100 cases"* – đồng bộ theo thời điểm khởi phát
* **Giải pháp 2:** Truncate/Pad về cùng độ dài – đơn giản nhưng mất thông tin
* **Giải pháp 3:** Dùng **DTW (Dynamic Time Warping)** – không yêu cầu cùng độ dài ✓ *(được chọn)*

## Chuẩn hóa per-country (Min-Max Normalization)

Mỗi chuỗi quốc gia được chuẩn hóa riêng bằng Min-Max:

```python
normalized = (value - min) / (max - min)
```

Mục đích: so sánh **shape/pattern**, không so sánh **magnitude** (absolute numbers). Nhờ đó, một quốc gia nhỏ và một quốc gia lớn có pattern tương tự sẽ được group lại dù số ca tuyệt đối rất khác nhau.

```python
import pandas as pd
import numpy as np

# Pivot sang wide format
pivot_df = df.pivot_table(index='date', columns='country',
                         values='new_cases_smoothed_7d')

# Chuẩn hóa per-country (Min-Max)
def normalize_series(s):
    s_min, s_max = s.min(), s.max()
    if s_max == s_min:
        return s * 0
    return (s - s_min) / (s_max - s_min)

normalized_df = pivot_df.apply(normalize_series, axis=0)

# Điền NaN bằng 0 (giai đoạn chưa có dịch)
normalized_df = normalized_df.fillna(0)
print(f'Shape sau chuẩn hóa: {normalized_df.shape}')
```


