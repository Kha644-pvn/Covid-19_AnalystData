## 1. Giới thiệu về tập dữ liệu

Bộ dữ liệu được sử dụng được thu thập từ **Our World in Data (OWID)**, tổ chức phi lợi nhuận đã liên tục thu thập và chuẩn hóa dữ liệu COVID-19 từ nhiều lĩnh vực kể từ khi đại dịch bắt đầu.

* **Nguồn dữ liệu:** [Our World in Data CSV](https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv)

### Tổng quan thông số

| Chỉ số | Giá trị |
| :--- | :--- |
| **Tổng số dòng** | 570,606 |
| **Tổng số cột** | 61 |
| **Khoảng thời gian** | 01/2020 – 02/2026 |
| **Quốc gia / vùng lãnh thổ** | ~200 |
| **Quốc gia nghiên cứu trọng tâm** | 7: Mỹ, Trung Quốc, Ấn Độ, Brazil, Anh, Việt Nam, Nam Phi |

---

### Phân nhóm các cột dữ liệu chính
Tập dữ liệu gồm **25 cột chính** được chia thành 4 nhóm cụ thể:

* **Nhóm 1: Thông tin định danh & Thời gian**
  * `country`: Tên quốc gia
  * `continent`: Châu lục
  * `date`: Ngày ghi nhận
* **Nhóm 2: Chỉ số dịch tễ (Epidemiology)**
  * `total_cases` / `new_cases`: Tổng / mới ca nhiễm
  * `total_deaths` / `new_deaths`: Tổng / mới ca tử vong
  * `reproduction_rate`: Hệ số lây nhiễm R
  * `total_cases_per_million`: Ca nhiễm trên 1 triệu dân
* **Nhóm 3: Xét nghiệm & Vaccine**
  * `total_tests` / `new_tests` / `positive_rate`: Dữ liệu xét nghiệm
  * `total_vaccinations` / `people_fully_vaccinated` / `people_vaccinated_per_hundred`: Tiến độ tiêm chủng
* **Nhóm 4: Nhân khẩu học & Kinh tế**
  * `population` / `population_density` / `gdp_per_capita` / `stringency_index`

---

### 1.3. Thông tin chi tiết các cột dữ liệu

| Tên cột | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| `country` | object | Quốc gia |
| `date` | object | Ngày ghi nhận |
| `continent` | object | Châu lục |
| `total_cases` | float64 | Tổng số ca nhiễm |
| `new_cases` | float64 | Số ca nhiễm mới |
| `new_cases_smoothed` | float64 | Ca nhiễm TB 7 ngày |
| `total_deaths` | float64 | Tổng số ca tử vong |
| `new_deaths` | float64 | Số ca tử vong mới |
| `reproduction_rate` | float64 | Hệ số lây nhiễm R |
| `positive_rate` | float64 | Tỷ lệ xét nghiệm dương tính |
| `total_vaccinations` | float64 | Tổng số mũi tiêm |
| `people_fully_vaccinated`| float64 | Số người tiêm đủ mũi |
| `stringency_index` | float64 | Mức độ nghiêm ngặt chính sách |
| `gdp_per_capita` | float64 | GDP bình quân đầu người |
| `population` | float64 | Dân số |
| `population_density` | float64 | Mật độ dân số |
