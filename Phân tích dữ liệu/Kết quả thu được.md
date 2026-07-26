## Phân tích mối tương quan giữa số ca nhiễm mới trong ngày và số ca tử vong mới trong ngày để quan sát sự ảnh hưởng của 2 cột new_cases và new_deaths.

![Biểu đồ DeathRace vs Country](BieuDo/0.jpg)

1)	Brazil: Death Rate bắt đầu rất cao (~0.08) vào đầu năm 2020. Dao động mạnh trong suốt năm 2020–2021 với nhiều đỉnh nhỏ liên tục. Từ giữa năm 2022 trở đi, tỷ lệ tử vong giảm rõ rệt và duy trì ở mức thấp. Brazil có dịch kéo dài với nhiều đợt sóng nhỏ, cho thấy khả năng kiểm soát tử vong cải thiện chậm. Tổng thể là quốc gia có Death Rate cao và biến động mạnh trong giai đoạn đầu dịch.
2)	China: có 2 đỉnh dịch cực lớn: đỉnh đầu ~0.6 vào đầu những năm 2020 và đỉnh thứ hai ~0.5 cuối năm 2022 - đầu năm 2023. Giữa hai đỉnh là giai đoạn Death_Rate rất thấp. Hai đợt bùng phát tử vong cách nhau rất xa phản ánh chính sách covid-19 nghiêm ngặt sau đó đột nhiên nới lỏng. Sau năm 2023, Death_Rate giảm mạnh và ổn định ở mức thấp.
3)	South_Africa: Death Rate biến động rất mạnh với nhiều đỉnh cao (tối đa ~0.12). Xuất hiện nhiều đợt sóng liên tiếp trong giai đoạn 2020–2022. Sau năm 2022, tỷ lệ tử vong giảm rõ rệt nhưng vẫn còn dao động nhẹ. Đây là quốc gia có Death Rate cao và không ổn định nhất trong nhóm.
4)	India: có nhiều đỉnh sắc nhọn, cao nhất đạt ~0.39 vào giữa năm 2021 (đợt Delta). Sau đợt Delta, Death Rate giảm mạnh nhưng vẫn xuất hiện thêm vài đỉnh nhỏ năm 2022–2023. Từ cuối 2023 trở đi, Death Rate gần như bằng 0. India có đặc trưng là các đợt bùng phát tử vong rất mạnh nhưng ngắn.
5)	United Kingdom:  đỉnh cao nhất (~0.32) xuất hiện ngay đầu năm 2020. Sau đó giảm rất nhanh và duy trì ở mức thấp (0.01–0.05) với các đợt sóng nhỏ. Từ năm 2022 trở đi, Death Rate khá ổn định ở mức thấp. Anh kiểm soát tốt tử vong sau đợt dịch đầu tiên.
6)	United States: Đỉnh cao đầu tiên ~0.085 vào năm 2020. Có nhiều đợt sóng rõ rệt trong năm 2020–2022, sau đó giảm dần. Từ năm 2023 trở đi, Death Rate giảm mạnh và duy trì ở mức rất thấp. Mỹ có quy mô tử vong cao ở giai đoạn đầu nhưng cải thiện rõ rệt sau vaccine.
7)	Việt Nam: Có một đỉnh cực sắc và cao (~0.35) vào cuối năm 2020 / đầu năm 2021. Sau đỉnh này, Death Rate giảm rất nhanh và gần như bằng 0 từ giữa năm 2022 đến nay.Việt Nam có đặc trưng là một đợt bùng phát tử vong mạnh nhưng ngắn, sau đó kiểm soát rất tốt.

## Phân tích mối tương quan giữa tỉ lệ tiêm vắc xin và tỉ lệ tỉ vong của người dân để tìm hiểu quy luật: Liệu tỉ lệ tiêm vắc xin cao có giúp giảm thiểu tỉ lệ tử vong hay không?
![](BieuDo/1.jpg)

⟶ Đồ thị biểu diễn mức độ tử vong và mức độ tiêm vaccinate ở Trung Quốc theo thời gian. Nhìn vào đồ thị ta thấy được ở những năm 2020-2022, khi mà tỉ lệ tiêm vắc-xin còn khá thấp thì tỉ lệ tử vong khi đó rất cao. Từ giữa năm 2021 trở về sau, khi mà tỉ lệ tiêm vắc-xin bắt đầu tăng cao thì khi đó tỉ lệ tử vong có xu hướng giảm đi.

![](BieuDo/2.jpg)

⟶ Biểu đồ thể hiện mức độ tử vong và mức độ tiêm vắc-xin ở Việt Nam. Nhìn vào biểu đồ ra có thể thấy tỉ lệ tử vong ở Việt Nam là rất thấy. Mức cao nhất ghi nhận được chưa tới 0.05. Điều này cho thấy được sự thành công của Việt Nam trong việc kiểm soát dịch bệnh. Bên cạnh đó tỉ lệ tiêm vắc-xin ở việc nam cũng tăng cao từ giữa năm 2021.

![](BieuDo/3.jpg)

⟶ Biểu đồ thể hiện tỉ lệ tử vong và tỉ lệ tiêm vắc xin ở Hoa Kỳ trong từng giai đoạn. Nhìn vào biểu đồ ta thấy trong những giai đoạn đầu chống dịch thì tỉ lệ tử vong cao hơn so với tỉ lệ tiêm phòng. Nhưng từ giai đoạn 2021 trở đi tỉ lệ tủ vong bắt đầu có xu hướng giảm do tỉ lệ tiêm vắc-xin bắt đầu tăng cao.

## Phân tích tỉ lệ được tiêm đủ mũi vắc xin so với những người được tiêm vắc xin

![](BieuDo/4.jpg)

⟶ Biểu đồ biểu diễn tỉ lệ tiêm phòng đủ vắc-xin ở Trung Quốc qua từng năm. Nhìn vào biểu đồ ta có thể thấy trong những năm đầu của giai đoạn mới bắt đầu bùng phát dịch, số người được tiêm đủ vắc xin ở giai đoạn đầu vẫn rất cao. Nhưng sao đó dịch diễn biến phức tạp và lan rộng khiến vắc-xin thiếu hụt và phải chia đều cho toàn dân nên tỉ lệ những người tiêm đủ vắc-xin thấp dần. Đến cuối  những năm 2021 khi mà vắc-xin bắt đầu được nghiên cứu và thử nghiệm thành công cao hơn cũng như sản xuất mạnh hơn thì số người được tiêm đủ vắc-xin bắt đầu tăng mạnh và đỉnh điểm ở năm 2022 khi cứ 100 người được tiêm vắc-xin thì có tới hơn 90 đã tiêm đủ số mũi.

![](BieuDo/5.jpg)

⟶ Biểu đồ cho thấy chiến dịch tiêm vaccine COVID-19 của Việt Nam diễn ra rất nhanh và hiệu quả. Từ cuối năm 2020 đến giữa năm 2021, tỷ lệ tiêm đầy đủ giảm mạnh xuống gần 0% (giai đoạn chờ vaccine và chuẩn bị). Từ cuối năm 2021 đến đầu năm 2022, tỷ lệ tăng bùng nổ, từ gần 0% vọt lên gần 95% chỉ trong vòng khoảng 4–5 tháng. Có một nhịp giảm nhẹ ngắn vào đầu năm 2022 (có thể do điều chỉnh dữ liệu hoặc trì hoãn một số nhóm), sau đó tiếp tục tăng và ổn định. Từ giữa năm 2022 đến năm 2026, tỷ lệ tiêm đầy đủ duy trì rất cao và ổn định ở mức ~95–98%.

![](BieuDo/6.jpg)

⟶ Biểu đồ biểu diễn sô người tiêm đủ vắc-xin ở Brazil. Trong những năm đầu, số người được tiêm đủ mũi vắc-xin chiếm tỉ lệ khá thấp bởi sự khan hiếm vắc-xin. Trong giai đoạn từ năm 2021-2022 tỉ lệ này biến đổi lên xuống đầy biến động và tăng mạnh từ cuối năm 2022.


