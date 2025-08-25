# Đánh Giá Mô Hình Phát Hiện Tin Giả

Thư mục này chứa các script để đánh giá hiệu suất của mô hình phát hiện tin giả sử dụng các metrics chuẩn.

## 📊 Các Metrics Đánh Giá

Việc đánh giá sử dụng 4 metrics chính:

1. **Precision**: Đo độ chính xác của các dự đoán dương tính
2. **F1-Score**: Trung bình điều hòa của precision và recall
3. **RMSE**: Root Mean Square Error (càng thấp càng tốt)
4. **Accuracy**: Tỷ lệ dự đoán đúng tổng thể

## 🚀 Các Script Có Sẵn

### 1. `clean_evaluate.py` (Khuyến nghị)
- **Code sạch, có tổ chức** với xử lý lỗi tốt
- **Visualization chuyên nghiệp** với matplotlib
- **Hệ thống đánh giá cuối cùng** với xếp loại (A+, A, B, C, D)
- **Mock predictions** để test nhanh

### 2. `real_model_evaluate.py`
- **Tích hợp AI model thật** (khi sẵn sàng)
- Sử dụng model Vietnamese LLaMA thật
- Phân tích RAG-based với knowledge base

## 🎯 Cách Sử Dụng

### Test Nhanh (Mock Model)
```bash
python clean_evaluate.py
```

### Với AI Model Thật
```bash
# Chỉnh sửa real_model_evaluate.py và set use_real_model=True
python real_model_evaluate.py
```

## 📈 Kết Quả Đầu Ra

Việc đánh giá cung cấp:

1. **Kết quả Console**: Metrics chi tiết và confusion matrix
2. **Đánh giá cuối cùng**: Xếp loại và khuyến nghị
3. **Visualization**: Dashboard 4 panel với:
   - Biểu đồ cột các metrics hiệu suất
   - Heatmap confusion matrix
   - Visualization RMSE
   - Tóm tắt hiệu suất

## 🏆 Hệ Thống Đánh Giá

| Điểm | Xếp Loại | Đánh Giá | Khuyến Nghị |
|------|----------|----------|-------------|
| ≥0.8 | A+ | XUẤT SẮC | Sẵn sàng cho production |
| ≥0.7 | A | TỐT | Có thể cải thiện nhỏ |
| ≥0.6 | B | CHẤP NHẬN ĐƯỢC | Cần cải thiện trước khi deploy |
| ≥0.5 | C | CẦN CẢI THIỆN | Cần cải thiện đáng kể |
| <0.5 | D | KÉM | Cần overhaul hoàn toàn |

## 🔧 Tùy Chỉnh

### Thay Đổi Kích Thước Mẫu
```python
# Trong clean_evaluate.py
evaluator.evaluate(sample_size=50)  # Test trên 50 mẫu
```

### Sử Dụng Model Thật
```python
# Trong real_model_evaluate.py
results, verdict = evaluate_model(use_real_model=True, sample_size=20)
```

### Thêm Metrics Tùy Chỉnh
```python
# Thêm metrics của bạn trong calculate_metrics() method
metrics['custom_metric'] = your_calculation()
```

## 📁 Các File

- `clean_evaluate.py` - Script đánh giá chính
- `real_model_evaluate.py` - Phiên bản AI model thật
- `evaluation_results.png` - Visualization được tạo
- `llm_training_complete.csv` - Dataset test

## 🎉 Ví Dụ Kết Quả

```
🎯 Precision:    0.8100
⚡ F1-Score:     0.8526
📈 Accuracy:     0.9000
📏 RMSE:         0.8434

👍 ĐÁNH GIÁ CUỐI CÙNG: TỐT (A)
💡 Khuyến nghị: Mô hình hoạt động tốt, có thể cải thiện nhỏ
```

## 🚨 Xử Lý Sự Cố

### Thiếu Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib
```

### Vấn Đề Dataset
- Script tự động tạo dữ liệu giả nếu không tìm thấy dataset
- Đảm bảo `llm_training_complete.csv` có cột 'content' và 'label'

### Vấn Đề Bộ Nhớ
- Giảm sample_size cho dataset lớn
- Sử dụng mock predictions để test nhanh

## 📞 Hỗ Trợ

Để giải quyết vấn đề hoặc câu hỏi về các script đánh giá, kiểm tra:
1. Tương thích phiên bản Python
2. Dependencies cần thiết
3. Định dạng dataset
4. Giới hạn bộ nhớ cho đánh giá model thật

## 💡 Lưu Ý

- Script này được viết để dễ sử dụng và mở rộng
- Có thể tùy chỉnh theo nhu cầu cụ thể của dự án
- Nên test với mock trước khi chạy model thật
- Kết quả được lưu dưới dạng hình ảnh để dễ chia sẻ 