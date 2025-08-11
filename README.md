<<<<<<< HEAD
# NCKH_new
This is the new project for NCKH
=======
# 🌐 AI Phát Hiện Tin Giả - Vietnamese Fake News Detection Website

Một website hiện đại để phát hiện tin giả sử dụng trí tuệ nhân tạo, được viết bằng tiếng Việt.

## 🚀 Tính Năng

- **Giao diện hiện đại**: Thiết kế responsive với animations mượt mà
- **Phân tích AI**: Mô phỏng phân tích tin tức bằng AI (demo version)
- **Đa ngôn ngữ**: Hỗ trợ tiếng Việt và tiếng Anh
- **Kết quả chi tiết**: Hiển thị độ tin cậy và các yếu tố phân tích
- **Tương tác**: Form liên hệ và navigation mượt mà

## 📁 Cấu Trúc Dự Án

```
NCKH/
├── index.html          # Trang chủ website
├── styles.css          # CSS styles và animations
├── script.js           # JavaScript functionality
├── server.py           # Python HTTP server
├── README.md           # Hướng dẫn sử dụng
└── llm_training_complete.csv  # Dataset training (nếu có)
```

## 🛠️ Cài Đặt và Chạy

### Yêu Cầu Hệ Thống
- Python 3.6+ (để chạy server)
- Trình duyệt web hiện đại (Chrome, Firefox, Safari, Edge)

### Cách 1: Sử Dụng Python Server (Khuyến Nghị)

1. **Mở terminal/command prompt**
2. **Di chuyển đến thư mục dự án**:
   ```bash
   cd /path/to/NCKH
   ```

3. **Chạy server**:
   ```bash
   python server.py
   ```

4. **Mở trình duyệt** và truy cập:
   ```
   http://localhost:8000
   ```

### Cách 2: Sử Dụng Live Server (VS Code)

1. Cài đặt extension "Live Server" trong VS Code
2. Mở file `index.html`
3. Click chuột phải và chọn "Open with Live Server"

### Cách 3: Mở Trực Tiếp File

1. Double-click vào file `index.html`
2. Website sẽ mở trong trình duyệt mặc định

## 🎯 Cách Sử Dụng

### 1. Phân Tích Tin Tức
- **Nhập nội dung**: Dán nội dung bài báo vào ô "Nội dung bài báo"
- **Hoặc nhập URL**: Dán URL bài báo vào ô "URL bài báo"
- **Click "Phân Tích Ngay"**: Hệ thống sẽ mô phỏng phân tích AI

### 2. Xem Kết Quả
- **Độ tin cậy**: Hiển thị phần trăm tin cậy của bài báo
- **Kết luận**: Tin tức đáng tin cậy / Cần thận trọng / Không đáng tin cậy
- **Chi tiết**: Các yếu tố phân tích (nguồn tin, ngôn ngữ, thời gian, liên kết)

### 3. Thử Nghiệm Mẫu
- Click nút "Thử Nghiệm Mẫu" (góc dưới bên phải)
- Hệ thống sẽ tự động điền nội dung mẫu để test

## 🔧 Tùy Chỉnh

### Thay Đổi Port Server
Mở file `server.py` và thay đổi dòng:
```python
PORT = 8000  # Thay đổi số port mong muốn
```

### Thêm Nội Dung Mẫu
Trong file `script.js`, tìm mảng `sampleTexts` và thêm nội dung mẫu:
```javascript
const sampleTexts = [
    "Nội dung mẫu 1...",
    "Nội dung mẫu 2...",
    // Thêm nội dung mới ở đây
];
```

## 🤖 Tham Khảo AI Model

Website này hiện tại sử dụng mô phỏng AI. Để tích hợp AI thực tế, bạn có thể:

### 1. Sử Dụng Dataset Có Sẵn
File `llm_training_complete.csv` có thể chứa dữ liệu training cho model AI.

### 2. Tích Hợp AI Model
- **BERT Vietnamese**: Sử dụng mô hình BERT đã fine-tune cho tiếng Việt
- **Transformer Architecture**: Attention mechanism để phân tích văn bản
- **NLP Pipeline**: Xử lý ngôn ngữ tự nhiên tiếng Việt

### 3. API Integration
Thay thế hàm `simulateAnalysis()` trong `script.js` bằng API call thực tế:
```javascript
async function realAnalysis(input) {
    const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: input })
    });
    return await response.json();
}
```

## 📱 Responsive Design

Website được thiết kế responsive và hoạt động tốt trên:
- **Desktop**: 1200px+
- **Tablet**: 768px - 1199px
- **Mobile**: < 768px

## 🎨 Tính Năng UI/UX

- **Smooth Scrolling**: Cuộn mượt giữa các section
- **Loading Animations**: Hiệu ứng loading khi phân tích
- **Interactive Elements**: Hover effects và transitions
- **Modern Design**: Gradient backgrounds và shadow effects
- **Accessibility**: Hỗ trợ keyboard navigation

## 🐛 Troubleshooting

### Port Đã Được Sử Dụng
```
❌ Error: Port 8000 is already in use!
```
**Giải pháp**: Thay đổi port trong `server.py` hoặc dừng process đang sử dụng port 8000

### File Không Tìm Thấy
```
❌ Error: Missing required files
```
**Giải pháp**: Đảm bảo tất cả file (index.html, styles.css, script.js) ở cùng thư mục với server.py

### Trình Duyệt Không Mở Tự Động
```
⚠️ Could not open browser automatically
```
**Giải pháp**: Mở trình duyệt thủ công và truy cập `http://localhost:8000`

## 📞 Hỗ Trợ

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra console của trình duyệt (F12)
2. Đảm bảo tất cả file đã được tải đúng
3. Thử refresh trang hoặc restart server

## 🔮 Phát Triển Tương Lai

- [ ] Tích hợp AI model thực tế
- [ ] Thêm database để lưu lịch sử phân tích
- [ ] Hỗ trợ đa ngôn ngữ (tiếng Anh, tiếng Trung)
- [ ] Mobile app version
- [ ] API documentation
- [ ] User authentication system

## 📄 License

Dự án này được phát triển cho mục đích giáo dục và nghiên cứu.

---

**🎉 Chúc bạn sử dụng website vui vẻ!** 
>>>>>>> 3941e93 (update)
