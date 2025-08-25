"""
Script đánh giá mô hình phát hiện tin giả
Sử dụng các metrics chuẩn: Precision, F1-Score, RMSE, và Accuracy
Tác giả: Ngõng Luyên
Ngày tạo: 25/08/2025
"""
#Imprt thư việnn
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, f1_score, mean_squared_error, confusion_matrix
import matplotlib.pyplot as plt
import random
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Class để đánh giá mô hình với các metrics toàn diện và visualization
    Mình viết class này để code dễ quản lý và mở rộng hơn
    """
    
    def __init__(self, use_mock=True):
        self.use_mock = use_mock
        self.results = {}
        self.verdict = {}
        
    def mock_predict_accuracy(self, news_snippet):
        """
        Hàm giả lập dự đoán để test nhanh
        Trong thực tế sẽ thay bằng model AI thật
        """
        # Giả lập dựa trên độ dài text và một chút randomness
        length_factor = min(len(news_snippet) / 500, 1.0)
        random_factor = random.random() * 0.3
        base_accuracy = 0.6 + (length_factor * 0.3) + random_factor
        return min(max(base_accuracy, 0.1), 0.95)  # Giới hạn từ 0.1 đến 0.95
    
    def load_data(self, file_path='llm_training_complete.csv'):
        """
        Load và chuẩn bị dataset để đánh giá
        Nếu không tìm thấy file thì tạo data giả để test
        """
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Đã load dataset: {len(df)} dòng dữ liệu")
            return df
        except FileNotFoundError:
            print("⚠️ Không tìm thấy dataset. Tạo dữ liệu giả để test...")
            return self.create_synthetic_data()
    
    def create_synthetic_data(self, n_samples=100):
        """
        Tạo dữ liệu giả để test khi không có dataset thật
        """
        np.random.seed(42)  # Để kết quả reproducible
        df = pd.DataFrame({
            'content': [f"Tin tức mẫu {i} với nội dung chi tiết" * (i % 5 + 1) for i in range(n_samples)],
            'label': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # 70% tin thật, 30% tin giả
        })
        return df
    
    def calculate_metrics(self, y_true, y_pred_binary, y_pred_proba):
        """
        Tính toán tất cả các metrics đánh giá
        Mình dùng weighted average để xử lý imbalanced data
        """
        metrics = {}
        
        # Các metrics cơ bản
        metrics['precision'] = precision_score(y_true, y_pred_binary, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred_binary, average='weighted', zero_division=0)
        metrics['accuracy'] = np.mean(np.array(y_true) == np.array(y_pred_binary))
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred_proba))
        
        # Confusion matrix để phân tích chi tiết
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred_binary)
        
        return metrics
    
    def generate_verdict(self, metrics):
        """
        Tạo đánh giá cuối cùng dựa trên các metrics
        Mình dùng weighted score để cân bằng các metrics khác nhau
        """
        precision = metrics['precision']
        f1 = metrics['f1_score']
        accuracy = metrics['accuracy']
        rmse = metrics['rmse']
        
        # Tính điểm tổng thể (weighted average)
        overall_score = (precision * 0.3 + f1 * 0.3 + accuracy * 0.3 + (1 - rmse) * 0.1)
        
        # Xác định mức độ đánh giá
        if overall_score >= 0.8:
            verdict = "XUẤT SẮC"
            grade = "A+"
            recommendation = "Mô hình sẵn sàng cho production"
            emoji = "🎉"
        elif overall_score >= 0.7:
            verdict = "TỐT"
            grade = "A"
            recommendation = "Mô hình hoạt động tốt, có thể cải thiện nhỏ"
            emoji = "👍"
        elif overall_score >= 0.6:
            verdict = "CHẤP NHẬN ĐƯỢC"
            grade = "B"
            recommendation = "Cần cải thiện trước khi deploy"
            emoji = "✅"
        elif overall_score >= 0.5:
            verdict = "CẦN CẢI THIỆN"
            grade = "C"
            recommendation = "Cần cải thiện đáng kể"
            emoji = "⚠️"
        else:
            verdict = "KÉM"
            grade = "D"
            recommendation = "Cần overhaul hoàn toàn"
            emoji = "❌"
        
        return {
            'overall_score': overall_score,
            'verdict': verdict,
            'grade': grade,
            'recommendation': recommendation,
            'emoji': emoji
        }
    
    def create_visualizations(self, metrics, save_path='evaluation_results.png'):
        """
        Tạo dashboard visualization toàn diện
        Mình dùng matplotlib để vẽ 4 biểu đồ khác nhau
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dashboard Đánh Giá Mô Hình Phát Hiện Tin Giả', fontsize=18, fontweight='bold')
        
        # 1. Biểu đồ cột các metrics
        metrics_names = ['Precision', 'F1-Score', 'Accuracy']
        metrics_values = [metrics['precision'], metrics['f1_score'], metrics['accuracy']]
        colors = ['#2E86AB', '#A23B72', '#F18F01']  # Màu đẹp cho biểu đồ
        
        bars = ax1.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('Các Metrics Hiệu Suất Mô Hình', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Điểm số', fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.grid(axis='y', alpha=0.3)
        
        # Thêm giá trị lên các cột
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # 2. Confusion Matrix
        cm = metrics['confusion_matrix']
        im = ax2.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
        ax2.set_title('Ma Trận Nhầm Lẫn', fontweight='bold', fontsize=14)
        
        # Thêm số liệu vào matrix
        for i in range(2):
            for j in range(2):
                ax2.text(j, i, str(cm[i, j]), ha="center", va="center", 
                        color="white" if cm[i, j] > cm.max() / 2 else "black", 
                        fontweight='bold', fontsize=14)
        
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        ax2.set_xticklabels(['Thật', 'Giả'], fontweight='bold')
        ax2.set_yticklabels(['Thật', 'Giả'], fontweight='bold')
        ax2.set_xlabel('Dự đoán', fontweight='bold')
        ax2.set_ylabel('Thực tế', fontweight='bold')
        
        # 3. Biểu đồ RMSE
        ax3.plot(['Precision', 'F1-Score', 'Accuracy'], [metrics['rmse']] * 3, 
                'ro-', linewidth=3, markersize=10, markerfacecolor='red', markeredgecolor='black')
        ax3.set_title('RMSE (Càng thấp càng tốt)', fontweight='bold', fontsize=14)
        ax3.set_ylabel('Giá trị RMSE', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        ax3.text(1, metrics['rmse'], f'RMSE: {metrics["rmse"]:.3f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 4. Tóm tắt hiệu suất
        ax4.axis('off')
        summary_text = f"""
        📊 TÓM TẮT HIỆU SUẤT MÔ HÌNH
        
        🎯 Precision: {metrics['precision']:.3f}
        ⚡ F1-Score: {metrics['f1_score']:.3f}
        📈 Accuracy: {metrics['accuracy']:.3f}
        📏 RMSE: {metrics['rmse']:.3f}
        
        📋 MA TRẬN NHẦM LẪN:
        ✅ True Positives: {cm[0][0]}
        ❌ False Positives: {cm[1][0]}
        ✅ True Negatives: {cm[1][1]}
        ❌ False Negatives: {cm[0][1]}
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8, edgecolor='black'))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        return fig
    
    def print_results(self, metrics, verdict):
        """
        In kết quả đánh giá ra console
        Format đẹp và dễ đọc
        """
        print("\n" + "="*70)
        print("📊 KẾT QUẢ ĐÁNH GIÁ")
        print("="*70)
        print(f"🎯 Precision:    {metrics['precision']:.4f}")
        print(f"⚡ F1-Score:     {metrics['f1_score']:.4f}")
        print(f"📈 Accuracy:     {metrics['accuracy']:.4f}")
        print(f"📏 RMSE:         {metrics['rmse']:.4f}")
        
        cm = metrics['confusion_matrix']
        print(f"\n📋 Ma Trận Nhầm Lẫn:")
        print("      Dự đoán")
        print("      Thật  Giả")
        print(f"Thật  {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"Giả   {cm[1][0]:4d}  {cm[1][1]:4d}")
        
        print(f"\n{verdict['emoji']} ĐÁNH GIÁ CUỐI CÙNG: {verdict['verdict']}")
        print(f"📊 Điểm tổng thể: {verdict['overall_score']:.3f}")
        print(f"🏆 Xếp loại: {verdict['grade']}")
        print(f"💡 Khuyến nghị: {verdict['recommendation']}")
        print("="*70)
    
    def evaluate(self, sample_size=30):
        """
        Hàm chính để đánh giá mô hình
        Mình giới hạn sample size để chạy nhanh, có thể điều chỉnh
        """
        print("🚀 Bắt đầu đánh giá mô hình...")
        
        # Load dữ liệu
        df = self.load_data()
        df = df.dropna(subset=['content'])  # Bỏ các dòng trống
        
        # Lấy mẫu để đánh giá
        sample_size = min(sample_size, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)  # Để reproducible
        print(f"📝 Đánh giá trên {len(df_sample)} mẫu...")
        
        # Chuẩn bị dữ liệu
        X = df_sample['content'].tolist()
        y_true = df_sample['label'].tolist()  # 0 = thật, 1 = giả
        
        # Lấy dự đoán
        y_pred_proba = []
        y_pred_binary = []
        
        for i, news in enumerate(X, 1):
            print(f"Đang xử lý {i}/{len(X)}...", end='\r')
            try:
                accuracy = self.mock_predict_accuracy(news)
                y_pred_proba.append(accuracy)
                # Chuyển thành binary: accuracy > 0.5 = thật (0), <= 0.5 = giả (1)
                y_pred_binary.append(0 if accuracy > 0.5 else 1)
            except Exception as e:
                print(f"\nLỗi xử lý mẫu {i}: {e}")
                y_pred_proba.append(0.5)
                y_pred_binary.append(1)
        
        print(f"\n✅ Hoàn thành xử lý!")
        
        # Tính toán metrics
        self.results = self.calculate_metrics(y_true, y_pred_binary, y_pred_proba)
        
        # Tạo đánh giá cuối cùng
        self.verdict = self.generate_verdict(self.results)
        
        # In kết quả
        self.print_results(self.results, self.verdict)
        
        # Tạo visualization
        print("\n📈 Đang tạo biểu đồ...")
        self.create_visualizations(self.results)
        
        return self.results, self.verdict

def main():
    """
    Hàm main để chạy đánh giá
    Mình dùng mock predictions để test nhanh
    """
    evaluator = ModelEvaluator(use_mock=True)
    results, verdict = evaluator.evaluate(sample_size=30)
    
    print(f"\n📋 TÓM TẮT CUỐI CÙNG:")
    print(f"Precision: {results['precision']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Đánh giá: {verdict['verdict']} ({verdict['grade']}) {verdict['emoji']}")

if __name__ == "__main__":
    main() 