"""
Script đánh giá mô hình AI thật cho phát hiện tin giả
Sử dụng model Vietnamese LLaMA và RAG system
Tác giả: [Tên của bạn]
Ngày tạo: 2024
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, f1_score, mean_squared_error, confusion_matrix
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import re

# Set style cho biểu đồ đẹp hơn
plt.style.use('default')

# Load models (uncomment khi muốn dùng model thật)
def load_models():
    """
    Load tất cả các models cần thiết
    Mình load một lần để tránh load lại nhiều lần
    """
    print("Đang load các models...")
    
    # Load knowledge base
    with open("documents.txt", "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f.readlines()]
    index = faiss.read_index("knowledge_base.index")
    
    # Load embedding model
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # Load LLM - model Vietnamese LLaMA
    MODEL_NAME = "vilm/vinallama-7b"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    return documents, index, embedding_model, model, tokenizer

def extract_accuracy_from_response(response):
    """
    Trích xuất độ chính xác từ response của model
    Mình dùng regex để tìm số phần trăm trong text
    """
    try:
        match = re.search(r"Độ chính xác:.*?(\d+\.?\d*)", response, re.IGNORECASE)
        if match:
            return float(match.group(1)) / 100.0
        return 0.5  # Default nếu không tìm thấy
    except:
        return 0.5

def predict_news_accuracy_real(news_snippet, documents, index, embedding_model, model, tokenizer):
    """
    Dự đoán thật bằng AI model
    Sử dụng RAG system để tìm thông tin liên quan
    """
    k = 3  # Số documents tương tự nhất
    
    # Tạo embedding và tìm documents tương tự
    query_embedding = embedding_model.encode([news_snippet], convert_to_tensor=True).cpu().numpy().astype('float32')
    distances, indices = index.search(query_embedding, k)
    retrieved_docs = [documents[i] for i in indices[0]]
    context = "\n\n---\n\n".join(retrieved_docs)
    
    # Tạo prompt cho model
    prompt = f"""<s>[INST] <<SYS>>
Bạn là một trợ lý AI chuyên phân tích tin tức tiếng Việt. Dựa vào thông tin trong BỐI CẢNH, hãy phân tích TIN TỨC CẦN KIỂM TRA và đưa ra câu trả lời gồm hai phần: một câu phân tích ngắn gọn, và một dòng riêng ghi "Độ chính xác:" theo sau là một con số phần trăm.
<</SYS>>

BỐI CẢNH:
{context}

TIN TỨC CẦN KIỂM TRA:
{news_snippet} [/INST]
PHÂN TÍCH:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256, 
        repetition_penalty=1.1,
        no_repeat_ngram_size=5,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    full_analysis = response.split("PHÂN TÍCH:")[-1].strip()
    
    return extract_accuracy_from_response(full_analysis)

def mock_predict_accuracy(news_snippet):
    """
    Hàm giả lập dự đoán để test nhanh
    """
    import random
    length_factor = min(len(news_snippet) / 500, 1.0)
    random_factor = random.random() * 0.3
    base_accuracy = 0.6 + (length_factor * 0.3) + random_factor
    return min(max(base_accuracy, 0.1), 0.95)

def plot_metrics(metrics_dict, save_path='evaluation_results.png'):
    """
    Tạo visualization toàn diện cho kết quả đánh giá
    Mình vẽ 4 biểu đồ khác nhau để phân tích đầy đủ
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Kết Quả Đánh Giá Mô Hình Phát Hiện Tin Giả', fontsize=16, fontweight='bold')
    
    # 1. Biểu đồ cột các metrics
    metrics_names = ['Precision', 'F1-Score', 'Accuracy']
    metrics_values = [metrics_dict['precision'], metrics_dict['f1_score'], metrics_dict['accuracy']]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    bars = ax1.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
    ax1.set_title('Các Metrics Hiệu Suất Mô Hình', fontweight='bold')
    ax1.set_ylabel('Điểm số')
    ax1.set_ylim(0, 1)
    
    # Thêm giá trị lên các cột
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Confusion Matrix
    cm = metrics_dict['confusion_matrix']
    im = ax2.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
    ax2.set_title('Ma Trận Nhầm Lẫn', fontweight='bold')
    
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
    metrics_for_rmse = ['Precision', 'F1-Score', 'Accuracy']
    rmse_values = [metrics_dict['rmse']] * 3
    
    ax3.plot(metrics_for_rmse, rmse_values, 'ro-', linewidth=2, markersize=8)
    ax3.set_title('RMSE (Càng thấp càng tốt)', fontweight='bold')
    ax3.set_ylabel('Giá trị RMSE')
    ax3.text(1, metrics_dict['rmse'], f'RMSE: {metrics_dict["rmse"]:.3f}', 
             ha='center', va='bottom', fontweight='bold', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 4. Tóm tắt hiệu suất
    ax4.axis('off')
    summary_text = f"""
    📊 TÓM TẮT HIỆU SUẤT MÔ HÌNH
    
    🎯 Precision: {metrics_dict['precision']:.3f}
    ⚡ F1-Score: {metrics_dict['f1_score']:.3f}
    📈 Accuracy: {metrics_dict['accuracy']:.3f}
    📏 RMSE: {metrics_dict['rmse']:.3f}
    
    📋 MA TRẬN NHẦM LẪN:
    ✅ True Positives: {cm[0][0]}
    ❌ False Positives: {cm[1][0]}
    ✅ True Negatives: {cm[1][1]}
    ❌ False Negatives: {cm[0][1]}
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8, edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig

def get_final_verdict(metrics_dict):
    """
    Tạo đánh giá cuối cùng dựa trên các metrics
    Mình dùng weighted score để cân bằng các metrics
    """
    precision = metrics_dict['precision']
    f1 = metrics_dict['f1_score']
    accuracy = metrics_dict['accuracy']
    rmse = metrics_dict['rmse']
    
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

def evaluate_model(use_real_model=False, sample_size=20):
    """
    Đánh giá mô hình sử dụng test dataset
    Mình cho phép chọn giữa model thật và mock để test
    """
    print("Đang load test dataset...")
    
    try:
        # Load dataset
        df = pd.read_csv('llm_training_complete.csv')
        print(f"Dataset đã load: {len(df)} dòng")
    except FileNotFoundError:
        print("Không tìm thấy dataset. Tạo dữ liệu giả để test...")
        np.random.seed(42)
        n_samples = 100
        df = pd.DataFrame({
            'content': [f"Tin tức mẫu {i} với nội dung chi tiết" * (i % 5 + 1) for i in range(n_samples)],
            'label': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        })
    
    # Lọc bỏ các dòng trống
    df = df.dropna(subset=['content'])
    
    # Lấy mẫu để đánh giá
    sample_size = min(sample_size, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    print(f"Đánh giá trên {len(df_sample)} mẫu...")
    
    # Load models nếu dùng model thật
    if use_real_model:
        print("Đang load AI models thật...")
        documents, index, embedding_model, model, tokenizer = load_models()
        predict_func = lambda x: predict_news_accuracy_real(x, documents, index, embedding_model, model, tokenizer)
    else:
        print("Dùng mock predictions để test nhanh...")
        predict_func = mock_predict_accuracy
    
    # Chuẩn bị dữ liệu
    X = df_sample['content'].tolist()
    y_true = df_sample['label'].tolist()
    
    # Lấy dự đoán
    y_pred_proba = []
    y_pred_binary = []
    
    for i, news in enumerate(X):
        print(f"Đang xử lý {i+1}/{len(X)}...")
        try:
            accuracy = predict_func(news)
            y_pred_proba.append(accuracy)
            y_pred_binary.append(0 if accuracy > 0.5 else 1)
        except Exception as e:
            print(f"Lỗi xử lý mẫu {i}: {e}")
            y_pred_proba.append(0.5)
            y_pred_binary.append(1)
    
    # Tính toán metrics
    print("\n=== KẾT QUẢ ĐÁNH GIÁ ===")
    
    precision = precision_score(y_true, y_pred_binary, average='weighted')
    f1 = f1_score(y_true, y_pred_binary, average='weighted')
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_proba))
    accuracy = np.mean(np.array(y_true) == np.array(y_pred_binary))
    
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    print(f"\nMa Trận Nhầm Lẫn:")
    print("      Dự đoán")
    print("      Thật  Giả")
    print(f"Thật  {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"Giả   {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # Lưu kết quả
    results = {
        'precision': precision,
        'f1_score': f1,
        'rmse': rmse,
        'accuracy': accuracy,
        'confusion_matrix': cm
    }
    
    # Tạo đánh giá cuối cùng
    verdict = get_final_verdict(results)
    
    # In đánh giá cuối cùng
    print(f"\n{'='*50}")
    print(f"🎯 ĐÁNH GIÁ CUỐI CÙNG: {verdict['verdict']}")
    print(f"📊 Điểm tổng thể: {verdict['overall_score']:.3f}")
    print(f"🏆 Xếp loại: {verdict['grade']}")
    print(f"💡 Khuyến nghị: {verdict['recommendation']}")
    print(f"{'='*50}")
    
    # Tạo visualization
    print("\n📈 Đang tạo biểu đồ...")
    plot_metrics(results)
    
    return results, verdict

if __name__ == "__main__":
    # Set use_real_model=True để test với AI model thật
    # Set use_real_model=False để test nhanh với mock predictions
    results, verdict = evaluate_model(use_real_model=False, sample_size=20)
    
    print(f"\n📋 TÓM TẮT CUỐI CÙNG:")
    print(f"Precision: {results['precision']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Đánh giá: {verdict['verdict']} ({verdict['grade']})") 