import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re

# --- Load all models ONCE at the start ---
print("Loading all necessary models... This may take a minute.")

# 1. Load the knowledge base and index
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f.readlines()]
index = faiss.read_index("knowledge_base.index")

# 2. Load the embedding model
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 3. Load the NEW, CORRECT, and STABLE Language Model and its tokenizer
MODEL_NAME = "vilm/vinallama-7b" 
print(f"Loading verified model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Set the padding token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("All components loaded successfully! Launching the app...")

# --- Main RAG and analysis function ---
def analyze_news(news_snippet: str):
    if not news_snippet or not news_snippet.strip():
        return "Please enter a news snippet to analyze.", {"N/A": 1.0}, ""

    k = 3
    
    query_embedding = embedding_model.encode([news_snippet], convert_to_tensor=True).cpu().numpy().astype('float32')
    
    distances, indices = index.search(query_embedding, k)
    retrieved_docs = [documents[i] for i in indices[0]]
    context = "\n\n---\n\n".join(retrieved_docs)
    
    # --- Llama-2 specific prompt format for stability ---
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
    
    # Use stable generation parameters
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256, 
        repetition_penalty=1.1,
        no_repeat_ngram_size=5,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    full_analysis_text = response.split("PHÂN TÍCH:")[-1].strip()

    accuracy_label = {"N/A": 1.0}
    try:
        match = re.search(r"Độ chính xác:.*?(\d+\.?\d*)", full_analysis_text, re.IGNORECASE)
        if match:
            accuracy = float(match.group(1))
            accuracy_label = {"Thật (Real)": accuracy / 100, "Giả (Fake)": 1 - (accuracy / 100)}
        else:
             accuracy_label = {"Không xác định": 1.0}
    except (ValueError, IndexError):
        accuracy_label = {"Lỗi phân tích": 1.0}

    retrieved_facts_display = "\n\n".join([f"Fact {i+1} (Distance: {distances[0][i]:.4f}):\n{doc}" for i, doc in enumerate(retrieved_docs)])
    
    return full_analysis_text, accuracy_label, retrieved_facts_display

# --- Build the interface ---
with gr.Blocks(theme="soft") as app:
    gr.Markdown("# 📰 Vietnamese News Analyzer")
    gr.Markdown("An AI tool to estimate the accuracy of a news snippet by comparing it against a knowledge base of verified articles.")

    with gr.Row():
        with gr.Column(scale=2):
            input_textbox = gr.Textbox(lines=8, placeholder="Paste a Vietnamese news snippet here...", label="News to Analyze")
            submit_button = gr.Button("Analyze", variant="primary")
        
        with gr.Column(scale=1):
            confidence_label = gr.Label(label="Estimated News Accuracy")

    main_analysis_output = gr.Textbox(label="AI Analysis", lines=6, interactive=False)
    
    with gr.Accordion("Show Detailed Retrieved Facts", open=False):
        retrieved_facts_output = gr.Textbox(label="Retrieved Facts from Knowledge Base", interactive=False)

    submit_button.click(
        fn=analyze_news,
        inputs=input_textbox,
        outputs=[main_analysis_output, confidence_label, retrieved_facts_output]
    )

if __name__ == "__main__":
    app.launch()