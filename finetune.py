import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# --- Configuration ---
MODEL_NAME = "vilm/vinallama-7b"
NEW_MODEL_NAME = "vinallama-7b-fake-news-tuned" # The name for your new, fine-tuned model
DATASET_PATH = "train_data.csv" # The 80% training data we just created

# --- Load the dataset and format it ---
dataset = load_dataset("csv", data_files=DATASET_PATH, split="train")

# We need a function to format our data into a prompt
def format_prompt(example):
    # Map the label '0' to the word 'thật' (real) and '1' to 'giả' (fake)
    label_text = "thật" if example["label"] == 0 else "giả"
    
    # This is the instruction template the model will learn from
    prompt = f"""<s>[INST] <<SYS>>
Bạn là một chuyên gia phân loại tin tức. Phân loại tin tức sau đây là 'thật' hoặc 'giả'.
<</SYS>>

Tin tức: {example['post_message']} [/INST]
Phân loại: {label_text}
"""
    return {"text": prompt}

formatted_dataset = dataset.map(format_prompt)

# --- Set up quantization and LoRA ---
# 4-bit quantization to reduce memory usage
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# LoRA configuration
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"], # Specific layers to apply LoRA to
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- Load the model and tokenizer ---
print(f"Loading base model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
)

# --- Set up Training ---
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True, # Use bfloat16 for stability on your GPU
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)

# --- Start Training ---
print("Starting fine-tuning...")
trainer.train()

# --- Save the fine-tuned model ---
print(f"Saving fine-tuned model to ./{NEW_MODEL_NAME}")
trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)