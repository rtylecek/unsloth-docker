import unsloth
import os
import torch

from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset, load_from_disk, load_dataset
from unsloth import FastLanguageModel

MODEL_ID = "unsloth/gemma-7b-bnb-4bit" # Quantized models from unsloth for faster downloading
TRAINING_DATA_PATH = "path/to/training-dataset"
OUTPUT_DATA_PATH = "path/where/model/is/stored"
NUM_EPOCHS = 1
NUM_PROC = 12  # Number of processes for dataset processing
max_seq_length = 2048  # Added definition

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=max_seq_length, # use variable
    dtype=None,
    load_in_4bit=True,
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=1133,
    use_rslora=False,
    loftq_config=None,
)

#dataset = load_from_disk(TRAINING_DATA_PATH)
dataset = load_dataset("mlabonne/FineTome-100k", split = "train")

# Define a simple formatting function (placeholder)
def format_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

# Initialize data collator
data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template="<|endoftext|>", padding=True, max_length=max_seq_length)

sft_trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=data_collator,
    formatting_func=format_prompts_func,
    max_seq_length=max_seq_length,
    dataset_num_proc=NUM_PROC,
    packing=False, 
    args=TrainingArguments(
        gradient_accumulation_steps=4,
        auto_find_batch_size=True,
        warmup_steps=5,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=2.5e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=1133,
        output_dir=OUTPUT_DATA_PATH,
        report_to="tensorboard",  # Enable TensorBoard logging
        logging_dir=os.path.join(OUTPUT_DATA_PATH, "runs"),  # TensorBoard log directory
    ),
)
sft_trainer.train()

try:
    model.save_pretrained_merged(
        os.path.join(OUTPUT_DATA_PATH, "model-16bit"),
        tokenizer,
        save_method="merged_16bit",
    )
except Exception as e:
    print("Error saving merged_16bit model")
    print(e)

try:
    # Merge to 4bit
    model.save_pretrained_merged(
        os.path.join(OUTPUT_DATA_PATH, "model-4bit"),
        tokenizer,
        save_method="merged_4bit",
    )
except Exception as e:
    print("Error saving merged_4bit model")
    print(e)


try:
    # Just LoRA adapters
    model.save_pretrained_merged(
        os.path.join(OUTPUT_DATA_PATH, "model-lora"),
        tokenizer,
        save_method="lora",
    )
except Exception as e:
    print("Error saving lora model")
    print(e)
