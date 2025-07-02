import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only

import os, subprocess
import torch
import argparse

from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset, load_from_disk, load_dataset

def main():
    parser = argparse.ArgumentParser(description="Unsloth Trainer Script")
    parser.add_argument('--model', type=str, default="unsloth/gemma-7b-bnb-4bit", help='Model ID to use')
    parser.add_argument('--output_path', type=str, default=".", help='Path to save the output model')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--num_proc', type=int, default=12, help='Number of processes for dataset processing')
    parser.add_argument('--seq_length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--host', type=str, default="localhost", help='Host for TensorBoard')
    #parser.add_argument('--training_data_path', type=str, default="~/Code/unsloth-docker/data/FineTome-100k", help='Path to the training data')
    args = parser.parse_args()

    model_name = args.model
    max_seq_length = args.seq_length
    output_path = os.path.expanduser(args.output_path)
    log_path = os.path.join(output_path, "log")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    print(f"Logging to {log_path}")

    subprocess.Popen(["tensorboard", "--logdir", log_path, "--host", args.host, "--port", "6006"]) #, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length, # use variable
        dtype=None,
        load_in_4bit=True,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma-3",
    )

    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        # finetune_vision_layers     = False, # Turn off for just text!
        # finetune_language_layers   = True,  # Should leave on!
        # finetune_attention_modules = True,  # Attention good for GRPO
        # finetune_mlp_modules       = True,  # SHould leave on always!
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

    #dataset = load_from_disk(args.training_data_path)
    dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
    dataset = standardize_data_formats(dataset)

    # Define a simple formatting function (placeholder)
    def format_prompts_func(examples):
       convos = examples["conversations"]
       texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
       return { "text" : texts, }

    dataset = dataset.map(format_prompts_func, batched = True)

    # Initialize data collator
    data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template="<|endoftext|>")

    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=data_collator,
        formatting_func=format_prompts_func,
        # max_seq_length=max_seq_length,
        # dataset_num_proc=NUM_PROC,
        # packing=False, 
        args=TrainingArguments(
            gradient_accumulation_steps=4,
            auto_find_batch_size=True,
            warmup_steps=5,
            num_train_epochs=args.num_epochs,
            learning_rate=2.5e-5,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=1133,
            output_dir=output_path,
            report_to="tensorboard",  # Enable TensorBoard logging
            logging_dir=os.path.join(log_path, str(args.model).replace("/", ".")),  # TensorBoard log directory
        ),
    )

    sft_trainer = train_on_responses_only(
        sft_trainer,
        instruction_part = "<start_of_turn>user\n",
        response_part = "<start_of_turn>model\n",
    )

    train_stats = sft_trainer.train()

    try:
        model.save_pretrained_merged(
            os.path.join(output_path, "model-16bit"),
            tokenizer,
            save_method="merged_16bit",
        )
    except Exception as e:
        print("Error saving merged_16bit model")
        print(e)

    try:
        # Merge to 4bit
        model.save_pretrained_merged(
            os.path.join(output_path, "model-4bit"),
            tokenizer,
            save_method="merged_4bit",
        )
    except Exception as e:
        print("Error saving merged_4bit model")
        print(e)

    try:
        # Just LoRA adapters
        model.save_pretrained_merged(
            os.path.join(output_path, "model-lora"),
            tokenizer,
            save_method="lora",
        )
    except Exception as e:
        print("Error saving lora model")
        print(e)

if __name__ == "__main__":
    main()
