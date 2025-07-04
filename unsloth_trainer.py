import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only

import os, subprocess
import torch
import argparse
import logging

from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset, load_from_disk, load_dataset

parser = argparse.ArgumentParser(description="Unsloth Trainer Script")
parser.add_argument('--model', type=str, default="unsloth/gemma-7b-bnb-4bit", help='Model ID to use')
parser.add_argument('--template', type=str, default="gemma-3", help='Chat template to use for the tokenizer')
parser.add_argument('--output_path', type=str, default=".", help='Path to save the output model')
parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs')
parser.add_argument('--num_proc', type=int, default=12, help='Number of processes for dataset processing')
parser.add_argument('--seq_length', type=int, default=2048, help='Maximum sequence length')
parser.add_argument('--host', type=str, default="localhost", help='External host address for TensorBoard')
parser.add_argument('--port', type=int, default=6006, help='External port for TensorBoard')
parser.add_argument('--dataset', type=str, default="mlabonne/FineTome-100k", help='Dataset hf name or local path')
parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

def train(model, template, output_path, log_path, num_epochs, num_proc, max_seq_length, dataset, save=True):
    """
    Fine-tunes a language model using a specified dataset and saves the resulting models and logs.
    Args:
        model (str): The name or path of the pretrained model to load.
        output_path (str): Directory where the trained models will be saved.
        log_path (str): Directory where training logs (e.g., TensorBoard) will be stored.
        num_epochs (int): Number of training epochs.
        num_proc (int): Number of processes to use for data preprocessing (currently unused).
        max_seq_length (int): Maximum sequence length for model inputs.
        dataset (str): Path to a local dataset or identifier for a Hugging Face dataset.
    Workflow:
        1. Loads the specified pretrained model and tokenizer.
        2. Applies a chat template to the tokenizer.
        3. Patches the model for parameter-efficient fine-tuning (LoRA).
        4. Loads and standardizes the dataset, either from disk or Hugging Face Hub.
        5. Formats the dataset for chat-based training.
        6. Initializes a data collator for language modeling.
        7. Sets up the supervised fine-tuning trainer with training arguments.
        8. Trains the model, logging progress to TensorBoard.
        9. Waits for a keyboard interrupt to allow inspection of logs.
        10. Saves the trained model in multiple formats: merged 16-bit, merged 4-bit, and LoRA adapters only.
    Exceptions:
        Logs errors if saving any of the model formats fails.
    Note:
        - The function blocks after training until interrupted by the user.
    """
    
   
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model,
        max_seq_length=max_seq_length, # use variable
        dtype=None,
        load_in_4bit=True,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = template,
    )

    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        finetune_vision_layers     = False, # Turn off for just text!
        finetune_language_layers   = True,  # Should leave on!
        finetune_attention_modules = True,  # Attention good for GRPO
        finetune_mlp_modules       = True,  # Should leave on always!
        # target_modules=[
        #     "q_proj",
        #     "k_proj",
        #     "v_proj",
        #     "o_proj",
        #     "gate_proj",
        #     "up_proj",
        #     "down_proj",
        # ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=1133,
        use_rslora=False,
        loftq_config=None,
    )

    if dataset.startswith('/') or os.path.exists(dataset):
        logging.info(f"Loading local dataset from {dataset}")
        dataset = load_from_disk(dataset)
    else:
        logging.info(f"Loading dataset from Hugging Face: {dataset}")
        dataset = load_dataset(dataset, split = "train")
    
    # Ensure the dataset is in the correct format
    dataset = standardize_data_formats(dataset)

    # Define a simple formatting function (placeholder)
    def format_prompts_func(examples):
       convos = examples["conversations"]
       texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
       return { "text" : texts, }

    dataset = dataset.map(format_prompts_func, batched=True, num_proc=num_proc)

    # Initialize data collator
    data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template="<|endoftext|>")
    tb_log_dir = os.path.join(log_path, str(args.model).replace("/", "."))
    logging.info(f"TensorBoard logs will be saved to {tb_log_dir}")

    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=data_collator,
        formatting_func=format_prompts_func,
        args=TrainingArguments(
            gradient_accumulation_steps=4,
            auto_find_batch_size=True,
            warmup_steps=5,
            num_train_epochs=num_epochs,
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
            logging_dir=tb_log_dir,  # TensorBoard log directory
        ),
    )

    # Train on responses only
    sft_trainer = train_on_responses_only(
        sft_trainer,
        instruction_part = "<start_of_turn>user\n",
        response_part = "<start_of_turn>model\n",
    )
    # Train the model
    logging.info(f"Starting training for {num_epochs} epochs with {num_proc} processes")
    train_stats = sft_trainer.train()
    logging.info(f"Training completed with stats: {train_stats}")

    if save:
        # Save the model in various formats
        try:
            model.save_pretrained_merged(
                os.path.join(output_path, "model-16bit"),
                tokenizer,
                save_method="merged_16bit",
            )
        except Exception as e:
            logging.error("Error saving merged_16bit model")
            logging.error(e)

        try:
            # Merge to 4bit
            model.save_pretrained_merged(
                os.path.join(output_path, "model-4bit"),
                tokenizer,
                save_method="merged_4bit",
            )
        except Exception as e:
            logging.error("Error saving merged_4bit model")
            logging.error(e)

        try:
            # Just LoRA adapters
            model.save_pretrained_merged(
                os.path.join(output_path, "model-lora"),
                tokenizer,
                save_method="lora",
            )
        except Exception as e:
            logging.error("Error saving lora model")
            logging.error(e)

if __name__ == "__main__":
    args = parser.parse_args()

    output_path = os.path.expanduser(args.output_path)
    log_path = os.path.join(output_path, "log")
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='[unsloth_trainer] %(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_path, "unsloth_trainer.log")),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to {log_path}")
    if len(args.host) > 0:
        logging.info(f"TensorBoard will be hosted on {args.host}:{args.port}")
        subprocess.Popen(["tensorboard", "--logdir", log_path, "--host", args.host, "--port", str(args.port)]) #, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Start training
    train(
        args.model,
        args.template,
        args.output_path,
        log_path,
        args.num_epochs,
        args.num_proc,
        args.seq_length,
        args.dataset
    )

    logging.info("Training completed. Press Ctrl+C to exit.")
    # wait for keyboard interrupt
    try:
        while True:
            pass
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Exiting...")
