"""
Training script for fine-tuning Qwen 2.5 using Unsloth + LoRA.
"""

import argparse
import logging
import sys
from pathlib import Path

from dataset import load_examples_as_dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Now it is safe to import modules that use transformers


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen 2.5 using Unsloth + LoRA"
    )

    # Model config
    parser.add_argument(
        "--model_name", type=str, default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=4096
    )  # Qwen supports up to 32k, but 4k-8k is usually enough for RAG

    # LoRA config
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument(
        "--lora_alpha", type=int, default=16
    )  # usually alpha = r or alpha = 2*r
    parser.add_argument(
        "--lora_dropout", type=float, default=0.0
    )  # 0.0 is optimized for Unsloth

    # Data config
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument(
        "--task_type",
        type=str,
        default="generate_content",
        choices=["generate_content", "predict_index"],
    )

    # Training config
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hub_model_id", type=str, default=None,
                       help="HuggingFace Hub model ID (e.g., 'username/model-name')")
    parser.add_argument("--hub_token", type=str, default=None,
                       help="HuggingFace Hub token. Use HF_TOKEN env var if not specified.")
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Push model to HuggingFace Hub after training")

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )

    logger.info(f"Loading model: {args.model_name}")

    # 1. Load Model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detection (Float16 or Bfloat16)
        load_in_4bit=True,
    )

    # 2. Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # Ensure pad token is set for padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Load & Prepare Data
    logger.info(f"Loading data from {args.train_file}")

    # Load dataset with formatted text column (chat template applied)
    train_dataset = load_examples_as_dataset(
        args.train_file,
        model_base_class=args.model_name,
        include_doc_index=(args.task_type == "predict_index"),
        tokenizer=tokenizer,
    )

    if args.max_train_samples:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))

    eval_dataset = None
    if args.validation_file:
        eval_dataset = load_examples_as_dataset(
            args.validation_file,
            model_base_class=args.model_name,
            include_doc_index=(args.task_type == "predict_index"),
            tokenizer=tokenizer,
        )
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Dataset columns: {train_dataset.column_names}")

    # 4. Training Arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        logging_first_step=True,
        optim="adamw_8bit",  # Use 8-bit optimizer to save memory
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        report_to="none",  # Change to "wandb" if you want tracking
        save_total_limit=2,
    )

    # 5. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
    )

    # 6. Train
    logger.info("Starting training...")
    trainer_stats = trainer.train()

    # 7. Save
    logger.info(f"Saving to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Push to HuggingFace Hub if requested
    if args.push_to_hub:
        if args.hub_model_id is None:
            logger.warning("--hub_model_id is required for push_to_hub. Skipping...")
        else:
            logger.info(f"Pushing to HuggingFace Hub: {args.hub_model_id}")
            model.push_to_hub(
                args.hub_model_id,
                token=args.hub_token,
            )
            tokenizer.push_to_hub(
                args.hub_model_id,
                token=args.hub_token,
            )
            logger.info("Successfully pushed to HuggingFace Hub!")

    # Save GGUF if needed (Unsloth feature)
    # model.save_pretrained_gguf(args.output_dir, tokenizer, quantization_method = "q4_k_m")

    logger.info("Done!")


if __name__ == "__main__":
    main()
