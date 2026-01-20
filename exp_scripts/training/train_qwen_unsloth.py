"""
Training script for fine-tuning Qwen 2.5 using Unsloth + LoRA.

Unsloth is 2x faster and uses less memory than standard HF Trainer.
Based on: https://www.kaggle.com/code/ksmooi/fine-tuning-qwen-2-5-coder-14b-llm-sft-peft
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from dataset import (
    RetrievalFineTuningDataset,
    RetrievalFineTuningDatasetWithIndex,
    load_examples,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 2.5 using Unsloth + LoRA")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit",
                        help="Model name (use Unsloth quantized version for memory efficiency)")
    parser.add_argument("--max_seq_length", type=int, default=16384,
                        help="Maximum sequence length")
    parser.add_argument("--dtype", type=str, default=None,
                        choices=[None, "float16", "bfloat16", "float32"],
                        help="Data type (None = auto)")

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, default="all-linear",
                        help="Target modules for LoRA")

    # Data arguments
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training data (JSON)")
    parser.add_argument("--validation_file", type=str, default=None,
                        help="Path to validation data (JSON)")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit training samples")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Limit eval samples")
    parser.add_argument("--task_type", type=str, default="generate_content",
                        choices=["generate_content", "predict_index"],
                        help="Training task type")
    parser.add_argument("--query_key", type=str, default="query",
                        help="Query key in JSON")
    parser.add_argument("--docs_key", type=str, default="docs",
                        help="Documents key in JSON")
    parser.add_argument("--gold_idx_key", type=str, default="gold_doc_idx",
                        help="Gold index key in JSON")

    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max steps (overrides epochs)")
    parser.add_argument("--warmup_steps", type=int, default=10,
                        help="Warmup steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save frequency")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error("Unsloth not installed. Install with: pip install unsloth")
        sys.exit(1)

    from transformers import TrainingArguments

    logger.info(f"Loading model: {args.model_name}")

    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None if args.dtype is None else args.dtype,
        load_in_4bit=True,  # 4-bit quantization for memory efficiency
    )

    # Configure LoRA
    # Handle target_modules - "all-linear" is a special string for Unsloth
    target_modules = args.target_modules
    if target_modules == "all-linear":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    logger.info(f"Loading training data from {args.train_file}")
    train_examples = load_examples(
        args.train_file,
        query_key=args.query_key,
        docs_key=args.docs_key,
        gold_idx_key=args.gold_idx_key,
    )

    if args.max_train_samples:
        train_examples = train_examples[:args.max_train_samples]

    eval_examples = None
    if args.validation_file:
        eval_examples = load_examples(
            args.validation_file,
            query_key=args.query_key,
            docs_key=args.docs_key,
            gold_idx_key=args.gold_idx_key,
        )
        if args.max_eval_samples:
            eval_examples = eval_examples[:args.max_eval_samples]

    # Create dataset
    if args.task_type == "generate_content":
        train_dataset = RetrievalFineTuningDataset(
            examples=train_examples,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            model_base_class="Qwen2.5-7B-Instruct",
        )
        if eval_examples:
            eval_dataset = RetrievalFineTuningDataset(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_length=args.max_seq_length,
                model_base_class="Qwen2.5-7B-Instruct",
            )
        else:
            eval_dataset = None
    else:
        train_dataset = RetrievalFineTuningDatasetWithIndex(
            examples=train_examples,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            model_base_class="Qwen2.5-7B-Instruct",
        )
        if eval_examples:
            eval_dataset = RetrievalFineTuningDatasetWithIndex(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_length=args.max_seq_length,
                model_base_class="Qwen2.5-7B-Instruct",
            )
        else:
            eval_dataset = None

    logger.info(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval dataset size: {len(eval_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        bf16=True,  # Use bfloat16
        fp16=False,
        max_grad_norm=1.0,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        report_to="none",
        save_strategy="steps",
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=args.save_steps if eval_dataset else None,
        do_eval=True if eval_dataset else False,
    )

    # Create trainer
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save model
    logger.info(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()