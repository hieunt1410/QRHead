"""
Training script for fine-tuning Qwen 2.5 using Unsloth + LoRA.

Based on Kaggle notebook: https://www.kaggle.com/code/ksmooi/fine-tuning-qwen-2-5-coder-14b-llm-sft-peft
And Unsloth docs: https://github.com/unslothai/unsloth
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

    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    parser.add_argument("--max_seq_length", type=int, default=16384)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--target_modules", type=str, default="all-linear")

    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--task_type", type=str, default="generate_content",
                        choices=["generate_content", "predict_index"])
    parser.add_argument("--query_key", type=str, default="query")
    parser.add_argument("--docs_key", type=str, default="docs")
    parser.add_argument("--gold_idx_key", type=str, default="gold_doc_idx")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)  # -1 means use num_train_epochs
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

    # Unsloth imports
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error("Unsloth not installed. Install with: pip install unsloth")
        sys.exit(1)

    from transformers import TrainingArguments
    from trl import SFTTrainer

    logger.info(f"Loading model: {args.model_name}")

    # Load model with Unsloth - following the standard pattern
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detection
        load_in_4bit=True,
    )

    # Configure LoRA
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

    # Print trainable params
    model.print_trainable_parameters()

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
        bf16=True,
        fp16=False,
        max_grad_norm=1.0,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        report_to="none",
        save_strategy="steps",
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.save_steps if eval_dataset else None,
        do_eval=True if eval_dataset else False,
    )

    # Create trainer - use SFTTrainer from TRL (works well with Unsloth)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="",  # Not used since we pre-process in dataset
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