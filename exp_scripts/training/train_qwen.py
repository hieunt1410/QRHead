"""
Training script for fine-tuning Qwen 2.5 7B on retrieval task.

This script fine-tunes the model to output the correct/relevant document
given a query and a set of retrieved documents.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from qrretriever.custom_modeling_qwen2 import Qwen2ForCausalLM
from dataset import (
    RetrievalFineTuningDataset,
    RetrievalFineTuningDatasetWithIndex,
    load_examples,
)


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 2.5 7B on retrieval task")

    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co")
    parser.add_argument("--config_name", type=str, default=None,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Where to store the pretrained models downloaded from huggingface.co")
    parser.add_argument("--model_revision", type=str, default="main",
                        help="The specific model version to use (branch, tag name or commit id)")
    parser.add_argument("--torch_dtype", type=str, default=None,
                        choices=["auto", "bfloat16", "float16", "float32"],
                        help="Override the default torch.dtype for the model")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        help="Which attention implementation to use")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Whether to trust remote code when loading the model")

    # Data arguments
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to the training file (JSONL format)")
    parser.add_argument("--validation_file", type=str, default=None,
                        help="Path to the validation file (JSONL format)")
    parser.add_argument("--max_length", type=int, default=32768,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Truncate the number of training examples")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Truncate the number of evaluation examples")
    parser.add_argument("--preprocessing_num_workers", type=int, default=None,
                        help="The number of processes to use for preprocessing")
    parser.add_argument("--model_base_class", type=str, default="Qwen2.5-7B-Instruct",
                        help="Base model class for prompt formatting")
    parser.add_argument("--task_type", type=str, default="generate_content",
                        choices=["generate_content", "predict_index"],
                        help="Type of training task")
    parser.add_argument("--query_key", type=str, default="query",
                        help="Key name for the query field in JSONL")
    parser.add_argument("--docs_key", type=str, default="docs",
                        help="Key name for the documents field in JSONL")
    parser.add_argument("--gold_idx_key", type=str, default="gold_doc_idx",
                        help="Key name for the gold document index field in JSONL")

    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite output directory if it exists")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run evaluation")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size per GPU/TPU core/CPU for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="Batch size per GPU/TPU core/CPU for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="The initial learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay for AdamW")
    parser.add_argument("--num_train_epochs", type=float, default=3.0,
                        help="Total number of training epochs to perform")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="If > 0: set total number of training steps to perform")
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                        help="Linear warmup over warmup_ratio fraction of total steps")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Linear warmup over warmup_steps")
    parser.add_argument("--logging_steps", type=int, default=500,
                        help="Log every X updates steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X updates steps")
    parser.add_argument("--save_total_limit", type=int, default=None,
                        help="Limit the total amount of checkpoints")
    parser.add_argument("--eval_steps", type=int, default=None,
                        help="Run an evaluation every X steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use fp16 mixed precision training")
    parser.add_argument("--bf16", action="store_true",
                        help="Whether to use bf16 mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint for resuming training")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--report_to", type=str, default=None,
                        help="The integration to report results to")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name of the run for reporting")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    # Log on each process
    logger.info(f"Training parameters: {args}")

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. "
                "To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir`."
            )

    # Set seed
    set_seed(args.seed)

    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "use_fast": True,
        "revision": args.model_revision,
        "trust_remote_code": args.trust_remote_code,
    }

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, **tokenizer_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load datasets
    logger.info(f"Loading training data from {args.train_file}")
    train_examples = load_examples(
        args.train_file,
        query_key=args.query_key,
        docs_key=args.docs_key,
        gold_idx_key=args.gold_idx_key,
    )

    if args.max_train_samples is not None:
        train_examples = train_examples[:args.max_train_samples]
        logger.info(f"Truncated training data to {len(train_examples)} examples")

    eval_examples = None
    if args.validation_file:
        logger.info(f"Loading validation data from {args.validation_file}")
        eval_examples = load_examples(
            args.validation_file,
            query_key=args.query_key,
            docs_key=args.docs_key,
            gold_idx_key=args.gold_idx_key,
        )
        if args.max_eval_samples is not None:
            eval_examples = eval_examples[:args.max_eval_samples]
            logger.info(f"Truncated validation data to {len(eval_examples)} examples")

    # Create datasets
    if args.task_type == "generate_content":
        train_dataset = RetrievalFineTuningDataset(
            examples=train_examples,
            tokenizer=tokenizer,
            max_length=args.max_length,
            model_base_class=args.model_base_class,
        )
        if eval_examples:
            eval_dataset = RetrievalFineTuningDataset(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_length=args.max_length,
                model_base_class=args.model_base_class,
            )
        else:
            eval_dataset = None
    else:  # predict_index
        train_dataset = RetrievalFineTuningDatasetWithIndex(
            examples=train_examples,
            tokenizer=tokenizer,
            max_length=args.max_length,
            model_base_class=args.model_base_class,
        )
        if eval_examples:
            eval_dataset = RetrievalFineTuningDatasetWithIndex(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_length=args.max_length,
                model_base_class=args.model_base_class,
            )
        else:
            eval_dataset = None

    logger.info(f"Training dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Validation dataset size: {len(eval_dataset)}")

    # Load model
    logger.info(f"Loading model from {args.model_name_or_path}")

    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
    )

    model = Qwen2ForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=args.config_name,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
    )

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Create TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        run_name=args.run_name,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Training
    if args.do_train:
        logger.info("Starting training...")

        checkpoint = None
        if args.resume_from_checkpoint is not None:
            checkpoint = args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            args.max_train_samples if args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if args.do_eval:
        logger.info("Starting evaluation...")
        metrics = trainer.evaluate()

        max_eval_samples = (
            args.max_eval_samples if args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = max_eval_samples

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()