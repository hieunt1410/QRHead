#!/bin/bash

# Training script for fine-tuning Qwen 2.5 7B on retrieval task
# Adjust these parameters based on your hardware and requirements

# Model and data paths
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
TRAIN_FILE="data/train.json"
VALID_FILE="data/valid.json"
OUTPUT_DIR="outputs/qwen25-7b-retrieval"

# Create output directory
mkdir -p $OUTPUT_DIR

# Training hyperparameters
BATCH_SIZE=2              # Per-device batch size (adjust based on GPU memory)
GRAD_ACCUM=4              # Gradient accumulation steps (effective batch size = BATCH_SIZE * GRAD_ACCUM * NUM_GPUS)
NUM_EPOCHS=3              # Number of training epochs
LEARNING_RATE=2e-5        # Learning rate
MAX_LENGTH=16384           # Maximum sequence length
WARMUP_RATIO=0.1          # Warmup ratio

# System settings
NUM_GPUS=1                # Number of GPUs
SEED=42                   # Random seed

# Task type: "generate_content" or "predict_index"
TASK_TYPE="generate_content"

# Run training
python exp_scripts/training/train_qwen.py \
    --model_name_or_path $MODEL_NAME \
    --train_file $TRAIN_FILE \
    --validation_file $VALID_FILE \
    --output_dir $OUTPUT_DIR \
    --task_type $TASK_TYPE \
    --max_length $MAX_LENGTH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 3 \
    --bf16 \
    --gradient_checkpointing \
    --report_to wandb \
    --run_name qwen25-7b-retrieval-ft \
    --seed $SEED \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --trust_remote_code \
    --do_train \
    --do_eval \
    "$@"

# Alternative single GPU command (for testing or if you have 1 GPU):
# python exp_scripts/training/train_qwen.py \
#     --model_name_or_path $MODEL_NAME \
#     --train_file $TRAIN_FILE \
#     --validation_file $VALID_FILE \
#     --output_dir $OUTPUT_DIR \
#     --task_type $TASK_TYPE \
#     --max_length $MAX_LENGTH \
#     --per_device_train_batch_size $BATCH_SIZE \
#     --per_device_eval_batch_size $BATCH_SIZE \
#     --gradient_accumulation_steps $GRAD_ACCUM \
#     --num_train_epochs $NUM_EPOCHS \
#     --learning_rate $LEARNING_RATE \
#     --bf16 True \
#     --gradient_checkpointing True \
#     --torch_dtype bfloat16