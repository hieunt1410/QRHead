#!/bin/bash

# Training script for fine-tuning Qwen 2.5 7B on retrieval task using LoRA
# This is much more memory-efficient than full fine-tuning

# Model and data paths
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
TRAIN_FILE="data/train.json"
VALID_FILE="data/valid.json"
OUTPUT_DIR="outputs/qwen25-7b-retrieval-lora"

# Create output directory
mkdir -p $OUTPUT_DIR

# Training hyperparameters (optimized for memory efficiency)
BATCH_SIZE=1              # Per-device batch size
GRAD_ACCUM=4             # Gradient accumulation (effective batch size = 1 * 16 = 16)
NUM_EPOCHS=3              # Number of training epochs
LEARNING_RATE=2e-4        # Learning rate (higher for LoRA)
MAX_LENGTH=16392           # Maximum sequence length (reduced for memory)
WARMUP_RATIO=0.03         # Warmup ratio

# LoRA hyperparameters
LORA_R=16                 # LoRA rank (higher = more parameters but better quality)
LORA_ALPHA=32             # LoRA alpha (typically 2*r)
LORA_DROPOUT=0.05         # LoRA dropout
TARGET_MODULES="all-linear"  # Apply LoRA to all linear layers

# System settings
NUM_GPUS=1                # Number of GPUs
SEED=42                   # Random seed

# Task type: "generate_content" or "predict_index"
TASK_TYPE="generate_content"

echo "Starting LoRA training with the following settings:"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Effective batch size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Max length: $MAX_LENGTH"
echo "  LoRA rank: $LORA_R"
echo "  Learning rate: $LEARNING_RATE"
echo ""

# Run training
python exp_scripts/training/train_qwen_lora.py \
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
    --save_steps 100 \
    --eval_steps 100 \
    --save_total_limit 2 \
    --bf16 \
    --gradient_checkpointing \
    --report_to none \
    --run_name qwen25-7b-retrieval-lora \
    --seed $SEED \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --trust_remote_code \
    --use_lora \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules $TARGET_MODULES \
    --do_train \
    --do_eval \
    "$@"