#!/bin/bash

# Training script using Unsloth - 2x faster, less memory
# Based on: https://www.kaggle.com/code/ksmooi/fine-tuning-qwen-2-5-coder-14b-llm-sft-peft

MODEL_NAME="unsloth/Qwen2.5-7B"
TRAIN_FILE="data/train.json"
VALID_FILE="data/valid.json"
OUTPUT_DIR="outputs/qwen25-7b-unsloth-lora"

mkdir -p $OUTPUT_DIR

# Hyperparameters
BATCH_SIZE=1
GRAD_ACCUM=2
NUM_EPOCHS=3
LEARNING_RATE=2e-4
MAX_SEQ_LENGTH=8192

# LoRA config
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.0

echo "Training with Unsloth..."
echo "  Model: $MODEL_NAME"
echo "  Max seq length: $MAX_SEQ_LENGTH"
echo "  Batch size: $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
echo ""

python exp_scripts/training/train_qwen_unsloth.py \
    --model_name $MODEL_NAME \
    --train_file $TRAIN_FILE \
    --validation_file $VALID_FILE \
    --output_dir $OUTPUT_DIR \
    --max_seq_length $MAX_SEQ_LENGTH \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --warmup_steps 10 \
    --logging_steps 10 \
    --save_steps 100 \
    "$@"