#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

set -e

TRAIN_SCRIPT="train_vit_gradact_paired_complete.py"
TEST_SCRIPT="test_vit_gradact_paired_complete.py"
MODEL_MODULE="train_vit_gradact_paired_complete"
EXP_DIR="run_files_GradActPaired_final"
DATASET_ROOT_BASE="dataset"
RUN_TEST=1
TEST_CHECKPOINT="checkpoint_best.pth"

methods=(
  "GradActPaired"
)

declare -A dataset_classes
dataset_classes=(
  ["DermaMNIST"]=7
  ["BTMRI"]=4
  ["COVID_19"]=4
  ["RETINA"]=4
  ["Kvasir"]=8
  ["CHMNIST"]=8
  ["KneeXray"]=5
)

datasets=(
"DermaMNIST"
  "BTMRI"
  "COVID_19"
  "RETINA"
  "Kvasir"
  "CHMNIST"
  "KneeXray"
)

BATCH_SIZE=32
TEST_BATCH_SIZE=32
EPOCHS=100
LR=1e-4
CLASSIFIER_LR=1e-4
WARMUP_LR=1e-4
WARMUP_EPOCHS=1
WEIGHT_DECAY=0.01
SEED=42
NUM_WORKERS=8
TEST_NUM_WORKERS=8
EARLY_STOPPING_PATIENCE=50

RANK_PER_BLOCK=64
TOTAL_BUDGET=0
MIN_RANK_PER_BLOCK=16
MAX_RANK_PER_BLOCK=128
CALIBRATION_BATCHES=200
NORMALIZE_MODE="mean"
TOKEN_MODE="all"
TARGET_BLOCKS="all"

CLASS_BALANCED_GRADIENT=0

UPDATE_SELECTED_BIAS=0

PRETRAINED_PATH=""

if [ ! -f "${TRAIN_SCRIPT}" ]; then
  echo "❌ Training script not found: ${TRAIN_SCRIPT}"
  exit 1
fi

if [ "${RUN_TEST}" -eq 1 ] && [ ! -f "${TEST_SCRIPT}" ]; then
  echo "❌ Test script not found: ${TEST_SCRIPT}"
  exit 1
fi

for method in "${methods[@]}"; do
  for dataset in "${datasets[@]}"; do
    num_classes="${dataset_classes[$dataset]}"
    dataset_root="${DATASET_ROOT_BASE}/${dataset}"

    if [ -z "${num_classes}" ]; then
      echo "❌ Missing num_classes for dataset: ${dataset}"
      exit 1
    fi

    if [ ! -d "${dataset_root}/train" ]; then
      echo "❌ Training directory not found: ${dataset_root}/train"
      exit 1
    fi

    if [ ! -d "${dataset_root}/val" ]; then
      echo "❌ Validation directory not found: ${dataset_root}/val"
      exit 1
    fi

    echo "============================================================"
    echo "Training Experiment"
    echo "  Method                    : ${method}"
    echo "  Dataset                   : ${dataset}"
    echo "  Dataset root              : ${dataset_root}"
    echo "  Num classes               : ${num_classes}"
    echo "  Main epochs               : ${EPOCHS}"
    echo "  Warm-up epochs            : ${WARMUP_EPOCHS}"
    echo "  Warm-up LR                : ${WARMUP_LR}"
    echo "  Path LR                   : ${LR}"
    echo "  Classifier LR             : ${CLASSIFIER_LR}"
    echo "  Rank per block            : ${RANK_PER_BLOCK}"
    echo "  Total budget              : ${TOTAL_BUDGET}"
    echo "  Calibration batches       : ${CALIBRATION_BATCHES}"
    echo "  Token mode                : ${TOKEN_MODE}"
    echo "  Class-balanced gradient   : ${CLASS_BALANCED_GRADIENT}"
    echo "  Update selected fc1 bias  : ${UPDATE_SELECTED_BIAS}"
    echo "============================================================"

    python "${TRAIN_SCRIPT}" \
      --method "${method}" \
      --dataset "${dataset}" \
      --dataset_root "${dataset_root}" \
      --num_classes "${num_classes}" \
      --batch_size "${BATCH_SIZE}" \
      --epochs "${EPOCHS}" \
      --lr "${LR}" \
      --classifier_lr "${CLASSIFIER_LR}" \
      --warmup_lr "${WARMUP_LR}" \
      --warmup_epochs "${WARMUP_EPOCHS}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --seed "${SEED}" \
      --num_workers "${NUM_WORKERS}" \
      --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
      --rank_per_block "${RANK_PER_BLOCK}" \
      --total_budget "${TOTAL_BUDGET}" \
      --min_rank_per_block "${MIN_RANK_PER_BLOCK}" \
      --max_rank_per_block "${MAX_RANK_PER_BLOCK}" \
      --calibration_batches "${CALIBRATION_BATCHES}" \
      --normalize_mode "${NORMALIZE_MODE}" \
      --token_mode "${TOKEN_MODE}" \
      --target_blocks "${TARGET_BLOCKS}" \
      --class_balanced_gradient "${CLASS_BALANCED_GRADIENT}" \
      --update_selected_bias "${UPDATE_SELECTED_BIAS}" \
      --pretrained_path "${PRETRAINED_PATH}"

    echo "✅ Training finished: ${method} on ${dataset}"

    if [ "${RUN_TEST}" -eq 1 ]; then
      if [ ! -d "${dataset_root}/test" ]; then
        echo "❌ Test directory not found: ${dataset_root}/test"
        exit 1
      fi

      echo "------------------------------------------------------------"
      echo "Testing Experiment"
      echo "  Method       : ${method}"
      echo "  Dataset      : ${dataset}"
      echo "  Checkpoint   : ${TEST_CHECKPOINT}"
      echo "------------------------------------------------------------"

      python "${TEST_SCRIPT}" \
        --method "${method}" \
        --dataset "${dataset}" \
        --dataset_root "${dataset_root}" \
        --exp_dir "${EXP_DIR}" \
        --checkpoint "${TEST_CHECKPOINT}" \
        --model_module "${MODEL_MODULE}" \
        --batch_size "${TEST_BATCH_SIZE}" \
        --num_workers "${TEST_NUM_WORKERS}"

      echo "✅ Testing finished: ${method} on ${dataset}"
    fi
  done
done

echo "✅ All training and testing experiments finished!"