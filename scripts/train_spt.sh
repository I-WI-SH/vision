#!/bin/bash

set -u
set -o pipefail

DATA=./DATA
TRAINER=SPT

DATASETS=(
eurosat
)

SEEDS=(
1
)

TEMPERATURES=(
1.0
)

CFG=${1:-vit_b32}
CTP=end
NCTX=${2:-16}
SHOTS=${3:-16}
CSC=False

EPS=${4:-4}
ALPHA=${5:-0.67}
STEPS=${6:-3}

W1=${7:-10.0}
W2=${8:-20.0}
W3=${9:-0.5}
W4=${10:-0.1}
A=${11:-0.5}

ROB_TEST=True

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0

FAILED_JOBS=()

TRAINER_CONFIG="configs/trainers/${TRAINER}/${CFG}.yaml"

if [ ! -f "${TRAINER_CONFIG}" ]; then
    echo "Trainer config does not exist: ${TRAINER_CONFIG}"
    exit 1
fi

# 开始训练前检查所有数据集配置。
for DATASET in "${DATASETS[@]}"; do
    DATASET_CONFIG="configs/datasets/${DATASET}.yaml"
    if [ ! -f "${DATASET_CONFIG}" ]; then
        echo "Dataset config does not exist: ${DATASET_CONFIG}"
        exit 1
    fi
done

EXPERIMENT_ROOT="output/${TRAINER}/${CFG}_${SHOTS}shots/eps${EPS}_alpha${ALPHA}_step${STEPS}"

# 每次脚本执行使用独立目录。
RUN_ID="$(date +"%Y%m%d_%H%M%S")_$$"

for DATASET in "${DATASETS[@]}"; do
    DATASET_CONFIG="configs/datasets/${DATASET}.yaml"

    for SEED in "${SEEDS[@]}"; do
        export PYTHONHASHSEED="${SEED}"

        for T in "${TEMPERATURES[@]}"; do
            JOB="${DATASET}/seed${SEED}/temperature${T}"
            DIR="${EXPERIMENT_ROOT}/${JOB}/${RUN_ID}"

            echo
            echo "============================================================"
            echo "Dataset:     ${DATASET}"
            echo "Seed:        ${SEED}"
            echo "Temperature: ${T}"
            echo "Shots:       ${SHOTS}"
            echo "Output:      ${DIR}"
            echo "============================================================"

            if python train.py \
                --root "${DATA}" \
                --trainer "${TRAINER}" \
                --dataset_config_file "${DATASET_CONFIG}" \
                --config_file "${TRAINER_CONFIG}" \
                --output_dir "${DIR}" \
                --eps "${EPS}" \
                --alpha "${ALPHA}" \
                --steps "${STEPS}" \
                --seed "${SEED}" \
                TRAINER.COOP.N_CTX "${NCTX}" \
                TRAINER.COOP.CSC "${CSC}" \
                TRAINER.COOP.CLASS_TOKEN_POSITION "${CTP}" \
                TRAINER.COOP.ATK_TEST "${ROB_TEST}" \
                DATASET.NUM_SHOTS "${SHOTS}" \
                TRAINER.SPT.W1 "${W1}" \
                TRAINER.SPT.W2 "${W2}" \
                TRAINER.SPT.W3 "${W3}" \
                TRAINER.SPT.W4 "${W4}" \
                TRAINER.SPT.ALPHA "${A}" \
                TRAINER.SPT.TEMPERATURE "${T}"
            then
                echo "Completed: ${JOB}"
            else
                echo "Failed: ${JOB}"
                FAILED_JOBS+=("${JOB}")
            fi
        done
    done
done

echo
echo "============================================================"
echo "All dataset, seed and temperature jobs have finished."

if [ "${#FAILED_JOBS[@]}" -eq 0 ]; then
    echo "All jobs completed successfully."
else
    echo "Failed jobs:"
    printf "  - %s\n" "${FAILED_JOBS[@]}"
    exit 1
fi