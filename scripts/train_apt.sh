#!/bin/bash

set -u
set -o pipefail

DATA=./DATA
TRAINER=APT

DATASETS=(
eurosat
)

SEEDS=(
1
)

CFG=${1:-vit_b32}
CTP=end
NCTX=${2:-16}
SHOTS=${3:-16}
CSC=False

EPS=${4:-4}
ALPHA=${5:-0.67}
STEPS=${6:-3}
ATP=onfly # onfly or perturbed
PALPHA=0.001


ROB_TEST=True

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0

FAILED_JOBS=()

for DATASET in "${DATASETS[@]}"; do
    DATASET_CONFIG="configs/datasets/${DATASET}.yaml"
    TRAINER_CONFIG="configs/trainers/${TRAINER}/${CFG}.yaml"

    if [ ! -f "${DATASET_CONFIG}" ]; then
        echo "Dataset config does not exist: ${DATASET_CONFIG}"

        for SEED in "${SEEDS[@]}"; do
            FAILED_JOBS+=("${DATASET}/seed${SEED}")
        done

        continue
    fi

    if [ ! -f "${TRAINER_CONFIG}" ]; then
        echo "Trainer config does not exist: ${TRAINER_CONFIG}"
        exit 1
    fi

    for SEED in "${SEEDS[@]}"; do
        export PYTHONHASHSEED="${SEED}"

        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

        EXPERIMENT_ROOT="output/${TRAINER}/${CFG}_${SHOTS}shots/eps${EPS}_alpha${ALPHA}_step${STEPS}"

        DIR="${EXPERIMENT_ROOT}/${DATASET}/seed${SEED}/${TIMESTAMP}"

        echo
        echo "============================================================"
        echo "Dataset: ${DATASET}"
        echo "Seed:    ${SEED}"
        echo "Shots:   ${SHOTS}"
        echo "Output:  ${DIR}"
        echo "============================================================"

        if [ -d "${DIR}" ]; then
            echo "Result directory already exists, skipping: ${DIR}"
            continue
        fi

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
            TRAINER.APT.PROMPT ${ATP} \
            TRAINER.APT.PALPHA ${PALPHA} \
        
        then
            echo "Completed: ${DATASET}, seed=${SEED}"
        else
            echo "Failed: ${DATASET}, seed=${SEED}"
            FAILED_JOBS+=("${DATASET}/seed${SEED}")
        fi
    done
done

echo
echo "============================================================"
echo "All dataset and seed jobs have finished."

if [ ${#FAILED_JOBS[@]} -eq 0 ]; then
    echo "All jobs completed successfully."
else
    echo "Failed jobs:"
    printf "  - %s\n" "${FAILED_JOBS[@]}"
    exit 1
fi
