#!/bin/bash

set -x

echo "args: $@"

# set the dataset dir via `DATADIR_JetClass`
DATADIR=${DATADIR_JetClass}
[[ -z $DATADIR ]] && DATADIR='./datasets/JetClass'

# subdirs for train/val
[[ -z ${TRAIN_DIR} ]] && TRAIN_DIR=train_100M_parquet
[[ -z ${VAL_DIR} ]] && VAL_DIR=val_5M_parquet

# set a comment via `COMMENT`
suffix=${COMMENT}

# set the number of gpus for DDP training via `DDP_NGPUS`
NGPUS=${DDP_NGPUS}
[[ -z $NGPUS ]] && NGPUS=1
if ((NGPUS > 1)); then
    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $(which weaver) --backend nccl"
else
    CMD="weaver"
fi

epochs=50
samples_per_epoch=$((10000 * 1024 / $NGPUS))
samples_per_epoch_val=$((10000 * 128 / $NGPUS))
dataopts="--num-workers 10 --data-split-num 10 --fetch-step 0.02 --data-split-val 1"

model=ParT
if [[ "$model" == "ParT" ]]; then
    modelopts="networks/example_ParT.py --use-amp --compile --compile-optimizer"
    batchopts="--batch-size 2048 --start-lr 2e-3"
else
    echo "Invalid model $model!"
    exit 1
fi

# "kin", "kinpid", "full"
FEATURE_TYPE=full

# currently only Pythia
SAMPLE_TYPE=Pythia

$CMD \
    --data-train \
    "HToBB:${DATADIR}/${SAMPLE_TYPE}/${TRAIN_DIR}/HToBB_*.parquet" \
    "HToCC:${DATADIR}/${SAMPLE_TYPE}/${TRAIN_DIR}/HToCC_*.parquet" \
    "HToGG:${DATADIR}/${SAMPLE_TYPE}/${TRAIN_DIR}/HToGG_*.parquet" \
    "HToWW2Q1L:${DATADIR}/${SAMPLE_TYPE}/${TRAIN_DIR}/HToWW2Q1L_*.parquet" \
    "HToWW4Q:${DATADIR}/${SAMPLE_TYPE}/${TRAIN_DIR}/HToWW4Q_*.parquet" \
    "TTBar:${DATADIR}/${SAMPLE_TYPE}/${TRAIN_DIR}/TTBar_*.parquet" \
    "TTBarLep:${DATADIR}/${SAMPLE_TYPE}/${TRAIN_DIR}/TTBarLep_*.parquet" \
    "WToQQ:${DATADIR}/${SAMPLE_TYPE}/${TRAIN_DIR}/WToQQ_*.parquet" \
    "ZToQQ:${DATADIR}/${SAMPLE_TYPE}/${TRAIN_DIR}/ZToQQ_*.parquet" \
    "ZJetsToNuNu:${DATADIR}/${SAMPLE_TYPE}/${TRAIN_DIR}/ZJetsToNuNu_*.parquet" \
    --data-val \
    "HToBB:${DATADIR}/${SAMPLE_TYPE}/${VAL_DIR}/HToBB_*.parquet" \
    "HToCC:${DATADIR}/${SAMPLE_TYPE}/${VAL_DIR}/HToCC_*.parquet" \
    "HToGG:${DATADIR}/${SAMPLE_TYPE}/${VAL_DIR}/HToGG_*.parquet" \
    "HToWW2Q1L:${DATADIR}/${SAMPLE_TYPE}/${VAL_DIR}/HToWW2Q1L_*.parquet" \
    "HToWW4Q:${DATADIR}/${SAMPLE_TYPE}/${VAL_DIR}/HToWW4Q_*.parquet" \
    "TTBar:${DATADIR}/${SAMPLE_TYPE}/${VAL_DIR}/TTBar_*.parquet" \
    "TTBarLep:${DATADIR}/${SAMPLE_TYPE}/${VAL_DIR}/TTBarLep_*.parquet" \
    "WToQQ:${DATADIR}/${SAMPLE_TYPE}/${VAL_DIR}/WToQQ_*.parquet" \
    "ZToQQ:${DATADIR}/${SAMPLE_TYPE}/${VAL_DIR}/ZToQQ_*.parquet" \
    "ZJetsToNuNu:${DATADIR}/${SAMPLE_TYPE}/${VAL_DIR}/ZJetsToNuNu_*.parquet" \
    --data-config data/JetClass_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix training/JetClass/${SAMPLE_TYPE}/${FEATURE_TYPE}/${model}/{auto}${suffix}/net \
    $dataopts $batchopts \
    --samples-per-epoch ${samples_per_epoch} --samples-per-epoch-val ${samples_per_epoch_val} \
    --num-epochs $epochs --gpus 0 \
    --optimizer adamw --lr-scheduler warmup+cos \
    --log-file logs/JetClass_${SAMPLE_TYPE}_${FEATURE_TYPE}_${model}_{auto}${suffix}.log \
    --tensorboard JetClass_${SAMPLE_TYPE}_${FEATURE_TYPE}_${model}${suffix} \
    "${@:1}"
