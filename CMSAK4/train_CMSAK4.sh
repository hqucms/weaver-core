#!/bin/bash

set -x

echo "args: $@"

# set the dataset dir via `DATADIR_CMSAK4`
DATADIR=${DATADIR_CMSAK4}
[[ -z $DATADIR ]] && DATADIR='./datasets/CMSAK4'

# set a comment via `COMMENT`
suffix=${COMMENT}


# set the number of gpus for DDP training via `DDP_NGPUS`
NGPUS=${DDP_NGPUS}
[[ -z $NGPUS ]] && NGPUS=1
if ((NGPUS > 1)); then
    #CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $(which weaver) --backend nccl"
    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS ../train.py --backend nccl"


else
    #CMD="weaver"
    CMD="python ../train.py"
fi

epochs=20
samples_per_epoch=50000000 #$((10000 * 1024 / $NGPUS))
samples_per_epoch_val=1000000 #$((10000 * 128))
dataopts="--num-workers 4 --fetch-step 0.01"

# PN, PFN, PCNN, ParT
model=$1
[[ -z ${model} ]] && model="ParT"
if [[ "$model" == "ParT" ]]; then
    modelopts="networks/CMSAK4_ParT.py --use-amp"
    batchopts="--batch-size 512 --start-lr 1e-3"
elif [[ "$model" == "ParT_ef" ]]; then
    modelopts="networks/CMSAK4_ParT_ef.py --use-amp"
    batchopts="--batch-size 512 --start-lr 1e-3"
elif [[ "$model" == "PN" ]]; then
    modelopts="networks/CMSAK4_PN.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "PN_ef" ]]; then
    modelopts="networks/CMSAK4_PN_ef.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "PNXT" ]]; then
    modelopts="networks/CMSAK4_PNXT.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "PNXT_ef" ]]; then
    modelopts="networks/CMSAK4_PNXT_ef.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "PNXT_ef_aux_clas" ]]; then
    modelopts="networks/CMSAK4_PNXT_ef.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "PNXT_ef_aux_regr" ]]; then
    modelopts="networks/CMSAK4_PNXT_ef.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "PNXT_ef_aux_bin" ]]; then
    modelopts="networks/CMSAK4_PNXT_ef.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "PNXT_ef_aux" ]]; then
    modelopts="networks/CMSAK4_PNXT_ef.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "PNXT_ef_aux_tot" ]]; then
    modelopts="networks/CMSAK4_PNXT_ef.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
else
    echo "Invalid model $model!"
    exit 1
fi

suffix_specs=$2

$CMD \
    --data-train \
    "ttjets:${CINECA_SCRATCH}/output_big/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/output_10Mevents_*.root" \
    "qcd:${CINECA_SCRATCH}/output_big/QCD_Pt-15to7000_TuneCP5_Flat2018_13TeV_pythia8/output_10Mevents_*.root" \
    --data-val \
    ${CINECA_SCRATCH}/output_big/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8/output_10Mevents_*.root \
    --data-config data/CMSAK4_${model}.yaml --network-config $modelopts \
    --model-prefix training/CMSAK4/${model}/{auto}${suffix}_${suffix_specs}/net \
    $dataopts $batchopts \
    --samples-per-epoch ${samples_per_epoch} --samples-per-epoch-val ${samples_per_epoch_val} \
    --num-epochs $epochs --gpus 0,1,2,3 --no-aux-epoch 6 \
    --optimizer ranger --log logs/{auto}${suffix}_${suffix_specs}.log \
    --tensorboard CMSAK4_${model}${suffix}_${suffix_specs} \
    "${@:3}"
