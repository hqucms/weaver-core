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
    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $(which weaver) --backend nccl"
else
    CMD="weaver"
fi

epochs=50
samples_per_epoch=$((10000 * 1024 / $NGPUS))
samples_per_epoch_val=$((10000 * 128))
dataopts="--num-workers 4 --fetch-step 0.01"

# PN, PFN, PCNN, ParT
model=$1
[[ -z ${model} ]] && model="ParT"
if [[ "$model" == "ParT" ]]; then
    modelopts="networks/CMSAK4_ParT.py --use-amp"
    batchopts="--batch-size 512 --start-lr 1e-3"
elif [[ "$model" == "PN" ]]; then
    modelopts="networks/CMSAK4_PN.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "PNXT" ]]; then
    modelopts="networks/CMSAK4_PNXT.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
else
    echo "Invalid model $model!"
    exit 1
fi

$CMD \
    --data-train \
    "BulkGravitonToHHTo4Q_part1:${DATADIR}/BulkGravitonToHHTo4Q_MX-600to6000_MH-15to250_part1_TuneCP5_13TeV-madgraph_pythia8/*/*/*/output_*.root" \
    "BulkGravitonToHHTo4Q_part2:${DATADIR}/BulkGravitonToHHTo4Q_MX-600to6000_MH-15to250_part2_TuneCP5_13TeV-madgraph_pythia8/*/*/*/output_*.root" \
    "BulkGravitonToHHTo4Q_part3:${DATADIR}/BulkGravitonToHHTo4Q_MX-600to6000_MH-15to250_part3_TuneCP5_13TeV-madgraph_pythia8/*/*/*/output_*.root" \
    "TTToSemiLeptonic_mtop171p5:${DATADIR}/TTToSemiLeptonic_mtop171p5_TuneCP5_13TeV-powheg-pythia8/*/*/*/output_*.root" \
    "TTToSemiLeptonic_mtop173p5:${DATADIR}/TTToSemiLeptonic_mtop173p5_TuneCP5_13TeV-powheg-pythia8/*/*/*/output_*.root" \
    "QCD_Pt_30to50_TuneCP5_13TeV_pythia8:${DATADIR}/QCD_Pt_30to50_TuneCP5_13TeV_pythia8/*/*/*/output_*.root" \
    "QCD_Pt_50to80_TuneCP5_13TeV_pythia8:${DATADIR}/QCD_Pt_50to80_TuneCP5_13TeV_pythia8/*/*/*/output_*.root" \
    "QCD_Pt_80to120_TuneCP5_13TeV_pythia8:${DATADIR}/QCD_Pt_80to120_TuneCP5_13TeV_pythia8/*/*/*/output_*.root" \
    "QCD_Pt_120to170_TuneCP5_13TeV_pythia8:${DATADIR}/QCD_Pt_120to170_TuneCP5_13TeV_pythia8/*/*/*/output_*.root" \
    "QCD_Pt_170to300_TuneCP5_13TeV_pythia8:${DATADIR}/QCD_Pt_170to300_TuneCP5_13TeV_pythia8/*/*/*/output_*.root" \
    "QCD_Pt_300to470_TuneCP5_13TeV_pythia8:${DATADIR}/QCD_Pt_300to470_TuneCP5_13TeV_pythia8/*/*/*/output_*.root" \
    "QCD_Pt_470to600_TuneCP5_13TeV_pythia8:${DATADIR}/QCD_Pt_470to600_TuneCP5_13TeV_pythia8/*/*/*/output_*.root" \
    "QCD_Pt_600to800_TuneCP5_13TeV_pythia8:${DATADIR}/QCD_Pt_600to800_TuneCP5_13TeV_pythia8/*/*/*/output_*.root" \
    "QCD_Pt_800to1000_TuneCP5_13TeV_pythia8:${DATADIR}/QCD_Pt_800to1000_TuneCP5_13TeV_pythia8/*/*/*/output_*.root" \
    "QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8:${DATADIR}/QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8/*/*/*/output_*.root" \
    --data-config data/CMSAK4_${model}.yaml --network-config $modelopts \
    --model-prefix training/CMSAK4/${model}/{auto}${suffix}/net \
    $dataopts $batchopts \
    --samples-per-epoch ${samples_per_epoch} --samples-per-epoch-val ${samples_per_epoch_val} --num-epochs $epochs --gpus 0 \
    --optimizer ranger --log logs/CMSAK4_${model}_{auto}${suffix}.log --predict-output pred.root \
    --tensorboard CMSAK4_${model}${suffix} \
    "${@:2}"
