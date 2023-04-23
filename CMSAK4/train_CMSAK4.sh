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
    CMD="torchrun  --master_port=99 --nnodes=2 --nproc_per_node=$NGPUS --max_restarts=0  --rdzv_id=98 --rdzv_backend=c10d  --rdzv_endpoint=r225n14:29800 ../train.py --backend nccl "


else
    #CMD="weaver"
    CMD="python ../train.py"
fi

epochs=15
samples_per_epoch=50000000 #$((10000 * 1024 / $NGPUS))
samples_per_epoch_val=1000000 #$((10000 * 128))
dataopts="--num-workers 4 --fetch-step 0.01"

model=$1

auxopts=""
if [[ "$model" == *"_saturation"* ]]; then
    model=${model%"_saturation"}
    auxopts="--no-aux-epoch 10 --aux-saturation"
fi

# PN, PFN, PCNN, ParT
[[ -z ${model} ]] && model="ParT"
if [[ "$model" == "ParT" ]]; then
    modelopts="networks/CMSAK4_ParT.py --use-amp"
    batchopts="--batch-size 512 --start-lr 1e-3"
elif [[ "$model" == "PN" ]] || [[ "$model" == "PN_noSV" ]]; then
    modelopts="networks/CMSAK4_PN.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "PNXT" ]] || [[ "$model" == "PNXT_noSV" ]]; then
    modelopts="networks/CMSAK4_PNXT.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "PNXT_lite" ]] || [[ "$model" == "PNXT_noSV_lite" ]]; then
    modelopts="networks/CMSAK4_PNXT_lite.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == *"ef"* ||  "$model" == *"aux"* ]] && [[ "$model" == *"PNXT"* && "$model" != *"lite"* && "$model" != *"new"* ]]; then
    modelopts="networks/CMSAK4_PNXT_ef.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == *"ef"* ||  "$model" == *"aux"* ]] && [[ "$model" == *"PNXT"* && "$model" == *"lite"* && "$model" != *"new"* ]]; then
    modelopts="networks/CMSAK4_PNXT_ef_lite.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == *"ef"* ||  "$model" == *"aux"* ]] && [[ "$model" == *"PNXT"* && "$model" != *"lite"* && "$model" == *"new"* ]]; then
    modelopts="networks/CMSAK4_PNXT_ef_new.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == *"ef"* ||  "$model" == *"aux"* ]] && [[ "$model" == *"PNXT"* && "$model" == *"lite"* && "$model" == *"new"* ]]; then
    modelopts="networks/CMSAK4_PNXT_ef_lite_new.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
else
    echo "Invalid model $model!"
    exit 1
fi

#remove _lite from model name
if [[ "$model" == *"_lite"* ]]; then
    model=${model%"_lite"}
fi
#remove _new from model name
if [[ "$model" == *"_new"* ]]; then
    model=${model%"_new"}
fi


suffix_specs=$2

if [[ "$CINECA_SCRATCH" == "" ]]; then
    store=""
else
    store=${CINECA_SCRATCH}/
fi

if [ $extra = true ]; then
    extra_selection="(np.abs(jet_eta)<1.4) & (jet_pt>30) & (jet_pt<200)"
else
    extra_selection=""
fi
echo "extra selections: ${extra_selection}"

$CMD \
    --data-train \
    "ttjets:${in_dir}/output_big/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/output_10Mevents_*.root" \
    "qcd:${in_dir}/output_big/QCD_Pt-15to7000_TuneCP5_Flat2018_13TeV_pythia8/output_10Mevents_*.root" \
    --data-val \
    ${in_dir}/output_big/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8_val/output_10Mevents_*.root \
    --data-test \
    ${in_dir}/output_big/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8_test/output_10Mevents_*.root \
    --data-config data/CMSAK4_${model}.yaml --network-config $modelopts \
    --model-prefix ${store}training/CMSAK4/${model}/{auto}${suffix}_${suffix_specs}/net \
    $dataopts $batchopts \
    --samples-per-epoch ${samples_per_epoch} --samples-per-epoch-val ${samples_per_epoch_val} \
    --num-epochs $epochs --gpus 0,1,2,3 --no-aux-epoch 6 --epoch-division 3\
    --optimizer ranger --log logs/{auto}${suffix}_${suffix_specs}.log \
    --tensorboard CMSAK4_${model}${suffix}_${suffix_specs} $auxopts \
    --extra-selection "${extra_selection}" --extra-test-selection "${extra_selection}" \
    "${@:3}"
