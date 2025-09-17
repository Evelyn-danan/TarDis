#!/bin/bash
echo "Start time: `date`"
echo "Script PID: $$"

if test $# != 3
then
    echo "Usage: bash para_search.sh [model] [gpu] [data_fold]"
    exit 1
fi

basic_fun="./basic_functions.sh"
source ${basic_fun}

#below is parameter
model_name=$1
gpu=$2
data_fold=$3
epoch=500
checkpoint_freq=60
save_emb=1
dir_name="baseline"
batch_size=10000 #10000 5000 1000
kge_emb=100 #100 200 300
lr=0.001 #0.1 0.01 0.001 0.0001 0.00001

if test "${model_name}" = "CompGCN"
then
    kge_emb=20
elif test "${model_name}" = "RESCAL"
then
    kge_emb=60
elif test "${model_name}" = "ConvE"
then
    kge_emb=80
fi

python -u train_pykeen.py --model ${model_name} --gpu ${gpu} --fold ${data_fold} --epochs_kge ${epoch} --batch_size_kge ${batch_size} \
    --emb_dim_kge ${kge_emb} --lr_kge ${lr} --save_emb ${save_emb} --dir_name ${dir_name} --checkpoint_freq ${checkpoint_freq}

echo
echo "End time: `date`"
