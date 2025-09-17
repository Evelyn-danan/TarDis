#!/bin/bash

echo "Start time: `date`"
echo "Script PID: $$"
echo

if test $# != 2
then
    echo "Usage: bash para_search.sh [mlp_mode] [gpu]"
    exit 1
fi

basic_fun="./basic_functions.sh"
source ${basic_fun}

# below is the best hyperparameters for the direct mode
curr_mode=$1 #test merged mergedAtt direct
gpu=$2
epoch=1000
batch_size=10000
kge_emb=20
lr=0.0001 #0.0001 0.00005 0.00001
dropout=0.3 #0.2 0.3 0.4
feat_emb=100 #100 200 300
hidden_dim=1024-1024-1024 #'512-512' '1024-1024' '512-512-512' '1024-1024-1024'
num_head_layer=2 #2 3 4
save_emb=1
dir_name="mlp_ablation_${curr_mode}"

log_path="./run_logs/mlp_ablation/${curr_mode}"
create_dir ${log_path}
# timestamp=$(date '+%Y%m%d_%H%M%S')

if test "${curr_mode}" = "merged"; then
    lr=0.0001
    dropout=0.3
    feat_emb=100
    batch_size=10000
elif test "${curr_mode}" = "mergedAtt"; then
    lr=0.0001
    dropout=0.3
    feat_emb=200
    batch_size=10000
elif test "${curr_mode}" = "direct"; then
    lr=0.00001
    dropout=0.2
    feat_emb=200
    batch_size=10000
fi

mlp_modes=("merged" "onlyKGE" "onlyFeat" "direct" "mergedAtt" "empty")
for mlp_mode in "${mlp_modes[@]}"
do
    echo "$(date). Running ${mlp_mode}"
    python -u train_mlp_run.py --gpu ${gpu} --epochs_mlp ${epoch} --batch_size_mlp ${batch_size} --emb_dim_kge ${kge_emb} --emb_dim_feat ${feat_emb} \
        --lr_mlp ${lr} --dropout_mlp ${dropout} --hidden_dims_mlp ${hidden_dim} --num_heads_layers ${num_head_layer} --mlp_mode ${mlp_mode} \
        --save_emb ${save_emb} --dir_name ${dir_name} > ${log_path}/MLP_${mlp_mode}.log 2>&1 &
    
    echo "Training PID: $!"
    sleep 1
    echo
done
wait

echo "End time: `date`"