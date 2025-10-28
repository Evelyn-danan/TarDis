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

#below is parameter
mlp_mode=$1 #merged mergedAtt direct
gpu=$2
epoch=300
batch_size=10000 #(10000 50000)
early_stop_patience=5
save_emb=0
dir_name="mlp_hpo"

kge_emb=20
lrs=(0.0001 0.00005 0.00001) #0.0001 0.00005 0.00001
dropouts=(0.2 0.3 0.4) #0.2 0.3 0.4
feat_embs=(100 200 300) #100 200 300
hidden_dims=('1024-1024-1024') #'512-512' '1024-1024' '512-512-512' '1024-1024-1024'
num_heads_layers=(2) #2 3 4

model_name="MLP_${mlp_mode}"
log_path="./run_logs/mlp_hpo/${model_name}_hpo"
create_dir ${log_path}

if test ${mlp_mode} == "direct"
then
    feat_embs=(200)
fi

echo "${model_name} model start training on gpu ${gpu}!"
echo

max_jobs=5  # 最大并行任务数
current_jobs=0

mark=1
n=0
for num_head_layer in "${num_heads_layers[@]}"
do
    for hidden_dim in "${hidden_dims[@]}"
    do
        for feat_emb in "${feat_embs[@]}"
        do
            for dropout in "${dropouts[@]}"
            do
                for lr in "${lrs[@]}"
                do
                    train_command="python -u train_mlp.py --gpu ${gpu} --epochs_mlp ${epoch} --batch_size_mlp ${batch_size} --emb_dim_kge ${kge_emb} \
                                    --emb_dim_feat ${feat_emb} --lr_mlp ${lr} --dropout_mlp ${dropout} --hidden_dims_mlp ${hidden_dim} \
                                    --num_heads_layers ${num_head_layer} --mlp_mode ${mlp_mode} --save_emb ${save_emb} --dir_name ${dir_name} \
                                    --early_stop_patience ${early_stop_patience}"

                    echo -e "$(date). Start training with ${n}th parameters"
                    echo -e "Params: LR=${lr},Dropout=${dropout},FeatDim=${feat_emb},HiddenDims=${hidden_dim},Heads=${num_head_layer}" > ${log_path}/${model_name}${mark}_${n}.log
                    ${train_command} >> ${log_path}/${model_name}${mark}_${n}.log 2>&1 & #time 
                    echo "Current PID: $!"
                    sleep 1m

                    current_jobs=$((current_jobs + 1))
                    if [ "$current_jobs" -ge "$max_jobs" ]; then
                        wait  # 等待当前任务完成
                        current_jobs=0
                    fi

                    n=`expr ${n} + 1`
                    # echo "Single train finished!!!"
                    echo
                done
            done
        done
    done
done
wait

echo "LR,Dropout,FeatDim,HiddenDims,Heads,AUC_ROC,AUC_PR,ACC" > ${log_path}/results${mark}_summary.csv
for log in ${log_path}/${model_name}${mark}*.log; do
    params=$(grep "^Params:" $log)
    metrics=$(grep "^Best model" $log)  # 假设日志中最后有关键指标
    echo "${params},${metrics}" >> ${log_path}/results${mark}_summary.csv
done


echo
echo "End time: `date`"
