#!/bin/bash
echo "Start time: `date`"
echo "Script PID: $$"

if test $# != 2
then
    echo "Usage: bash para_search.sh [model] [gpu]"
    exit 1
fi

basic_fun="./basic_functions.sh"
source ${basic_fun}

#below is parameter
model_name=$1
gpu=$2
epoch=60
checkpoint_freq=0
save_emb=0
dir_name="kge_hpo"
batch_sizes=(10000) #10000 5000 1000
kge_embs=(100) #100 200 300
lrs=(0.01 0.001 0.0001) #0.1 0.01 0.001

log_path="./run_logs/kge_hpo/${model_name}_hpo"
create_dir ${log_path}

if test "${model_name}" = "CompGCN"
then
    kge_embs=(20)
elif test "${model_name}" = "RESCAL"
then
    kge_embs=(60)
fi

echo "${model_name}-KGE model start training on gpu ${gpu}!"
echo

n=0
for batch_size in "${batch_sizes[@]}"
do
    for kge_emb in "${kge_embs[@]}"
    do
        for lr in "${lrs[@]}"
        do
            train_command="python -u train_pykeen_hpo.py --model ${model_name} --gpu ${gpu} --epochs_kge ${epoch} --batch_size_kge ${batch_size} \
                           --emb_dim_kge ${kge_emb} --lr_kge ${lr} --save_emb ${save_emb} --dir_name ${dir_name}"

            echo -e "$(date). Start training ${model_name} with ${n}th parameters"
            echo -e "Params: Model=${model_name},LR=${lr},KGE_Dim=${kge_emb},Batch_size=${batch_size}"

            echo -e "Params: LR=${lr},KGE_Dim=${kge_emb},Batch_size=${batch_size}" > ${log_path}/${model_name}_${n}.log
            ${train_command} >> ${log_path}/${model_name}_${n}.log 2>&1 & #time
            TRAIN_PID=$!
            echo "Training PID: ${TRAIN_PID}"

            wait ${TRAIN_PID}
            
            # 检测训练命令是否成功
            if [ $? -eq 0 ]; then
                echo "$(date). Training completed successfully."
            else
                echo "$(date). Training failed!"
            fi

            n=`expr ${n} + 1`
            echo
        done
    done
done

echo "LR,KGE_Dim,Batch_size,Hits@1,Hits@3,Hits@10,MRR,MR,AUC_ROC,AUC_PR,ACC" > ${log_path}/results_summary.csv
for log in ${log_path}/${model_name}*.log
do
    params=$(grep "^Params:" $log)
    hits1=$(grep "^Hits@1=" $log)
    hits3=$(grep "^Hits@3=" $log)
    hits10=$(grep "^Hits@10=" $log)
    mrr=$(grep "^MRR=" $log)
    mr=$(grep "^MR=" $log)
    auc_roc=$(grep "^AUC_ROC=" $log)
    auc_pr=$(grep "^AUC_PR=" $log)
    acc=$(grep "^ACC=" $log)  # 假设日志中最后有关键指标
    echo "${params},${hits1},${hits3},${hits10},${mrr},${mr},${auc_roc},${auc_pr},${acc}" >> ${log_path}/results_summary.csv
done

echo
echo "End time: `date`"
