echo "Start time: `date`"
echo "Script PID: $$"
echo

basic_fun="./basic_functions.sh"
source ${basic_fun}

# python -u mlp_predict.py --hpo_mode "merged" --mlp_mode "merged" --gpu 0 > ./run_logs/mlp_ablation/merged0215/mlp_predict_merged.log 2>&1
# python -u mlp_predict.py --hpo_mode "merged" --mlp_mode "onlyKGE" --gpu 0 > ./run_logs/mlp_ablation/merged0215/mlp_predict_onlyKGE.log 2>&1
# python -u mlp_predict.py --hpo_mode "merged" --mlp_mode "onlyFeat" --gpu 0 > ./run_logs/mlp_ablation/merged0215/mlp_predict_onlyFeat.log 2>&1
python -u mlp_predict.py --hpo_mode "merged" --mlp_mode "merged" --gpu 0 > ./run_logs/mlp_ablation/merged0213/mlp_predict_merged.log 2>&1
python -u mlp_predict.py --hpo_mode "merged" --mlp_mode "onlyKGE" --gpu 0 > ./run_logs/mlp_ablation/merged0213/mlp_predict_onlyKGE.log 2>&1
python -u mlp_predict.py --hpo_mode "merged" --mlp_mode "onlyFeat" --gpu 0 > ./run_logs/mlp_ablation/merged0213/mlp_predict_onlyFeat.log 2>&1
# wait

echo
echo "End time: `date`"