echo "Start time: `date`"
echo "Script PID: $$"
echo

basic_fun="./basic_functions.sh"
source ${basic_fun}

# python -u pykeen_predict.py --model "CompGCN" --gpu 0 > ./run_logs/baseline/evaluate_GD/CompGCN.log 2>&1 &
# python -u pykeen_predict.py --model "TransE" --gpu 1 > ./run_logs/baseline/evaluate_GD/TransE.log 2>&1 &
# python -u pykeen_predict.py --model "TransR" --gpu 2 > ./run_logs/baseline/evaluate_GD/TransR.log 2>&1 &
# python -u pykeen_predict.py --model "RotatE" --gpu 3 > ./run_logs/baseline/evaluate_GD/RotatE.log 2>&1 &
# wait

# python -u pykeen_predict.py --model "RESCAL" --gpu 0 > ./run_logs/baseline/evaluate_GD/RESCAL.log 2>&1 &
# python -u pykeen_predict.py --model "ComplEx" --gpu 1 > ./run_logs/baseline/evaluate_GD/ComplEx.log 2>&1 &
# python -u pykeen_predict.py --model "DistMult" --gpu 2 > ./run_logs/baseline/evaluate_GD/DistMult.log 2>&1 &
# wait

# python -u pykeen_predict.py --model "ConvE" --gpu 1 > ./run_logs/baseline/evaluate_GD/ConvE.log 2>&1 &
# wait

python -u pykeen_predict.py --model "TransR" --gpu 2 >> ./run_logs/baseline/evaluate_GD/TransR.log 2>&1 &
python -u pykeen_predict.py --model "RESCAL" --gpu 0 >> ./run_logs/baseline/evaluate_GD/RESCAL.log 2>&1 &

python -u pykeen_predict.py --model "ConvE" --gpu 1 >> ./run_logs/baseline/evaluate_GD/ConvE.log 2>&1 &
wait

echo
echo "End time: `date`"