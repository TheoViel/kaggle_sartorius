cd ../src

log_folder=$(python get_log_folder.py 2>&1)
printf "Logging results to $log_folder \n\n"

python main_training.py --fold 0 --log_folder ${log_folder} --lr 1e-4

log_folder=$(python get_log_folder.py 2>&1)
printf "Logging results to $log_folder \n\n"

python main_training.py --fold 0 --log_folder ${log_folder} --lr 2e-4  --epochs 40