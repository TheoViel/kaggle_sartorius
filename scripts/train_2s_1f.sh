cd ../src

log_folder=$(python get_log_folder.py 2>&1)

printf "[Training with reduce_stride=False] \n\n"
printf "Logging results to $log_folder \n\n"

python main_training.py --fold 0 --log_folder ${log_folder}

log_folder_2=$(python get_log_folder.py 2>&1)

printf "\n[Training with reduce_stride=True] \n\n"
printf "Logging results to $log_folder_2 \n\n"

python main_training.py --fold 0 --log_folder ${log_folder_2} --reduce_stride True  --pretrained_folder ${log_folder}