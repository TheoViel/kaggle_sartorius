cd src

log_folder=$(python get_log_folder.py 2>&1)

printf "[Training with reduce_stride=False] \n\n"
printf "Logging results to $log_folder \n\n"

for ((fold=0; fold<=4; fold++))
do
    echo 
    python main_training.py --fold ${fold} --log_folder ${log_folder}
done

log_folder_2=$(python get_log_folder.py 2>&1)

printf "\n[Training with reduce_stride=True] \n\n"
printf "Logging results to $log_folder_2 \n\n"

for ((fold=0; fold<=4; fold++))
do
    echo 
    python main_training.py --fold ${fold} --log_folder ${log_folder_2} --reduce_stride True --pretrained_folder ${log_folder}
done