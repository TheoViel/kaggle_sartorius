cd ../src

log_folder=$(python get_log_folder.py 2>&1)
printf "Logging results to $log_folder \n\n"

for ((fold=0; fold<=4; fold++))
do
    echo 
    python main_training.py --fold ${fold} --log_folder ${log_folder} --name "htc"  --encoder "resnet50"
done

log_folder=$(python get_log_folder.py 2>&1)
printf "Logging results to $log_folder \n\n"

for ((fold=4; fold<=4; fold++))
do
    echo 
    python main_training.py --fold ${fold} --log_folder ${log_folder}
done