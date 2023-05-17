declare -a StringArray=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "stsb" "wnli")

for task_name in "${StringArray[@]}";
do
   python run_glue_no_trainer.py --model_name_or_path bert-base-cased --task_name $task_name --max_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3  --output_dir /content/mod_runs/$task_name --seed 0
done