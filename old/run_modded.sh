declare -a StringArray=("cola" "mrpc"  "rte" "stsb" "wnli")
# done:
# small: 
# small: "sst2" "qnli"
# to big: "mnli" "qqp"
for task_name in "${StringArray[@]}";
do
   python run_glue_no_trainer_modded.py --model_name_or_path bert-base-cased \
										--task_name $task_name \
										--max_length 128 \
										--per_device_train_batch_size 32 \
										--learning_rate 2e-5 \
										--num_train_epochs 15  \
										--output_dir /content/mod_runs/$task_name \
										--seed 0 \
										--hidden_dropout 0.8 \
										--with_tracking \
										--report_to tensorboard
done