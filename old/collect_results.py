import os
import fnmatch
import json
import pandas as pd

task_names = []
modded_list = []
num_train_epochs_list = []
hidden_dropout_list = []
eval_metrics = []
eval_scores = []



for path,dirs,files in os.walk('.\\mod_runs'):
    for file in fnmatch.filter(files,'used_arguments.json'):
        # fullname = os.path.abspath(os.path.join(path,file))
        task_name = os.path.basename(path)
        with open(os.path.abspath(os.path.join(path,file))) as f:
            d = json.load(f)
            modded = d['modded']
            num_train_epochs = d['num_train_epochs']
            hidden_dropout = d['hidden_dropout']
                       


    for file in fnmatch.filter(files,'all_results.json'):
        fullname = os.path.abspath(os.path.join(path,file))
        task_name = os.path.basename(path)
        with open(fullname) as f:
            d = json.load(f)
            for key,value in d.items():
                task_names.append(task_name)
                modded_list.append(modded)
                hidden_dropout_list.append(hidden_dropout)
                num_train_epochs_list.append(num_train_epochs = d['num_train_epochs'])
                eval_metrics.append(key)
                eval_scores.append(value)






    
df_mod = pd.DataFrame({'task' : task_names,
                       'modded' : modded_list,
                       'num_train_epochs' : num_train_epochs_list,
                       'hidden_dropout' : hidden_dropout_list,
                       'metric' : eval_metrics,
                       'score_mod' : eval_scores})#.set_index(['task','metric'])

task_names = []
eval_metrics = []
eval_scores = []

for path,dirs,files in os.walk('.\\vanilla_runs'):
    for file in fnmatch.filter(files,'all_results.json'):
        fullname = os.path.abspath(os.path.join(path,file))
        task_name = os.path.basename(path)
        with open(fullname) as f:
            d = json.load(f)
            for key,value in d.items():
                task_names.append(task_name)
                eval_metrics.append(key)
                eval_scores.append(value)

df_vanilla = pd.DataFrame({'task' : task_names,
                           'metric' : eval_metrics,
                           'score_vanilla' : eval_scores}).set_index(['task','metric'])

df_joined = df_mod.join(df_vanilla)
df_joined['diff'] = df_joined['score_mod']-df_joined['score_vanilla']
print(df_joined)
