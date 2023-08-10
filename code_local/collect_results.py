import fnmatch
import os
import json
import pandas as pd

original_path = "G:\\Meine Ablage\\Masterarbeit\\fantastic-umbrella\\finished_runs\\04_mod_runs"
list_of_runs = []
for path,dirs,files in os.walk(original_path):
    for file in fnmatch.filter(files,'run_overview_extended.json'):
        file_path = os.path.abspath(os.path.join(path,file))
        print(f'Found file at: {file_path}\n')
        with open(file_path) as f:
            d = json.load(f)
            list_of_runs.append(d)
            break

df = pd.DataFrame(list_of_runs)
df["output_dir"] = df["output_dir"].apply(lambda x: str(x).replace('\r',''))

summary_path = original_path + "\\run_summary.csv"
print(f'Saving dataframe to {summary_path}\n')
df.to_csv(summary_path)
# df.to_pickle("/home/lange/fantastic-umbrella/runs/run_summary.pickle")

summary_small_path = original_path + "\\run_summary_small.csv"
print(f'Saving dataframe to {summary_small_path}\n')
columns_to_keep = [c for c in list(df) if len(df[c].unique()) > 1]
df[columns_to_keep].to_csv(summary_small_path)

