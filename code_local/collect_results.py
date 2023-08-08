import fnmatch
import os
import json
import pandas as pd

list_of_runs = []
for path,dirs,files in os.walk('/home/lange/fantastic-umbrella/runs'):
    for file in fnmatch.filter(files,'run_overview.json'):
        file_path = os.path.abspath(os.path.join(path,file))
        print(f'Found file at: {file_path}')
        with open(file_path) as f:
            d = json.load(f)
            list_of_runs.append(d)
            break

df = pd.DataFrame(list_of_runs)
df["output_dir"] = df["output_dir"].apply(lambda x: str(x).replace('\r',''))
df.to_csv("/home/lange/fantastic-umbrella/runs/run_summary.csv")
# df.to_pickle("/home/lange/fantastic-umbrella/runs/run_summary.pickle")


columns_to_keep = [c for c in list(df) if len(df[c].unique()) > 1]
df[columns_to_keep].to_csv("/home/lange/fantastic-umbrella/runs/run_summary_small.csv")
