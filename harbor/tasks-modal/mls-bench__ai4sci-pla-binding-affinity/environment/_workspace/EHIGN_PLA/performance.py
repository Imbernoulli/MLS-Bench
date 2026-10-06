# %%
from glob import glob
import os
import pandas as pd
from collections import defaultdict

# %%
models = ['EHIGN']
model_root = './model'
model_performance_dict = defaultdict(list)

results_dict = defaultdict(list)
for model_name in models:
    model_dir = glob(os.path.join(model_root, f'*_{model_name}_repeat*'))
    best_dict = defaultdict(list)
    for md in model_dir:
        log_path = os.path.join(md, 'log', 'train', 'Train.log')

        with open(log_path, 'r') as f:
            logs = f.readlines()
            for log in logs:
                if 'test2013_pr' in log:
                    messages = log.split(', ')
                    for msg in messages:
                        key, val = msg.split('-')[-2].rstrip().replace(',', ''), msg.split('-')[-1].rstrip().replace(',', '')
                        val = float(val)
                        results_dict[key].append(val)

model_df = pd.DataFrame(results_dict)
performance_df = model_df.describe()
for indicator in ['test2013_rmse', 'test2013_pr', 'test2016_rmse', 'test2016_pr', 'test2019_rmse', 'test2019_pr']:
    res = performance_df[indicator]
    print("%s: %.3f (%.3f)" % (indicator, res.iloc[1], res.iloc[2]))

# %%