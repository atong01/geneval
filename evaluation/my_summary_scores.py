# Get results of evaluation

import argparse
import os

import numpy as np
import pandas as pd

sample_dirs ={
    "fkc 5.5": "generated_samples",
    "fkc 7.5": "generated_samples_beta_75",
    "fkc 5.5 01 09": "generated_samples_beta_55_01_09",
    "sdxl 7.5": "generated_images_sdxl_cfg7.5",
}
sample_dirs = {k: f"/network/scratch/a/alexander.tong/geneval_results/{v}/results_val.jsonl" for k, v in sample_dirs.items()}

parser = argparse.ArgumentParser()
parser.add_argument("--valset-only", action="store_true")
parser.add_argument("--max-index", type=int, default=1000)
args = parser.parse_args()

# Load classnames

with open(os.path.join(os.path.dirname(__file__), "object_names.txt")) as cls_file:
    classnames = [line.strip() for line in cls_file]
    cls_to_idx = {"_".join(cls.split()):idx for idx, cls in enumerate(classnames)}

# Load results

dfs = []
for runname, filename in sample_dirs.items():
    df = pd.read_json(filename, orient="records", lines=True)
    df["example_index"] = df.filename.apply(lambda x: int(x.split("/")[-3]))
    df = df[df["example_index"] < args.max_index]

    task_df = df[["tag", "correct"]].groupby("tag").mean()
    task_df = pd.concat([task_df.T, task_df.mean().rename({"0":"overall"})], axis=1).T.rename(index={0: 'Overall'})

    order = ["Overall", "two_object", "counting", "colors", "position", "color_attr"]
    task_df = task_df.reindex(order).T
    task_df.index = [runname]
    dfs.append(task_df)
print(pd.concat(dfs).round(2))
exit()

# Measure overall success

print("Summary")
print("=======")
print(f"Total images: {len(df)}")
print(f"Total prompts: {len(df.groupby('metadata'))}")
print(f"% correct images: {df['correct'].mean():.2%}")
print(f"% correct prompts: {df.groupby('metadata')['correct'].any().mean():.2%}")
print()

# By group

task_scores = []

print("Task breakdown")
print("==============")
for tag, task_df in df.groupby('tag', sort=False):
    task_scores.append(task_df['correct'].mean())
    print(f"{tag:<16} = {task_df['correct'].mean():.2%} ({task_df['correct'].sum()} / {len(task_df)})")
print()

print(f"Overall score (avg. over tasks): {np.mean(task_scores):.5f}")
