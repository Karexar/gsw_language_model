# This scripts performs two split
# The first group the labelled dataset per dialect and split into train, valid,
# and test set.
# The second does the same, but on the predictions from the entire twitter
# module.

###  Settings  #################################################################
dataset_path = "data/twitter_all_predicted.csv"
label_names = "BE,CE,EA,GR,NW,VS,ZH".split(',')
out_dir = "data/dialect_specific"
################################################################################

import pandas as pd
from preprocessing.split_dataset import *
import os
from pathlib import Path

df = pd.read_csv(dataset_path, sep="\t")

#--- Group per known dialect  --------------------------------------------------

print("*"*80 + "\nCompute known dialects sets\n" + "*"*80)
known_path = os.path.join(out_dir, "known_labels")
Path(known_path).mkdir(parents=True, exist_ok=True)
all_known = dict()
for dialect in label_names:
    print(dialect)
    sub = df[df.dialect == dialect]
    sets = split_dataset(sub, 0.8, 0.1)

    dialect_dir = os.path.join(known_path, dialect)
    Path(dialect_dir).mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(["train", "valid", "test"]):
        print(name)
        path = os.path.join(dialect_dir, name + ".csv")
        cur_set = pd.DataFrame(sets[i].sentence)
        print(cur_set.shape[0])
        cur_set.to_csv(path, sep="\t", header=None, index=False)
        all_known[name] = pd.concat([all_known.get(name, pd.DataFrame()), cur_set])

#---  Group per predicted dialect  ---------------------------------------------

print("*"*80 + "\nCompute predicted dialects sets\n" + "*"*80)
predicted_path = os.path.join(out_dir, "predicted_labels")
Path(predicted_path).mkdir(parents=True, exist_ok=True)
all_pred = dict()
for dialect in label_names:
    print(dialect)
    sub = df[df.dialect_predicted == dialect]
    sets = split_dataset(sub, 0.8, 0.1)

    dialect_dir = os.path.join(predicted_path, dialect)
    Path(dialect_dir).mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(["train", "valid", "test"]):
        print(name)
        path = os.path.join(dialect_dir, name + ".csv")
        cur_set = pd.DataFrame(sets[i].sentence)
        print(cur_set.shape[0])
        cur_set.to_csv(path, sep="\t", header=None, index=False)
        all_pred[name] = pd.concat([all_pred.get(name, pd.DataFrame()), cur_set])

#---  Save the sets with all dialect  ------------------------------------------

# First we need to remove all train or valid sentences that appear in one of the
# two test sets. This is to avoid training from scratch twice, one for known
# dialects and one for predicted dialects. Once we know which method is the
# best, we can remove the other one.

print("*"*80 + "\nSaving sets with all dialects\n" + "*"*80)
test_sentences = pd.concat([all_known["test"], all_pred["test"]])
test_sentences = set(test_sentences.sentence.values)

for name in ["train", "valid", "test"]:
    print(name)
    print("known")
    all_dir = os.path.join(known_path, "all")
    Path(all_dir).mkdir(parents=True, exist_ok=True)
    sub = all_known[name]
    print(sub.shape[0])
    if name == "train" or name == "valid":
        sub = sub[~sub.iloc[:, 0].isin(test_sentences)]
    print(sub.shape[0])
    path = os.path.join(all_dir, name + ".csv")
    sub.to_csv(path, sep="\t", header=None, index=False)

    print("pred")
    all_dir = os.path.join(predicted_path, "all")
    Path(all_dir).mkdir(parents=True, exist_ok=True)
    sub = all_pred[name]
    print(sub.shape[0])
    if name == "train" or name == "valid":
        sub = sub[~sub.iloc[:, 0].isin(test_sentences)]
    print(sub.shape[0])
    path = os.path.join(all_dir, name + ".csv")
    sub.to_csv(path, sep="\t", header=None, index=False)
