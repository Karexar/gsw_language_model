# This scripts performs two split
# The first group the labelled dataset per dialect and split into train, valid,
# and test set.
# The second does the same, but on the predictions from the entire twitter
# module.

###  Settings  #################################################################
dataset_path = "data/whatsapp/labelled_predicted.csv"
label_names = "BE,CE,EA,GR,NW,VS,ZH".split(',')
out_dir = "data/whatsapp/predicted_dialect"
################################################################################

import pandas as pd
from preprocessing.split_dataset import *
import os
from pathlib import Path

df = pd.read_csv(dataset_path, sep="\t", header=None)

#---  Group per predicted dialect  ---------------------------------------------

Path(out_dir).mkdir(parents=True, exist_ok=True)
for dialect in label_names:
    print(dialect)
    sub = df[df.iloc[:, 2] == dialect]
    #sets = split_dataset(sub, 0.8, 0.1)

    dialect_dir = os.path.join(out_dir, dialect)
    Path(dialect_dir).mkdir(parents=True, exist_ok=True)

    path = os.path.join(dialect_dir, "test.csv")
    cur_set = pd.DataFrame(sub.iloc[:, 0])
    cur_set.to_csv(path, sep="\t", header=None, index=False)
