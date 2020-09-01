from bert_lid import *
import pandas as pd
from tqdm import tqdm
import math


###  Settings  #################################################################
dataset_path = "data/whatsapp.csv"
out_path = "data/whatsapp_predicted.csv"
out_path_filtered = "data/whatsapp_over_99.csv"
prediction_threshold = 0.99
###############################################################################

df = pd.read_csv(dataset_path, sep="\t", header=None)

sentences = list(df.iloc[:, 0].values)

lid = BertLid()

predictions = []
interval = 500
for i in tqdm(range(math.ceil(len(sentences)/interval))):
    left = i*interval
    right = (i+1)*interval
    batch = sentences[left:right]
    if len(batch) > 0:
        predictions.extend(lid.predict_label(batch))
if len(sentences) != len(predictions):
    raise Exception("predictions and sentences_list must have the " +
                    "same length")

df["predictions"] = predictions

df.to_csv(out_path, sep="\t", header=None, index=False)

if out_path_filtered is not None:
    df = df[df.predictions >= prediction_threshold]
    df = df.drop(["predictions"], axis=1)
    df.to_csv(out_path_filtered, sep="\t", header=None, index=False)
