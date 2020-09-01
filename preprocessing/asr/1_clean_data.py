# This script merges all GSW dataset except PMK. It cleans the data and process
# the text for the kaldi language model (i.e. replace digits by written numbers) 

import pandas as pd
import os
from pathlib import Path
from preprocessing.cleaner import *
from tqdm import tqdm
from plato_ai_asr_preprocessor.preprocessor import Preprocessor

###  Settings  #################################################################
twitter_labelled_path = "data/twitter/labelled/full.csv"
twitter_unlabelled_path = "data/twitter/unlabelled/full.csv"
whatsup_labelled_path = "data/whatsup/labelled/full.csv"
whatsup_unlabelled_path = "data/whatsup/unlabelled/full.csv"
leipzig_path = "data/leipzig/leipzig.txt"
swisstext_path = "data/swisstext/swisscrawl-2019-11-23.csv"
output_dir = "data/lm/asr"
################################################################################


# load the datasets
df_twitter_lab = pd.read_csv(twitter_labelled_path, header=None, sep="\t")
print(df_twitter_lab.shape)
df_twitter_unlab = pd.read_csv(twitter_unlabelled_path, header=None, sep="\t")
print(df_twitter_unlab.shape)
df_whatsup_lab = pd.read_csv(whatsup_labelled_path, header=None, sep="\t")
print(df_whatsup_lab.shape)
df_whatsup_unlab = pd.read_csv(whatsup_unlabelled_path, header=None, sep="\t")
print(df_whatsup_unlab.shape)
df_leipzig = pd.read_csv(leipzig_path, header=None, sep="\t")
print(df_leipzig.shape)
df_swisstext = pd.read_csv(swisstext_path)
print(df_swisstext.shape)

sentences = list(df_twitter_lab.iloc[:, 0].values) + \
            list(df_twitter_unlab.iloc[:, 0].values) + \
            list(df_whatsup_lab.iloc[:, 0].values) + \
            list(df_whatsup_unlab.iloc[:, 0].values) + \
            list(df_leipzig.iloc[:, 0].values) + \
            list(df_swisstext.iloc[:, 0].values)

print(f"Sentences count : {len(sentences)}")

preprocessor = Preprocessor(use_case='kaldi-lm')

def clean_text(text):
    text = Cleaner.remove_smileys(text)
    text = Cleaner.remove_hat_element(text)
    text = Cleaner.remove_html_entities(text)
    text = Cleaner.remove_special_chars(text, set("#@[]{}<>=^\\_~$'*/"))
    text = preprocessor.process(text=text, language='de')[0]
    text = Cleaner.remove_special_duplication(text)
    text = Cleaner.isolate_special_characters(text)
    text = Cleaner.remove_special_words(text)
    text = Cleaner.remove_duplicated_spaces(text)
    text = Cleaner.replace_non_gsw_accent(text)
    return text

cleaned = [clean_text(x) for x in tqdm(sentences)]
# remove duplicates
cleaned = list(set(cleaned))

# Save the cleaned sentences
Path(output_dir).mkdir(parents=True, exist_ok=True)
with open(os.path.join(output_dir, "gsw_sentences.txt"), "w") as f:
    for x in cleaned:
        f.write(x + "\n")
