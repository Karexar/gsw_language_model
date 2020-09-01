# This script prepare the data for the generic language model
# It takes the swisstext and leipzig corpus and and create a train, valid,
# and test set. Then it adds the twitter corpus to create another train,
# valid, and test set.
# Finally, it creates a test set using the whatsapp corpus.
# All files are saved as .txt files.
# The goal is to compare how generalizable the swisstext/leipzig corpus is
# compared to a swisstext/leipzig/twitter corpus.

import pandas as pd
import numpy as np
import random
import os
from pathlib import Path
from preprocessing.cleaner import *
from tqdm import tqdm
from phrasal.norm_punc import *

###  Settings  #################################################################
leipzig_path = "data/leipzig_over_99.csv"
swisstext_path = "data/swisscrawl_over_99.csv"
twitter_path = "data/twitter_over_99.csv"
whatsapp_path = "data/whatsapp_over_99.csv"
output_dir_sl = "data/swisstext_leipzig"
output_dir_tsl = "data/twitter_swisstext_leipzig"
output_dir_whatsapp = "data/whatsapp/"
################################################################################

# regexs to remove urls, mentions, and hashtags
preprocessing_regex = [('https?[\w\.\:\-\/]*($|\s)', ' '),
                       ('www\.[\w\.\:\-\/]*($|\s)', ' '),
                       ('@\S*($|\s)', ' '),
                       ('#\S*($|\s)', ' ')]

# good characters set. Any sentence with characters not in this set will be
# removed.

def remove_sentences_with_special_chars(sentences):
    chars_ok = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    chars_ok += "ÀÁÂÄÈÉÊËÍÌÎÏÓÒÔÖÚÙÛÜàáâäèéêëìíîïôöòóüùúûÿ"
    chars_ok += " -,.?!0123456789%&\"\'()/$*+:;<=>[]\\^_{}|\\~€°²#"
    chars_ok = set(chars_ok)

    return [x for x in sentences if len(set(x).difference(chars_ok))==0]

def remove_sentences_with_special_words(sentences, max_char):
    """Remove sentences with words containing too much special characters.
    """

    chars_ok = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    chars_ok += "ÀÁÂÄÈÉÊËÍÌÎÏÓÒÔÖÚÙÛÜàáâäèéêëìíîïôöòóüùúûÿ"
    chars_ok += "0123456789"
    chars_ok = set(chars_ok)

    res = []
    for sentence in tqdm(sentences):
        words = sentence.split()
        ok = True
        for word in words:
            if len([c for c in word if c not in chars_ok]) > max_char:
                ok = False
        if ok:
            res.append(sentence)
    return res

# load the datasets
df_leipzig = pd.read_csv(leipzig_path, header=None, sep="\t")
leipzig_sentences = list(df_leipzig.iloc[:, 0])
print(f"leipzig length : {len(leipzig_sentences)}")
leipzig_sentences = remove_sentences_with_special_chars(leipzig_sentences)
leipzig_sentences = remove_sentences_with_special_words(leipzig_sentences, 3)
print(f"leipzig filtered length : {len(leipzig_sentences)}")
print(leipzig_sentences[0])
print(leipzig_sentences[-1])

df_swisstext = pd.read_csv(swisstext_path, header=None, sep="\t")
swisstext_sentences = list(df_swisstext.iloc[:, 0])
print(f"swisstext length : {len(swisstext_sentences)}")
swisstext_sentences = remove_sentences_with_special_chars(swisstext_sentences)
swisstext_sentences = remove_sentences_with_special_words(swisstext_sentences, 3)
print(f"swisstext filtered length : {len(swisstext_sentences)}")
print(swisstext_sentences[0])
print(swisstext_sentences[-1])

df_twitter = pd.read_csv(twitter_path, header=None, sep="\t")
twitter_sentences = list(df_twitter.iloc[:, 0])
print(f"twitter length : {len(twitter_sentences)}")
twitter_sentences = remove_sentences_with_special_chars(twitter_sentences)
twitter_sentences = remove_sentences_with_special_words(twitter_sentences, 3)
print(f"twitter filtered length : {len(twitter_sentences)}")
print(twitter_sentences[0])
print(twitter_sentences[-1])

df_whatsapp = pd.read_csv(whatsapp_path, header=None, sep="\t")
whatsapp_sentences = list(df_whatsapp.iloc[:, 0])
print(f"whatsapp length : {len(whatsapp_sentences)}")
whatsapp_sentences = remove_sentences_with_special_chars(whatsapp_sentences)
whatsapp_sentences = remove_sentences_with_special_words(whatsapp_sentences, 3)
print(f"whatsapp filtered length : {len(whatsapp_sentences)}")
print(whatsapp_sentences[0])
print(whatsapp_sentences[-1])

### cleaning ###

def clean_text(text):
    text = Cleaner.preprocess(text, preprocessing_regex)
    text = normalize_text(text, strip_emojis=True)
    text = Cleaner.remove_smileys(text)
    text = Cleaner.remove_hat_element(text)
    text = Cleaner.remove_html_entities(text)
    text = Cleaner.clean_punc(text)
    text = Cleaner.remove_groups_of_special_chars(text, 2)
    text = Cleaner.remove_special_duplication(text)
    text = Cleaner.remove_isolated_special_chars(text)
    return text

leipzig_sentences = [clean_text(x) for x in tqdm(leipzig_sentences)]
leipzig_sentences = [x for x in leipzig_sentences if len(x.split()) >= 5]
print(f"leipzig final length : {len(leipzig_sentences)}")
swisstext_sentences = [clean_text(x) for x in tqdm(swisstext_sentences)]
swisstext_sentences = [x for x in swisstext_sentences if len(x.split()) >= 5]
print(f"swisstext final length : {len(swisstext_sentences)}")
twitter_sentences = [clean_text(x) for x in tqdm(twitter_sentences)]
twitter_sentences = [x for x in twitter_sentences if len(x.split()) >= 5]
print(f"twitter final length : {len(twitter_sentences)}")
whatsapp_sentences = [clean_text(x) for x in tqdm(whatsapp_sentences)]
whatsapp_sentences = [x for x in whatsapp_sentences if len(x.split()) >= 5]
print(f"whatsapp final length : {len(whatsapp_sentences)}")

### splitting into sets ###

def split_sets(sentences, train_proportion, valid_proportion):
    random.shuffle(sentences)
    first_break = train_proportion
    second_break = train_proportion + valid_proportion
    sets = np.split(sentences, [int(first_break*len(sentences)),
                                int(second_break*len(sentences))])
    return [list(x) for x in sets]

leipzig_sets = split_sets(leipzig_sentences, 0.8, 0.1)
print("Leipzig")
for i, name in enumerate(["Train", "Valid", "Test"]):
    print(f"{name} : {len(leipzig_sets[i])}")
swisstext_sets = split_sets(swisstext_sentences, 0.8, 0.1)
print("Swisstext")
for i, name in enumerate(["Train", "Valid", "Test"]):
    print(f"{name} : {len(swisstext_sets[i])}")
twitter_sets = split_sets(twitter_sentences, 0.8, 0.1)
print("Twitter")
for i, name in enumerate(["Train", "Valid", "Test"]):
    print(f"{name} : {len(twitter_sets[i])}")

### merging ###

print("Swisstext / Leipzig")
sl_sets = dict()
sl_sets["train"] = list(set(leipzig_sets[0] + swisstext_sets[0]))
random.shuffle(sl_sets["train"])
print(f"Train : {len(sl_sets['train'])}")
sl_sets["valid"] = list(set(leipzig_sets[1] + swisstext_sets[1]))
random.shuffle(sl_sets["valid"])
print(f"Valid : {len(sl_sets['valid'])}")
sl_sets["test"] = list(set(leipzig_sets[2] + swisstext_sets[2]))
random.shuffle(sl_sets["test"])
print(f"Test : {len(sl_sets['test'])}")
sl_sets["test_20k"] = sl_sets["test"][:20000]

print("Twitter / Swisstext / Leipzig")
tsl_sets = dict()
tsl_sets["train"] = list(set(sl_sets["train"] + twitter_sets[0]))
random.shuffle(tsl_sets["train"])
print(f"Train : {len(tsl_sets['train'])}")
tsl_sets["valid"] = list(set(sl_sets["valid"] + twitter_sets[1]))
random.shuffle(tsl_sets["valid"])
print(f"Valid : {len(tsl_sets['valid'])}")
tsl_sets["test"] = list(set(sl_sets["test"] + twitter_sets[2]))
random.shuffle(tsl_sets["test"])
print(f"Test : {len(tsl_sets['test'])}")
tsl_sets["test_20k"] = tsl_sets["test"][:20000]

print("Whatsapp")
w_sentences = list(set(whatsapp_sentences))
print(f"Full : {len(w_sentences)}")

### Saving on disk ###

Path(output_dir_sl).mkdir(parents=True, exist_ok=True)
Path(output_dir_tsl).mkdir(parents=True, exist_ok=True)
Path(output_dir_whatsapp).mkdir(parents=True, exist_ok=True)

for name in ["train", "valid", "test", "test_20k"]:
    path = os.path.join(output_dir_sl, name + ".csv")
    with open(path, "w", encoding="utf8") as f:
        f.write('\n'.join(sl_sets[name]) + '\n')

for name in ["train", "valid", "test", "test_20k"]:
    path = os.path.join(output_dir_tsl, name + ".csv")
    with open(path, "w", encoding="utf8") as f:
        f.write('\n'.join(tsl_sets[name]) + '\n')

path = os.path.join(output_dir_whatsapp, "full.csv")
with open(path, "w", encoding="utf8") as f:
    f.write('\n'.join(w_sentences) + '\n')
