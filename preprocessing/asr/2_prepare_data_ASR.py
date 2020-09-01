# This script takes the dataset containing all GSW sentences, and produce a
# dictionary to use for the ASR. The dictionary contains all words that appear
# in the input dataset, and that do not appear in the PMK dataset. 

import pandas as pd

###  Settings  #################################################################
input_path = "data/lm/asr/gsw_sentences.txt"
input_pmk_path = "data/lm/asr/words.txt"
output_dict_path = "../transformer-model/lexicon/dict_without_pmk.txt"
output_sentence_path = "../transformer-model/lexicon/gsw_sentences.txt"
################################################################################

sentences = []
with open(input_path, "r") as f:
    sentences = [x[:-1] if x[-1]=="\n" else x for x in f.readlines()]


print(f"Total sentences : {len(sentences)}")
filtered = []
for sentence in sentences:
    words = sentence.split()
    ok = True
    for word in words:
        if word.isupper():
            ok = False
            break
    if ok:
        filtered.append(sentence)
print(f"Filtered sentences : {len(filtered)}")

words = [y for x in filtered for y in x.split()]

counts = pd.Series(words).value_counts()
common_words = set(counts[counts > 2].index)
print(f"Common words count : {len(common_words)}")

sentences = []
for sentence in filtered:
    words = sentence.split()
    ok = True
    for word in words:
        if not word in common_words:
            ok = False
            break
    if ok:
        sentences.append(sentence)

print(f"Sentences count with common words : {len(sentences)}")

sentences = list(set(sentences))
print(f"Sentences count without duplicates : {len(sentences)}")

unique_words = common_words #set(words)
print(f"Unique words : {len(unique_words)}")

pmk = []
with open(input_pmk_path, "r") as f:
    pmk = [x[:-1] if x[-1]=="\n" else x for x in f.readlines()]
    pmk = [x.split()[0] for x in pmk]

unique_pmk_words = set(pmk)
print(f"Unique pmk words : {len(unique_pmk_words)}")

new_words = sorted(list(unique_words.difference(unique_pmk_words)))
print(f"New words : {len(new_words)}")

with open(output_dict_path, "w") as f:
    idx = 0
    for word in new_words:
        f.write(str(idx) + "\t" + word + "\n")

with open(output_sentence_path, "w") as f:
    f.write('\n'.join(sentences))
