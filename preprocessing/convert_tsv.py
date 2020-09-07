import csv
from preprocessing.cleaner import *
from tqdm import tqdm
import os

input_paths = []
for dialect in ["ZH", "BE", "CE", "NW", "EA", "GR", "VS"]:
    path = os.path.join("data/whatsapp/predicted_dialect", dialect)
    path = os.path.join(path, "test.csv")
    input_paths.append(path)

# for name in ["known_labels", "predicted_labels"]:
#     dir_labels = os.path.join("data/dialect_specific", name)
#     for dialect in ["ZH", "BE", "CE", "NW", "EA", "GR", "VS", "all"]:
#         dir_dialect = os.path.join(dir_labels, dialect)
#         for set_name in ["train.csv", "valid.csv", "test.csv"]:
#             path = os.path.join(dir_dialect, set_name)
#             input_paths.append(path)

# input_paths=["data/dialect_specific/known_labels/BE/train.csv",
#              "data/dialect_specific/known_labels/BE/valid.csv",
#              "data/dialect_specific/known_labels/BE/test.csv"]
# input_paths = ["data/swisstext_leipzig/test.csv",
#                "data/swisstext_leipzig/valid.csv",
#                "data/swisstext_leipzig/train.csv",
#                "data/swisstext_leipzig/test_20k.csv",
#                "data/twitter_swisstext_leipzig/test.csv",
#                "data/twitter_swisstext_leipzig/valid.csv",
#                "data/twitter_swisstext_leipzig/train.csv",
#                "data/twitter_swisstext_leipzig/test_20k.csv",
#                "data/whatsapp/full.csv",
            #    "data/twitter/train.csv",
            #    "data/twitter/valid.csv",
            #    "data/twitter/test.csv",
            #    "data/twitter/test_20k.csv"]


for input_path in input_paths:
    print(input_path)
    with open(input_path, "r", encoding="utf8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar="")
        lines = [[line[0]] for line in reader]
        # print("remove special chars")
        # lines = [Cleaner.remove_special_chars(x) for x in tqdm(lines)]
        # print("cleaning spaces")
        # lines = [Cleaner.clean_spaces(x) for x in tqdm(lines)]

    # print(lines[0])
    #
    # print("computing unique chars")
    # cc = [z for x in lines for y in x.split() for z in list(y)]
    # cc = sorted(list(set(cc)))
    # print("\t" in cc)
    # print("\"" in cc)
    # print("\\" in cc)

    # res = []
    # for i in range(len(lines)):
    #     sentence = lines[i].replace("\"", "").replace("\\", "")
    #     res.append([str(i) + "\t" + sentence])
    #res = [[x] for x in res]
    # small = res[:10000]
    #
    # with open("data/swisstext_leipzig/small.tsv", "w") as f:
    #     writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar="", escapechar="\\")
    #     writer.writerows(small)

    dir_path = os.path.dirname(input_path)
    name = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(dir_path, name + ".tsv")
    # with open(out_path, "w", encoding="utf8") as f:
    #     writer = csv.writer(f,
    #                         delimiter="\t",
    #                         quoting=csv.QUOTE_NONE,
    #                         quotechar="",
    #                         escapechar="\\")
    #     writer.writerows(res)

    with open(out_path, "w",encoding="utf8", newline='') as f:
    	writer = csv.writer(f,
                            delimiter="\t",
                            quoting=csv.QUOTE_NONE,
                            quotechar="",
                            escapechar="\\")
    	writer.writerows(lines)
