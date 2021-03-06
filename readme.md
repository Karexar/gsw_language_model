# Swiss-German language models

This project uses a GPT-2 language model to model Swiss-German. The implementation is slightly modified from [https://github.com/jungomi/swiss-language-model](https://github.com/jungomi/swiss-language-model)

## Generic Swiss-German language model

This is for creating generic Swiss-German language models. In our case we use a Twitter dataset on top of Leipzig and SwissCrawl. For all the following commands, make sure you updated the settings in the scripts.

First you will need to predict the probabilities of each sentence to be Swiss-German, in order to filter out the sentences with too low prediction.

```zsh
python -m preprocessing.predict_gsw
```
Then you need to prepare the dataset by filtering out low predictions, splitting into train, valid and test set, and in our case, merging SwissCrawl/Leipzig and SwissCrawl/Leipzig/Twitter.

```zsh
python -m preprocessing.generic.clean_data
```

The GPT-2 model takes tsv files as input, so you need to convert the files

```zsh
python -m preprocessing.convert_tsv
```

Now you can run the GPT-2 model. First you need to create a vocabulary, for example :

```zsh
python -m prepare_vocab -i data/twitter/train.tsv -o data/twitter/vocab
```

Then you can train a GPT-2 model from scratch using e.g.

```zsh
python train.py --train-text data/twitter/train.tsv \
                --validation-text data/twitter/valid.tsv \
                --name twitter_generic \
                --model "gpt2-scratch" \
                --fp16 \
                --num-epochs 100 \
                --vocab data/twitter/vocab \
```

Note that you will certainly stop before 20 epochs. Refer to [https://github.com/jungomi/swiss-language-model](https://github.com/jungomi/swiss-language-model) for more insight on the parameters.

## Dialect-specific language models

This is for creating dialect specific language models. For all the following commands, make sure you updated the settings in the scripts.

The first script takes a dataset containing a 'sentence', 'dialect', and 'predicted_dialect' column. It splits the dataset into train, valid, and test set for all dialect. The second script convert into tsv file format.

```zsh
python -m preprocessing.dialect_lm.split_dataset
python -m preprocessing.convert_tsv
```

Then train from scratch the GSW model, e.g.

```zsh
python -m prepare_vocab -i data/twitter/all/train.tsv -o data/twitter/all/vocab
python train.py --train-text data/twitter/all/train.tsv \
                --validation-text data/twitter/all/valid.tsv \
                --name twitter_all \
                --model "gpt2-scratch" \
                --fp16 \
                --num-epochs 100 \
                --vocab data/twitter/all/vocab \
```

And fine_tune on each dialect. For example here, we take the twitter_all model at epoch 12 and finetune it on BE.

```zsh
python train.py --train-text data/twitter/BE/train.tsv \
                --validation-text data/twitter/BE/valid.tsv \
                --name all_BE \
                --model gpt2 \
                --fp16 \
                --num-epochs 1 \
                --pre-trained log/twitter_all/0012
```

Finally evaluate using e.g. 

```zsh
python evaluate.py --dataset data\sl\test_20k.tsv --checkpoint log/sl_generic/0010

```
