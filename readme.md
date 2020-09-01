# Swiss-German language models

This project uses a GPT-2 language model to model Swiss-German. There are two parts .
- Create a Swiss-German language model using Twitter on top of Leipzig and SwissCrawl corpus
- Create dialect-specific language models to outperform a generic language model if dialects are known.

## Generic Swiss-German language model

For all the following commands, make sure you updated the settings in the scripts.

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

First prepare the data
