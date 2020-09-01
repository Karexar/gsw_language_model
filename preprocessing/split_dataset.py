import numpy as np

def split_dataset(df, train_proportion, valid_proportion):
    """Split a dataframe into a train, validation, and test set. The size of the
    test set will be 1.0 minus the given train and valid proportion.

    Parameters
        df - pandas.core.frame.DataFrame
            The dataframe to split
        train_proportion - float
            A value between 0.0 and 1.0 indicating the proportion of sentences
            to take for the training set
        valid_proportion - float
            A value between 0.0 and 1.0 indicating the proportion of sentences
            to take for the validation set.

    Returns
        List[pandas.core.frame.DataFrame]
            A list [train, valid, test] containing the corresponding dataframes
    """
    shuffled = df.sample(frac=1)
    first_break = train_proportion
    second_break = train_proportion + valid_proportion
    sets = np.split(shuffled, [int(first_break*len(shuffled)),
                               int(second_break*len(shuffled))])
    return sets
