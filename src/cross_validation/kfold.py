import pandas as pd
import random
from statistics import median, stdev

SEED = 42


def calculate_ratio(df, target_column, target_value):
    '''
    calculate_ratio calculate the ratio that a value appears in the target column.
    '''
    t = df[target_column].value_counts(normalize=True)
    return t[target_value]


def split_data(df, k: int, target_name: str, target_values: list):
    '''
    split_data split the data into k folds, stratifying the data.
    '''
    # get info to stratify data
    target_ratios = []
    for t in target_values:
        r = calculate_ratio(df, target_name, t)
        target_ratios.append(r)
    fold_size = df[target_name].size / k
    n_target_per_fold = []
    for tr in target_ratios:
        n = fold_size * tr
        n_target_per_fold.append(n)

    if any(x < 1 for x in n_target_per_fold) :
        raise "Number of folds to high, will not be able to stratify"

    # since we divide into the folds getting from the start of the table, it is good to shuffle the rows first
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # divide targets 
    targets_df = []
    for t in target_values:
        d = df.loc[df[target_name] == t]
        targets_df.append(d)

    # divide in folds based on the stratified info
    folds = []
    for i in range(0, k):
        if i != k-1:
            fold_dfs = []
            for i in range(len(targets_df)):
                # extract the first n rows 
                d =  targets_df[i].iloc[:int(n_target_per_fold[i]), :]
                targets_df[i] = targets_df[i].iloc[int(n_target_per_fold[i]):, :]
                fold_dfs.append(d)
            f = pd.concat(fold_dfs)
        else:
            # on the last fold we just add everything that is left,
            # this is to avoid rounding errors
            f = pd.concat(targets_df)

        folds.append(f)

    return folds

# This will be changed for the function of testing the score of the forest
def mock_get_forest_score(train_folds, test_fold):
    return random.random()


def kfold(df, k: int, target_name: str, target_values: list):
    folds = split_data(df, k, target_name, target_values)
    scores = []
    for i in range(len(folds)):
        test_fold = folds[i]
        train_folds = [x for j, x in enumerate(folds) if j != i]
        s = mock_get_forest_score(train_folds, test_fold)
        scores.append(s)

    return median(scores), stdev(scores)

def validate_stratify_folds(df, k: int, target_name: str, target_values: list):
    '''
    validate_stratify_folds can be use to validate the stratifying of the data.
    It will print: 
    1 - The ratios of the original data frame and of all the folds.
    2 - The sizes
    '''
    original_stats = ''
    for v in target_values:
        d = df.loc[df[target_name] == v]
        original_stats += ' ratio ' + str(v) + ': ' + str(d[target_name].size / df[target_name].size)

    print('[ORIG] Size: ' + str(df[target_name].size) + original_stats)
    folds = split_data(df, k, target_name, target_values)
    total_folds_size = 0
    for f in folds:
        fold_stats = ''
        for v in target_values:
            d = f.loc[f[target_name] == v]
            fold_stats += ' ratio ' + str(v) + ': ' + str(d[target_name].size / f[target_name].size)
        total_folds_size += f[target_name].size
        print('[FOLD] Size: ' + str(f[target_name].size) + fold_stats)
    
    print('Original size: ' + str(df[target_name].size) + ' Sum of folds size: ' + str(total_folds_size))



if __name__ == "__main__":
    random.seed(a=SEED)
    # df = pd.read_csv("./data/dadosBenchmark_validacaoAlgoritmoAD.csv", sep=';')
    # target_name = 'Joga'
    # target_values = ['Sim', 'Nao']

    # df = pd.read_csv("./data/house_votes_84.tsv", sep='\t')
    # target_name = 'target'
    # target_values = [0,1]

    df = pd.read_csv("./data/wine_recognition.tsv", sep='\t')
    target_name = 'target'
    target_values = [1, 2, 3]

    m, d = kfold(df, 10, target_name, target_values)
    print('Median: ' + str(m))
    print('Stddev: ' + str(d))

    # validate_stratify_folds(df, 10, target_name, target_values)
    