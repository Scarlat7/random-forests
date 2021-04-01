import pandas as pd
import random
from statistics import median, stdev

SEED = 42


def calculate_ratio(df, target_column, target_value):
    t = df[target_column].value_counts(normalize=True)
    return t[target_value]


def split_data(df, k: int, target_name: str, target_values: list):
    # get info to stratify data
    r_yes = calculate_ratio(df, target_name, target_values[0])
    fold_size = df[target_name].size / k
    n_yes_per_fold = fold_size * r_yes
    n_no_per_fold = fold_size * (1.0 - r_yes)
    if n_yes_per_fold  < 1 or n_no_per_fold < 1 :
        raise "Number of folds to high, will not be able to stratify"

    # since we divide into the folds getting from the start of the table, it is good to shuffle the rows first
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # divide yes and no
    yes_df = df.loc[df[target_name] == target_values[0]]
    no_df = df.loc[df[target_name] == target_values[1]]

    # divide in folds based on the stratified info
    folds = []
    for i in range(0, k):
        if i != k-1:
            y = yes_df.iloc[:int(n_yes_per_fold), :]
            yes_df = yes_df.iloc[int(n_yes_per_fold):, :]
            n = no_df.iloc[:int(n_no_per_fold), :]
            no_df = no_df.iloc[int(n_no_per_fold):, :]
            f = pd.concat([n, y])
        else:
            # on the last fold we just add everything that is left,
            # this is to avoid rounding errors
            f = pd.concat([no_df, yes_df])

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
    
    print(scores)
    return median(scores), stdev(scores)


if __name__ == "__main__":
    random.seed(a=SEED)
    # df = pd.read_csv("./data/dadosBenchmark_validacaoAlgoritmoAD.csv", sep=';')
    # target_name = 'Joga'
    # target_values = ['Sim', 'Nao']
    df = pd.read_csv("./data/house_votes_84.tsv", sep='\t')
    target_name = 'target'
    target_values = [0,1]

    m, d = kfold(df, 10, target_name, target_values)
    print('Median: ' + str(m))
    print('Stddev: ' + str(d))
