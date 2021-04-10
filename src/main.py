import sys
import pandas as pd
import random

# Local imports
from utils.data import *
from decision_tree.tree import Node
from decision_tree.DecisionTree import DecisionTree
from cross_validation.kfold import kfold


NB_ARGUMENTS = 5
SEED = 42


if __name__ == "__main__":
    if len(sys.argv) == NB_ARGUMENTS:
        data_file = sys.argv[1]
        delimiter = sys.argv[2]
        nb_bins = int(sys.argv[3])
        target_attr = sys.argv[4]
        nb_attributes_node_split = 0  # sys.argv[5]
        number_of_tress = 10
        number_of_folds = 5

        df = pd.read_csv(data_file, sep=delimiter, engine='python')
        target_values = [0, 1]
        random.seed(a=SEED)
        m, d = kfold(df, number_of_folds, number_of_tress, target_attr,
                     target_values, nb_attributes_node_split)
        print("number_of_tress:", str(number_of_tress))
        print('Median: ' + str(m))
        print('Stddev: ' + str(d))

    else:
        print("Wrong number of arguments ({}). Please use script as python3 main.py <data_file_name> '<file_delimiter>' <nb_bins> <target_attribute>. E.g.: python3 main.py data.csv ',' 5 'target'\nThe number of bins will be disregarded if the data is composed of entirely categorical attributes.".format(len(sys.argv)))
