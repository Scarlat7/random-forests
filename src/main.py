import sys
import pandas as pd

# Local imports
from utils.data import *
from decision_tree.Tree import Node
from decision_tree.DecisionTree import DecisionTree

NB_ARGUMENTS = 3
NB_BINS = 5

if __name__ == "__main__":
    if len(sys.argv) == NB_ARGUMENTS:
        data_file = sys.argv[1]
        delimiter = sys.argv[2]
        df = pd.read_csv(data_file, sep = delimiter)
        decision_tree = DecisionTree(df)
        decision_tree.train_tree(df.columns[:-1].to_series())
        print("Predicted class: " + decision_tree.classify_sample(df.iloc[0]))
    else:
        print("Wrong number of arguments ({}). Please use script as python3 main.py <data_file_name> '<file_delimiter>'. E.g.: python3 main.py data.csv ','".format(len(sys.argv)))