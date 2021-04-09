import sys
import pandas as pd

# Local imports
from utils.data import *
from decision_tree.Tree import Node
from decision_tree.DecisionTree import DecisionTree

NB_ARGUMENTS = 5

if __name__ == "__main__":
    if len(sys.argv) == NB_ARGUMENTS:
        data_file = sys.argv[1]
        delimiter = sys.argv[2]
        nb_bins = int(sys.argv[3])
        target_attr = sys.argv[4]

        df = pd.read_csv(data_file, sep = delimiter)
        decision_tree = DecisionTree(df,target_attr)
        attributes_without_target = df.columns.drop(labels = target_attr)
        df = data_binning(df, attributes_without_target, nb_bins)
        decision_tree.train_tree(attributes_without_target.to_series())
        print("Predicted class: " + str(decision_tree.classify_sample(df.iloc[0])))
    else:
        print("Wrong number of arguments ({}). Please use script as python3 main.py <data_file_name> '<file_delimiter>' <nb_bins> <target_attribute>. E.g.: python3 main.py data.csv ',' 5 'target'\nThe number of bins will be disregarded if the data is composed of entirely categorical attributes.".format(len(sys.argv)))