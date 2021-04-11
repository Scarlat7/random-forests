import random
import pandas as pd

from .DecisionTree import DecisionTree
from utils.boostrap import Bootstrap


class Forest:
    def __init__(self, n_trees, train_data, target):
        self.train_data = train_data
        self.target = target
        self.n_trees = n_trees
        self.build_forest()

    def build_forest(self):
        bs = Bootstrap(self.train_data)
        # the size of the bootstrap is the size of the train data
        bs_list = bs.get_n_bootstrap_instances(
            self.n_trees, len(self.train_data))

        forest = []
        for train_data in bs_list:
            decision_tree = DecisionTree(
                train_data, self.target)
            decision_tree.train_tree()
            forest.append(decision_tree)

        self.forest = forest

    def forest_election(self, row):
        votes = {}
        for tree in self.forest:
            v = tree.classify_sample(row)
            if v in votes:
                votes[v] += 1
            else:
                votes[v] = 1

        winner = ""
        max_votes = 0
        for key in votes:
            # if it is a draw, we flip a coin
            if votes[key] > max_votes or (votes[key] == max_votes and random.randint(0, 1)):
                winner = key
                max_votes = votes[key]

        return winner

    def forest_score(self, test_data: pd.DataFrame):
        sum = 0
        for _, row in test_data.iterrows():
            result = self.forest_election(row)
            if result == row[self.target]:
                sum += 1
        return sum / len(test_data)
