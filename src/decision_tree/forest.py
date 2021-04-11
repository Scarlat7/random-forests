import random
import pandas as pd

from .DecisionTree import DecisionTree
from utils.boostrap import Bootstrap


class Forest:
    def __init__(self, n_trees, train_data, target, nb_attr_node_split):
        """
        Initialize and build a forest of DecisionTrees.
        @input: n_trees - number of trees in the forest
        @input: train_data - data used to bootstrap and train the trees
        @input: target - target attribute name
        @input: nb_attr_node_split - number of attributes to randomly pick on each node split
        """
        self.train_data = train_data
        self.target = target
        self.nb_attr_node_split = nb_attr_node_split
        self.n_trees = n_trees
        self.build_forest()


    def build_forest(self):
        """
        Bootstrap the train data and use to create and train n_trees.
        """
        bs = Bootstrap(self.train_data)
        # the size of the bootstrap is the size of the train data
        bs_list = bs.get_n_bootstrap_instances(
            self.n_trees, len(self.train_data))

        forest = []
        for train_data in bs_list:
            # create and train trees
            decision_tree = DecisionTree(
                train_data, self.target, self.nb_attr_node_split)
            attributes_without_target = train_data.columns.drop(
                labels=self.target)
            decision_tree.train_tree(attributes_without_target.to_series())
            forest.append(decision_tree)

        self.forest = forest


    def forest_election(self, row):
        """
        Classify a instace with every tree in the forest and return the result with more votes.
        @input: row - the instance to be classified.
        """
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

        if winner == "":
            raise "something went wrong on election"
        return winner

    def forest_score(self, test_data: pd.DataFrame):
        """
        Return de score of a forest for a given test_data.
        @input: test_data - data that will be used to score the forest.
        """
        sum = 0
        for _, row in test_data.iterrows():
            result = self.forest_election(row)
            if result == row[self.target]:
                sum += 1
        return sum / len(test_data)
