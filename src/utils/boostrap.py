import random
import pandas as pd


SEED = 42


class Bootstrap():
    def __init__(self, df: pd.DataFrame):
        """
        @inputs: df: data that will be bootstraped
        """
        self.data = df


    def get_new_bag(self, length, seed):
        """
        Creates a new bag using sampling with replacement
        @input: length - the size of the list of instances that is returned by the method
        @seed: seed - the seed that will be used to guarantee reproducibility
        @output: a dataframe of size length filled with test instances randomly chosen from the underlying sample
        """
        # replace allow sampling of the same row more than once.
        return self.data.sample(n=length, replace=True, random_state=seed)

    def get_n_bootstrap_instances(self, n: int, length: int):
        """
        Return a list of n dataframes of size length. The dataframes are bootstraps of the original data set.
        """
        # generate a list of seeds to use on the dataframe sample,
        # this guarantee reproducibility since the list of seed itself
        # guaranted by the the random.seed
        randomlist = []
        for _ in range(n):
            r = random.randint(0, n*1000)
            randomlist.append(r)

        l = []
        for i in range(n):
            bs = self.get_new_bag(length, randomlist[i])
            l.append(bs)

        return l
