import random
import pandas as pd


SEED = 42


class Bootstrap():
    """
    underlying_sample: list of test instances from where the bag items will be drawn
    """

    def __init__(self, df: pd.DataFrame):
        self.data = df

    """
  Creates a new bag using sampling with replacement
  @input: the size of the list of instances that is returned by the method
  @output: a list of size length filled with test instances randomly chosen from the underlying sample
  """

    def get_new_bag(self, length, seed):
        # replace allow sampling of the same row more than once.
        return self.data.sample(n=length, replace=True, random_state=seed)

    def get_n_bootstrap_instances(self, n: int, length: int):
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
