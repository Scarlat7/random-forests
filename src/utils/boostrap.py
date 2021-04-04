import random

class Bootstrap():

  def __init__(self, underlying_sample):
    self.underlying_sample = underlying_sample

  def get_new_bag(self, length):
    bag = []
    for _ in range(length):
      random_index = random.randint(0, len(self.underlying_sample) - 1)
      random_item = self.underlying_sample[random_index]
      bag.append(random_item)
    return bag
