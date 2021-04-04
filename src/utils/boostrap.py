import random

class Bootstrap():
  """
  underlying_sample: list of test instances from where the bag items will be drawn
  """
  def __init__(self, underlying_sample):
    self.underlying_sample = underlying_sample

  """
  Creates a new bag using sampling with replacement
  @input: the size of the list of instances that is returned by the method
  @output: a list of size length filled with test instances randomly chosen from the underlying sample
  """
  def get_new_bag(self, length):
    bag = []
    for _ in range(length):
      random_index = random.randint(0, len(self.underlying_sample) - 1)
      random_item = self.underlying_sample[random_index]
      bag.append(random_item)
    return bag
