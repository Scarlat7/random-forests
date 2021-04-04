# For pandas shape method
COLUMNS = 1
ROWS = 0

"""
Returns partition of dataset into attributes and target outcomes.
Target outcomes are always considered the last colum on the dataset
@input: df - data frame containing dataset
@output: data attributes and target outcomes
"""
def get_partition_of_dataset(df):
    data = df.iloc[: , :df.shape[COLUMNS]-1]
    outcomes = df.iloc[: , df.shape[COLUMNS]-1]
    return data, outcomes


"""
Checks if all elements in a data frame column are the same
@input column: the data frame column to be analyzed
@output boolean, whether all elements are equal
"""
def all_equal(column):
    vector = column.to_numpy()
    return (vector[0] == vector).all()