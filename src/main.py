import sys
import pandas as pd

# Local imports
from utils.data import get_partition_of_dataset

NB_ARGUMENTS = 3

if __name__ == "__main__":
    if len(sys.argv) == NB_ARGUMENTS:
        data_file = sys.argv[1]
        delimiter = sys.argv[2]
        df = pd.read_csv(data_file, sep = delimiter)
        X,y = get_partition_of_dataset(df)
    else:
        print("Wrong number of arguments ({}). Please use script as python3 main.py <data_file_name> '<file_delimiter>'. E.g.: python3 main.py data.csv ','".format(len(sys.argv)))