# random-forests
Random forests algorithm implementation for UFRGS' class INF1017 - Machine Learning of 2020/2.

# Installation instructions

## Requirements
- Python 3
- Linux / MacOS (the instructions may be adapted for Windows usage)

## Virtual environment (Optional)
To install all required packages in a virtual python environment, so as not to disturb your main Python 3 installation, you may install the `virtualenv` package.

The following commands will create a virtual environment called `env` so to install all project-required dependencies in it (those are listed on the `requirements.txt` file).

```shell
python3 -m pip install --user virtualenv  # Install package
python3 -m venv env                       # Create 'env' virtual environment
source env/bin/activate                   # Activate virtual environment
```

Calling the following command will deactivate the virtual environment.

````shell
deactivate
````

## Installing dependencies

To install all required dependencies from the `requirements.txt` file, run the following command:

```shell
python3 -m pip install -r requirements.txt
```

# Running unit tests

All unit tests follow a similar approach to how to be run. For example, to run the unit test for the TestTree file:

````shell
cd src
python3 -m decision_tree.test.TestTree
````

# Running the Random Forests algorithm

To run the test benchmark, to validate the basic decision tree implementation, run:

````shell
make test
````

To run the test with the categorical data (that is, the house votes dataset):
````shell
make categorical
````

To run the test with the numerical data (that is, the wine recognition dataset):
````shell
make numerical
````

These are make templates using the `Makefile` present in the root directory. To generally run the application, one may use:

````shell
python3 main.py <data_file_name> '<file_delimiter>' <nb_bins> <target_attribute> <attr_type>
````

For example:

````shell
python3 main.py data.csv ',' 5 'target' 'c'
````

Where 'c' means categorical data and 'n' means numerical data.

To vary the number of tree and folds, one may change the attributes `number_of_tress` and `number_of_folds` on the `main.py` file.