# random-forests
Random forests algorithm implementation for UFRGS' class INF1017 - Machine Learning of 2020/2.

# Tasks
- [ ] Decision Tree Algorithm - with Information Gain (Natália)
  - [X] Category attributes (Natália)
  - [ ] Numerical attributes (Natália)
- [ ] Function to classify new instance with decision tree (Natália)
- [ ] Bootstrap for data sampling
- [ ] Sampling of m attributes for each node division (Natália)
- [ ] Training of an ensemble of trees
- [ ] Majority voting for the ensemble
- [ ] Stratified cross-validation
- [ ] Assessment of the performance with different number of trees in the ensemble
- [ ] README with instructions on how to execute the code


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

TBD