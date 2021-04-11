# Program setup
PYTHON = python3
MAIN = main.py

# Datasets
TEST_DATASET=../data/dadosBenchmark_validacaoAlgoritmoAD.csv
HOUSE_DATASET=../data/house-votes-84.tsv
WINE_DATASET=../data/wine-recognition.tsv

# Delimiters
SEMI_COLON_DELIMITER = ';'
TAB_DELIMITER = '\t'

# Default number of bins for discretization
NB_BINS = 5

# Target attribute column names
TARGET_ATTR = 'target'
TARGET_ATTR_TEST = 'Joga'

# Target types
TARGET_TYPES_TEST = 'c'
TARGET_TYPES_HOUSE = 'c'
TARGET_TYPES_WINE = 'n'

.DEFAULT_GOAL = help

help:
	@echo "---------------HELP-----------------"
	@echo "To run the program with test data: make test"
	@echo "To run the program with house votes dataset: make categorical [NB_BINS=<integer>]"
	@echo "To run the program with wine dataset: make numerical"
	@echo "------------------------------------"

test:
	cd src; ${PYTHON}  ${MAIN} ${TEST_DATASET} ${SEMI_COLON_DELIMITER} ${NB_BINS} ${TARGET_ATTR_TEST} ${TARGET_TYPES_TEST}

categorical:
	cd src; ${PYTHON}  ${MAIN} ${HOUSE_DATASET} ${TAB_DELIMITER} ${NB_BINS} ${TARGET_ATTR} ${TARGET_TYPES_HOUSE}

numerical:
	cd src; ${PYTHON}  ${MAIN} ${WINE_DATASET} ${TAB_DELIMITER} ${NB_BINS} ${TARGET_ATTR} ${TARGET_TYPES_WINE}