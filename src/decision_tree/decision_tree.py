from numpy import log2

# Local Imports
from tree import Node
from src.utils.data import all_equal

FIRST_ELEMENT = 0

class DecisionTree():

    def __init__():
        self._tree = None

    """
    Trains a decision tree based on given data and outcomes
    @input: data - data frame containing data
    @input: attributes - attributes from the data frame to consider while node splitting
    @output: root node of the decision tree trained
    """
    def train(data, attributes):
        outcomes = data[df[df.columns[-1]]]

        Node node = new Node()
        if all_equal(outcomes):
            nodecategory = outcomes.iloc[FIRST_ELEMENT]
            return node
        elif attributes.empty:
            node.category = outcomes.mode().iloc[FIRST_ELEMENT]
            return node
        else:
            chosen_attribute = get_best_attribute(data)
            node.attribute = chosen_attribute
            attributes.drop(columns = chosen_attribute)
            # node split here

    """
    Chooses attribute with the greatest information gain
    @input data - data frame with training data
    @output attribute name
    """
    def get_best_attribute(data):
        # Disregards the last column (target)
        all_attributes = data.columns[:-1]
        information_gains = []
        for attribute in all_attributes:
            information_gains.append(information_gain(data_attribute))
        max_info_gain = max(information_gains)
        return all_attributes[information_gain.index(max_info_gain)]

    """
    Returns the information gain for a division based on this attribute
    @input: data - data frame with training data
    @attribute: attribute - attribute being considered on node split
    @output: information gain value
    """
    def information_gain(data, attribute):
        outcomes = data[df[df.columns[-1]]]
        general_entropy = general_entropy(outcomes)
        attribute_entropy = attribute_entropy(data, attribute)
        return general_entropy - attribute_entropy
    
    """
    Returns the entropy of a given attribute split
    @input: data - data frame with training data
    @input: attribute - attribute being considered for the split
    @output: entropy value for this attribute
    """
    def attribute_entropy(data, attribute):
        attr_entropy = 0
        counts = data[attribute].value_counts()
        for attr_value in data[attribute].unique():
            partition_weight = counts[value]/len(data)
            partition_data = data.loc[data[attribute] == attr_value]
            attr_entropy = attr_entropy + partition_weight*general_entropy(partition_data)
        return attr_entropy

    """
    Returns the general entropy considering the outcomes provided
    @input outcomes: data frame containing outcomes
    @output: general entropy value
    """
    def general_entropy(outcomes):
        entropy = 0
        category_counter = outcomes.value_counts()
        total = len(outcomes)
        for category in outcomes.unique():
            category_prob = outcomes[category]/total
            entropy = entropy + category_prob*log2(category_prob)
        return -entropy
        
