from numpy import log2

# Local Imports
from .Tree import Node
from utils.data import all_equal

FIRST_ELEMENT = 0

class DecisionTree:

    """
    @input: data - data frame containing data
    @input: all_attr_values - all possible values for all attributes
    """
    def __init__(self, data, all_attr_values = None):
        self.data = data
        if all_attr_values is None:
            all_attr_values = {}
        self.all_attr_values = all_attr_values
        self.tree = None
    
    """
    Trains a decision tree, saving the root in the DecisionTree object
    @input: attributes - attributes from the data frame to consider while node splitting
    """
    def train_tree(self, attributes):
        self.save_all_attr_values()
        self.tree = self.train(attributes)

    """
    Trains a decision tree based on given data and outcomes
    @input: attributes - attributes from the data frame to consider while node splitting (Series)
    @output: root node of the decision tree trained
    """
    def train(self, attributes):
        outcomes = self.data[self.data.columns[-1]]
        node = Node()
        if all_equal(outcomes):
            node.category = outcomes.iloc[FIRST_ELEMENT]
        elif attributes.empty:
            node.category = outcomes.mode().iloc[FIRST_ELEMENT]
        else:
            chosen_attribute, info_gain = self.get_best_attribute()
            node.attribute = chosen_attribute
            node.gain = info_gain
            attributes.drop(labels = chosen_attribute, inplace = True)
            node = self.node_split(node, chosen_attribute, attributes)
        return node
    
    """
    Splits the tree into N nodes. N being the number of different values
    that the chosen attribute can take. This only works for discrete attributes.
    For numerical attributes, a pre-processing on the data must be carried out.
    @input node - the tree node where the split takes place
    @input chosen_attribute - the attribute the split will be based on
    @input: attributes - attributes from the data frame to consider while node splitting
    @output - none (side-effect: the node's children will be updated)
    """
    def node_split(self, node, chosen_attribute, attributes):
        outcomes = outcomes = self.data[self.data.columns[-1]]
        for attr_value in self.all_attr_values[chosen_attribute]:
            attribute_data = self.get_all_samples_with_given_attribute_value(chosen_attribute, attr_value)
            if  attribute_data.empty:
                node.category = outcomes.mode().iloc[FIRST_ELEMENT]
            else:
                subtree = DecisionTree(attribute_data, self.all_attr_values)
                node.add_child(attr_value, subtree.train(attributes.copy()))
        return node

    """
    Chooses attribute with the greatest information gain
    @output attribute name
    @output information gain for this attribute
    """
    def get_best_attribute(self):
        # Disregards the last column (target)
        all_attributes = self.data.columns[:-1]
        information_gains = []
        for attribute in all_attributes:
            information_gains.append(self.information_gain(attribute))
        max_info_gain = max(information_gains)
        return all_attributes[information_gains.index(max_info_gain)], max_info_gain

    """
    Returns the information gain for a division based on this attribute
    @attribute: attribute - attribute being considered on node split
    @output: information gain value
    """
    def information_gain(self, attribute):
        outcomes = self.data[self.data.columns[-1]]
        general_entropy = self.general_entropy(outcomes)
        attribute_entropy = self.attribute_entropy(attribute)
        return general_entropy - attribute_entropy
    
    """
    Returns the entropy of a given attribute split
    @input: attribute - attribute being considered for the split
    @output: entropy value for this attribute
    """
    def attribute_entropy(self, attribute):
        attr_entropy = 0
        counts = self.data[attribute].value_counts()
        for attr_value in self.data[attribute].unique():
            partition_weight = counts[attr_value]/len(self.data)
            partition_data = self.data.loc[self.data[attribute] == attr_value]
            partition_outcomes = partition_data[partition_data.columns[-1]]
            attr_entropy = attr_entropy + partition_weight*self.general_entropy(partition_outcomes)
        return attr_entropy

    """
    Returns the general entropy considering the outcomes provided
    @input outcomes: data frame containing outcomes
    @output: general entropy value
    """
    def general_entropy(self, outcomes):
        entropy = 0
        category_counter = outcomes.value_counts()
        total = len(outcomes)
        for category in outcomes.unique():
            category_prob = category_counter[category]/total
            entropy = entropy + category_prob*log2(category_prob)
        return -entropy

    """
    Returns all rows of data frame that have an attribute with a given value
    @input attribute - the attribute being considered
    @input value - the value being tested for
    @output: all rows of the DF with the given attribute value
    """
    def get_all_samples_with_given_attribute_value(self, attribute, value):
        return self.data[self.data[attribute] == value] 

    """
    Classifies a sample based on the trained tree
    @input sample - the sample being classified
    @input node - the next node for the recursion, or None if it's the first call
    @output the sample target class
    """
    def classify_sample(self, sample, node = None):
        if node is None and self.tree is None:
            print("The tree hasn't been trained yet, can't classify new instance")
            return
        elif node is not None and node.category is not None:
            return node.category
        else:
            if node is None:
                node = self.tree
            return self.classify_sample(sample, node.children[sample[node.attribute]])
    
    """
    Save all possible values for each attribute.
    """
    def save_all_attr_values(self):
        for attr in self.data.columns[:-1]:
            self.all_attr_values[attr] = self.data[attr].unique()