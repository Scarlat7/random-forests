
"""
Node for a N-tree implementation
"""
class Node():

    CATEGORICAL = 1
    NUMERICAL = 2

    def __init__(self, category = None, gain = None, attribute = None, children = {}):
        self.children = children
        self.criterium_attribute = attribute
        self.crit_attribute_type = None
        self.gain = gain
        self.category = category
    
    def set_criterium_attribute(self, attribute, type):
        self.criterium_attribute = attribute
        self.crit_attribute_type = type
    
    """
    Add child so that if a sample has the "attributeValue" in the criterium attribute
    of this node, it will be analyzed going through the branch represented by "node"
    """
    def add_child(attributeValue, node):
        self._children[attribute_value] = node