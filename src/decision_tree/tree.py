"""
Node for a N-tree implementation
"""
class Node():

    def __init__(self, category = None, gain = None, attribute = None, children = {}):
        self.children = children
        self.attribute = attribute
        self.gain = gain
        self.category = category
    
    """
    Add child so that if a sample has the "attributeValue" in the criterium attribute
    of this node, it will be analyzed going through the branch represented by "node"
    """
    def add_child(self, attribute_value, node):
        self.children[attribute_value] = node

    """
    Returns true if node is a leaf node.
    @output boolean whether it's a leaf node
    """
    def is_leaf(self):
        return self.category is not None