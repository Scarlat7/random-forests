from decision_tree import Node

TEST_CHILD_NODE = Node()
TEST_CATEGORY = 'sampleCategory'
TEST_GAIN = 0.7
TEST_ATTRIBUTE = 'sampleAttribute'
TEST_ATTR_VALUE = 'sampleAttributeValue'

def test_init_tree():
    
    node = Node(TEST_CATEGORY, TEST_GAIN, TEST_ATTRIBUTE, {TEST_ATTR_VALUE : TEST_CHILD_NODE})

    if node.category == TEST_CATEGORY and node.gain == TEST_GAIN and node.attribute == TEST_ATTRIBUTE and node.children[TEST_ATTR_VALUE] is TEST_CHILD_NODE:
        print('Test test_init_tree SUCCEEDED')
    else:
        print('Test test_init_tree FAILED')

def test_add_child():
    node = Node()
    node_child = Node()

    node.add_child(TEST_ATTRIBUTE, node_child)

    if node.children[TEST_ATTRIBUTE] is node_child:
        print('Test test_add_child SUCCEEDED')
    else:
        print('Test test_add_child FAILED')

def test_is_leaf():
    node = Node(category = TEST_CATEGORY)

    if node.is_leaf():
        print('Test test_is_leaf SUCCEEDED')
    else:
        print('Test test_is_leaf FAILED')

if __name__ == "__main__":
    test_init_tree()
    test_add_child()
    test_is_leaf()