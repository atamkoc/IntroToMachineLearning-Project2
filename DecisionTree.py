import utility
import numpy as np
from node import Node


class DecisionTree:
    def __init__(self):
        self.root = None
        self.twig_list = []

    def create_tree(self, df_array: list, attr_list: list) -> None:
        """
        Creates the tree starting from its root node

        Arguments:
            df_array [np.ndarray]: The array of the given data set
            attr_list [list]: The list of the attribute names
        Returns:
        """
        counts = {}
        selected_attribute, max_gain = utility.find_max_gain(df_array, attr_list, counts)
        self.root = Node(selected_attribute, count=counts[selected_attribute], info_gain=max_gain, attributes=attr_list,
                         data_set=df_array)
        self.root.create_children()
        self._generate_leafs()

    def _generate_leafs(self) -> None:
        """
        Creates the rest of the tree
        Uses a node list to keep track of the new created nodes
        Every node that is created from an attribute is added to this node list
        Then the code takes every node and check its children, if a child has a decision, it skips that child
        If not, it continues and creates other attribute nodes with filtered data
        If there is no selected attr, or max_gain == 0, then it does not create a child but assigns the most common value
        from that child
        Arguments:
        Returns:
        """
        node_list = [self.root]
        while node_list:
            current_node = node_list.pop(0)
            attr = current_node.title
            attr_list = current_node.attributes.copy()
            attr_index = attr_list.index(attr)
            attr_list.remove(attr)

            counts = {}
            for child in current_node.children:
                if child.decision is not None:
                    continue
                data_set = utility.filter_data_set(current_node.data_set, attr_index, child.title)
                data_set = np.delete(data_set, attr_index, axis=1)
                selected_attr, max_gain = utility.find_max_gain(data_set, attr_list, counts)
                if selected_attr is not None:
                    new_node = Node(selected_attr, count=counts[selected_attr], info_gain=max_gain, attributes=attr_list,
                                    data_set=data_set, parent=child)
                    new_node.create_children()
                    child.children.append(new_node)
                    node_list.append(new_node)
                else:
                    child.decision = child.find_most_common()

    def prune(self) -> (Node, Node):
        """
        Prunes the tree with using twig list
        Find the least information gain node then deletes it from the list
        After deletion checks its parent whether it turns a twig or not, if yes adds the parent to the list

        Arguments:

        Returns:
            temp_node [node]: the deleted node from the twig list
            temp_node.parent [node]: the parent of the deleted node, in case of reverting the pruning
        """
        min_gain = float('inf')
        temp_node = None
        for twig in self.twig_list:
            if twig.info_gain < min_gain:
                min_gain = twig.info_gain
                temp_node = twig
        temp_node.parent.decision = temp_node.find_most_common()
        self.twig_list.remove(temp_node)
        temp_node.parent.children.clear()
        if temp_node.parent.parent.is_twig():
            self.twig_list.append(temp_node.parent.parent)

        return temp_node, temp_node.parent

    def revert_prune(self, twig: Node, parent: Node) -> None:
        """
        Reverts one step of the pruning operation
        Add the twig back to the tree, twig list and checks if its parent is in twig list
        If yes, deletes the parent from the twig list

        Arguments:

        Returns:

        """
        if twig.parent.parent.is_twig():
            self.twig_list.remove(twig.parent.parent)
        parent.children.append(twig)
        self.twig_list.append(twig)
        parent.decision = None

    def find_twigs(self, tree_node: Node) -> None:
        """
        Finds the twigs in the tree and adds them to the twig list before pruning operation starts
        Works similar to the depth first search in a recursive manner

        Arguments:
            tree_node [Node]: the current node of the tree
        Returns:

        """
        if not tree_node.is_twig():
            for child in tree_node.children:
                if child.decision is None:
                    self.find_twigs(child.children[0])
        else:
            self.twig_list.append(tree_node)
            return

    def predict(self, test: np.array) -> str:
        """
        Makes a prediction of the given test data

        Arguments:
            test [np.array]: A row from the test data
        Returns:
            decision [str]: Returns the decision found
        """
        current_node = self.root
        attr_list = self.root.attributes
        child_index = 0
        while child_index < len(current_node.children):
            child = current_node.children[child_index]
            attr_index = attr_list.index(current_node.title)
            if test[attr_index] == child.title:

                if child.decision is not None:
                    return child.decision
                current_node = child.children[0]
                child_index = 0
            else:
                child_index += 1

    def print_rules(self, tree_node: Node, print_list: list) -> None:
        """
        Prints out the whole tree
        Because the tree has different branches from a single node, adds every node that is not the end of the path to
        the print_list
        When it reaches to the end, prints out the nodes, then deletes the last node added
        Works recursively, in the first call tree_node is root, print_list is empty list

        Arguments:
            tree_node [Node]: the current node of the tree
            print_list [list]: the list of the nodes which are printed out recursively
        Returns:

        """
        print_list.append(tree_node)
        for child in tree_node.children:
            if child.decision is None:
                print_list.append(child)
                self.print_rules(child.children[0], print_list)
                print_list.remove(child)
            else:
                counter = 0
                for temp_node in print_list:
                    if temp_node.decision is None:
                        if counter % 2 == 0:
                            print(f"({temp_node.title} = ", end="")
                        else:
                            print(f"{temp_node.title}) ^ ", end="")
                        counter += 1
                print(f"{child.title}) -> {child.decision} ")

        print_list.remove(tree_node)
