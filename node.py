class Node:
    def __init__(self, title=None, count=None, info_gain=0, decision=None, data_set=None, attributes=None,
                 parent=None) -> None:
        self.title = title
        self.children = []
        self.count = count
        self.info_gain = info_gain
        self.decision = decision
        self.data_set = data_set
        self.attributes = attributes
        self.parent = parent

    def create_children(self) -> None:
        """
        When a node is created from an attribute, this function automatically creates children nodes from node's values
        In this way the path between the next node and attribute node is created.

        Arguments:
        Returns:
        """
        for key in self.count.keys():
            if key == "general":
                continue
            decision = None
            if self.count[key][0] == 0:
                decision = "No"
            elif self.count[key][1] == 0:
                decision = "Yes"
            child = Node(key, self.count[key], decision=decision, parent=self)
            self.children.append(child)

    def find_most_common(self) -> str:
        """
        Finds the most common result in the node
        For attributes count is a dictionary, for attribute values its a tuple

        Arguments:

        Returns:
            decision [str]: returns "Yes" or "No", depending on the number of "Yes" and "No"
        """
        if type(self.count) == tuple:
            yes_count = self.count[0]
            no_count = self.count[1]
        else:
            yes_count = self.count["general"][0]
            no_count = self.count["general"][1]

        if yes_count > no_count:
            return "Yes"
        return "No"

    def is_twig(self):
        """
        Checks if the node is twig
        The condition of the twig is that the node should not have any other paths to another attribute node
        So, if a node is twig, all of its children have a decision attribute assigned

        Arguments:

        Returns:
            bool [bool]: returns if the node is a twig or not
        """
        for child in self.children:
            if child.decision is None:
                return False
        return True
