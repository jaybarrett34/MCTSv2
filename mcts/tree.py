from .node import Node

class Tree:
    def __init__(self, root_state):
        self.root = Node(root_state)

    def get_root(self):
        return self.root

    def __repr__(self):
        return f"Tree(root={self.root})"