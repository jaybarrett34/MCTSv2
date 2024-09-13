import uuid

class Node:
    def __init__(self, state, action=None, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.is_terminal = False
        self.immediate_reward = 0
        self.id = str(uuid.uuid4())

    def add_child(self, state, action):
        child = Node(state, action, self)
        self.children.append(child)
        return child

    def __repr__(self):
        return f"Node(state={self.state}, action={self.action}, visits={self.visits}, value={self.value:.2f})"