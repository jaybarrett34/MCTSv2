import math
import numpy as np
from .tree import Tree
import logging

logging.basicConfig(level=logging.DEBUG)

class UCT:
    def __init__(self, simulator, max_iterations=1000, exploration_weight=1.4, epsilon=0.9):
        self.simulator = simulator
        self.max_iterations = max_iterations
        self.exploration_weight = exploration_weight
        self.epsilon = epsilon

    def search(self, state):
        self.simulator.reset()  
        tree = Tree(state)
        for i in range(self.max_iterations):
            logging.debug(f"Iteration {i}")
            if not tree.root.children:
                self.expand(tree.root)
            node = self.select(tree.root)
            reward, done = self.simulate(node.state)
            self.backpropagate(node, reward)
        
        return self.best_action(tree.root)

    def select(self, node):
        while node.children:
            if np.random.random() < self.epsilon:
                return np.random.choice(node.children)
            if not all(child.visits > 0 for child in node.children):
                return self.expand(node)
            else:
                node = self.ucb_select(node)
        return node

    def expand(self, node):
        unvisited_actions = set(range(self.simulator.get_action_space())) - set(child.action for child in node.children)
        if not unvisited_actions:
            logging.warning(f"No unvisited actions for node {node.state}")
            return node
        action = np.random.choice(list(unvisited_actions))
        next_state, reward, done = self.simulator.take_action(node.state, action)
        child = node.add_child(next_state, action)
        logging.debug(f"Expanded node {node.state} with action {action} to {next_state}")
        return child

    def ucb_select(self, node):
        log_n_visits = math.log(node.visits)
        return max(node.children, key=lambda c: c.value / (c.visits + 1e-5) + self.exploration_weight * math.sqrt(log_n_visits / (c.visits + 1e-5)))

    def simulate(self, state):
        current_state = state
        done = False
        total_reward = 0
        depth = 0
        max_depth = 100
        while not done and depth < max_depth:
            action = np.random.randint(self.simulator.get_action_space())
            next_state, reward, done = self.simulator.take_action(current_state, action)
            total_reward += reward
            current_state = next_state
            depth += 1
        logging.debug(f"Simulation from {state} ended with reward {total_reward}")
        return total_reward, done

    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
        logging.debug(f"Backpropagated reward {reward}")

    def best_action(self, node):
        if not node.children:
            logging.error("Attempting to select best action from node with no children")
            return np.random.randint(self.simulator.get_action_space())
        return max(node.children, key=lambda c: c.visits).action