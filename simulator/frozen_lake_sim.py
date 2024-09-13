import gymnasium as gym
from .custom_env import CustomFrozenLake

class FrozenLakeSim:
    def __init__(self, size=4, is_slippery=True):
        env = gym.make('FrozenLake-v1', map_name=f"{size}x{size}", is_slippery=is_slippery, render_mode=None)
        self.env = CustomFrozenLake(env)
        self.size = size
        self.is_slippery = is_slippery
        self.reset()

    def take_action(self, state, action):
        self.env.env.s = state  # Set the state directly
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # Adjust rewards based on the new structure
        if terminated and reward == 0:  # Hole
            reward = -10
        elif not terminated and not truncated:  # Regular tile
            reward = -1
        # Goal state keeps its 0 reward
        
        return observation, reward, done

    def reset(self):
        return self.env.reset()[0]  # Return only the initial state

    def get_action_space(self):
        return self.env.action_space.n