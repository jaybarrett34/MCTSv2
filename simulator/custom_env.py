from gymnasium import Wrapper

class CustomFrozenLake(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Adjust rewards based on the new structure
        if terminated and reward == 0:  # Hole
            reward = -10
        elif not terminated and not truncated:  # Regular tile
            reward = -1
        # Goal state keeps its 0 reward
        
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)