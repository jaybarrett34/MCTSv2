import gymnasium as gym
from simulator.frozen_lake_sim import FrozenLakeSim
from simulator.custom_env import CustomFrozenLake
from mcts.uct import UCT
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

def run_episode(env, uct):
    state, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    steps = 0

    while not (done or truncated):
        logging.debug(f"Current state: {state}")
        action = uct.search(state)
        logging.debug(f"Chosen action: {action}")
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        logging.debug(f"Action: {action}, Reward: {reward}, New state: {state}")

    logging.info(f"Episode finished. Total reward: {total_reward}, Steps: {steps}")
    return total_reward, steps

def main():
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="human")
    env = CustomFrozenLake(env)
    sim = FrozenLakeSim(size=4, is_slippery=False)
    uct = UCT(sim, max_iterations=1000, exploration_weight=1.4, epsilon=0.1)

    num_episodes = 100
    rewards = []
    steps_list = []

    for episode in range(num_episodes):
        logging.info(f"Starting episode {episode + 1}")
        reward, steps = run_episode(env, uct)
        rewards.append(reward)
        steps_list.append(steps)

    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps_list)
    logging.info(f"Average reward over {num_episodes} episodes: {avg_reward}")
    logging.info(f"Average steps over {num_episodes} episodes: {avg_steps}")

    env.close()

if __name__ == "__main__":
    main()