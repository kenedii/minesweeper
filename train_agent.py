from Minesweeper_ENV import Minesweeper
from RandomAgent import Agent as random
import time

num_episodes = 1000 # Number of episodes (games) to train the agent
render = True # Set to True to render the game, False to run in the background

env = Minesweeper(render=render)
agent = random()


def train_agents(env, agent, num_episodes):
    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        agent.action_space = env.action_space
        agent.games_played += 1
        print('Episode:', episode)

        while not done:
            env.render()
            if agent.type == 'random':
                random_agent_act(observation, agent, env)
                time.sleep(1)
        if done:
            print('game complete')
            if env.total_reward > 0: # If the agent wins
                agent.games_won += 1
                print(f"Episode {episode} won after {env.turns_taken} turns. Winrate: {agent.games_won/agent.games_played}")
            else:
                print(f"Episode {episode} lost. Winrate: {agent.games_won/agent.games_played}")
    return agent

def random_agent_act(observation, agent, env):
    action = agent.act(observation)       # Get the random action from the agent
    observation, reward, done, info = env.step(action) # Take the action in the environment
    return observation, reward, done, info

train_agents(env, agent, num_episodes)