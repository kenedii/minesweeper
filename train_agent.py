from Minesweeper_ENV import Minesweeper
from RandomAgent import Agent as random
from Q_Learning import Agent as qlearning
from DeepQ_Agent import Agent as deepq
import time
import numpy as np

num_episodes = 1000 # Number of episodes (games) to train the agent
render = True # Set to True to render the game, False to run in the background
save = True # Whether to save the model after training
saved_model_path = 'dqn_model.keras' # Path to save the model
save_after_every = 50 # Save the model after every n episodes

agent = deepq(20, 1)
env = Minesweeper(render=render,agent = agent)
env.reset()
#agent = deepq(len(env.observation), len(env.action_space))


def train_agents(env, agent, num_episodes):
    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        agent.action_space = env.action_space
        agent.games_played += 1
        print('Episode:', episode)

        while not done: # Main game loop
            if render:
                env.render()
            if agent.type == 'random':
                observation, reward, done, info = random_agent_act(observation, agent, env)
            elif agent.type == 'qlearning':
                observation, reward, done, info = qlearning_agent_act(observation, agent, env)
            elif agent.type == 'deepq':
                observation, reward, done, info = deepq_agent_act(observation, agent, env)
        if done: # If the game is over
            if env.total_reward > 0: # If the agent wins
                agent.games_won += 1
                print(f"Episode {episode} won after {env.turns_taken} turns. Winrate: {agent.games_won/agent.games_played}")
            else:
                print(f"Episode {episode} lost. Winrate: {agent.games_won/agent.games_played}")
            if save and episode % save_after_every == 0: # Save the model after every n episodes
                save_agent(agent, saved_model_path)
    return agent

def save_agent(agent, saved_model_path):
    agent.save_model(saved_model_path)
    print(f"Model saved to {saved_model_path}")

def random_agent_act(observation, agent, env):
    action = agent.act(observation)       # Get the random action from the agent
    observation, reward, done, info = env.step(action) # Take the action in the environment
    return observation, reward, done, info

def qlearning_agent_act(observation, agent, env):
    # Convert the observation to a tuple to use as a state in the Q-table
    state = observation
    # Agent chooses an action
    action = agent.choose_action(state)
    # Environment processes the action
    next_observation, reward, done, info = env.step(action)
    # Convert the next observation to a tuple to use as a state in the Q-table
    next_state = next_observation 
    # Agent learns from the experience
    agent.learn(state, action, agent.total_reward, next_state, done)
    return next_observation, reward, done, info

def deepq_agent_act(observation, agent, env):
    action = agent.choose_action(observation)
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(next_state, [1, len(next_state)])
    agent.remember(observation, action, reward, next_state, done)
    return next_state, reward, done, {}


train_agents(env, agent, num_episodes) # Start the training process