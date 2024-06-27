import numpy as np
from gymnasium import spaces
import pickle

class Agent:
    def __init__(self, epsilon=0.99, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.1, discount_factor=0.99):
        self.q_table = {}  # Q-table for storing state-action values
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.action_space = None
        self.total_reward = 0  # Initialize total reward
        self.games_played = 0  # Initialize games played
        self.games_won = 0  # Initialize games won
        self.type = 'qlearning'


    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)
    
    def set_q_value(self, state, action, value):
        self.q_table[(state, action)] = value

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        
        q_values = []
        if isinstance(self.action_space, spaces.Discrete):
            q_values = [self.get_q_value(state, a) for a in range(self.action_space.n)]
        elif isinstance(self.action_space, spaces.Tuple):
            # For Tuple spaces, assume the first space is Discrete and the second space is also Discrete
            q_values = [self.get_q_value(state, (a, b)) for a in range(self.action_space[0].n) for b in range(self.action_space[1].n)]
        else:
            raise ValueError("Unsupported action space type")

        max_q_value_index = np.argmax(q_values)
        if isinstance(self.action_space, spaces.Tuple):
            return (max_q_value_index // self.action_space[1].n, max_q_value_index % self.action_space[1].n)
        return max_q_value_index

    def learn(self, state, action, reward, next_state, done):
        if isinstance(self.action_space, spaces.Discrete):
            best_next_action = np.argmax([self.get_q_value(next_state, a) for a in range(self.action_space.n)])
        elif isinstance(self.action_space, spaces.Tuple):
            best_next_action = np.argmax([self.get_q_value(next_state, (a, b)) for a in range(self.action_space[0].n) for b in range(self.action_space[1].n)])

        target = reward
        if not done:
            target += self.discount_factor * self.get_q_value(next_state, best_next_action)
        
        q_update = self.learning_rate * (target - self.get_q_value(state, action))
        new_q_value = self.get_q_value(state, action) + q_update
        self.set_q_value(state, action, new_q_value)

        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save_model(self, filepath):
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'games_played': self.games_played,
            'games_won': self.games_won,
            'epsilon_decay' : self.epsilon_decay,
            'epsilon_min' : self.epsilon_min,
            'learning_rate' : self.learning_rate,
            'discount_factor' : self.discount_factor
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']
            self.games_played = data['games_played']
            self.games_won = data['games_won']
            self.epsilon_decay = data['epsilon_decay']
            self.epsilon_min = data['epsilon_min']
            self.learning_rate = data['learning_rate']
            self.discount_factor = data['discount_factor']