import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from gymnasium import spaces

class Agent:
    def __init__(self, state_size, action_size, epsilon=0.99, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001, discount_factor=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.action_space = None
        self.memory = []  # Experience replay memory
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.type = 'deepq'
        self.games_played = 0
        self.games_won = 0
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            x = np.random.choice(self.action_space[0].n)
            y = np.random.choice(self.action_space[1].n)
            return (x, y)
        #state = np.reshape(state, [1, self.state_size])  # Ensure the state is reshaped correctly
        q_values = self.model.predict(state)
        action_index = np.argmax(q_values[0])
        x = action_index // self.action_space[1].n
        y = action_index % self.action_space[1].n
        return (x, y)

    def replay(self, batch_size):
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        for index in minibatch:
            state, action, reward, next_state, done = self.memory[index]
            #state = np.reshape(state, [1, self.state_size])
            #next_state = np.reshape(next_state, [1, self.state_size])
            target = self.model.predict(state)
            action_index = action[0] * self.action_space[1].n + action[1]
            if done:
                target[0][action_index] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action_index] = reward + self.discount_factor * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
        self.update_target_model()