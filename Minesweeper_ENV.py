import random, pygame, sys
from pygame.locals import *
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import Minesweeper as ms

class Minesweeper(gym.Env):
    def __init__(self, render=False, agent=None) -> None:
        super().__init__()
        self.mineField = None
        self.zeroListXY = []
        self.markedMines = []
        self.revealedBoxes = []
        self.turns_taken = 0
        self.total_reward = 0

        self.observation = None
        self.action_space = spaces.Tuple((spaces.Discrete(ms.FIELDWIDTH), spaces.Discrete(ms.FIELDHEIGHT)))
        self.agent = agent
        self.renderOn = render

    def reset(self):
        self.mineField, self.zeroListXY, self.revealedBoxes, self.markedMines = ms.gameSetup()
        self.turns_taken = 0
        self.total_reward = 0
        self.observation = self.get_observation()
        if self.renderOn:
            self.render(init=True)
        return self.get_observation()
    
    def step(self, action):
        self.turns_taken += 1        
        box_x, box_y = action

        if self.revealedBoxes[box_x][box_y] == True:
            observation, reward, done = self.get_observation(), self.get_reward(negative_reward=True), self.check_done()
            return observation, reward, done, {}

        self.revealedBoxes[box_x][box_y] = True

        # when 0 is revealed, show relevant boxes
        if self.mineField[box_x][box_y] == '[0]':
            ms.showNumbers(self.revealedBoxes, self.mineField, box_x, box_y, self.zeroListXY)
            reward = self.get_reward()
            done = self.check_done()
            observation = self.get_observation()
            return observation, reward, done, {}

        # when mine is revealed, show mines
        if self.mineField[box_x][box_y] == '[X]':
            ms.showMines(self.revealedBoxes, self.mineField, box_x, box_y)
            ms.gameOverAnimation(self.mineField, self.revealedBoxes, self.markedMines, 'LOSS')
            reward = self.get_reward(negative_reward='loss')
            self.total_reward = 0
            done = self.check_done(gameLost=True)
            observation = self.get_observation()
            return observation, reward, done, {}

        # If neither condition is met, we need to return a default observation, reward, done, and info
        observation = self.get_observation()
        reward = self.get_reward()
        done = self.check_done()
        return observation, reward, done, {}


    def render(self, init=False):
        if init:
            ms.init_render()
            self.renderOn == 'rendered'
        ms.render(self.mineField, self.revealedBoxes, self.markedMines)
        return

    def get_observation(self):
        if self.agent.type == 'deepq':
            observation = []
            for x in range(ms.FIELDWIDTH):
                for y in range(ms.FIELDHEIGHT):
                    if self.revealedBoxes[x][y]:
                        observation.append(self.mineField[x][y])
                    else:
                        observation.append('hidden')
            return observation
        else:
            # Create a new empty tuple to store the observation
            observation_tuple = tuple()
            for x in range(ms.FIELDWIDTH):
                # Create a sub-tuple for each row
                row_tuple = tuple()
                for y in range(ms.FIELDHEIGHT):
                    if self.revealedBoxes[x][y]:
                        row_tuple += (self.mineField[x][y],)  # Add element to the row_tuple
                # Add the row_tuple to the main observation_tuple
                observation_tuple += (row_tuple,)
            return observation_tuple
    
    def get_reward(self, negative_reward=False):
        if negative_reward: # If the agent selects a tile they already selected
            return -1
        elif negative_reward == 'loss':
            return self.total_reward*-1
        reward =0
        for x in range(ms.FIELDWIDTH):
            for y in range(ms.FIELDHEIGHT):
                if self.revealedBoxes[x][y] == True:
                    reward += 1
        reward_t = reward
        reward = reward - self.total_reward
        self.total_reward = reward_t
        return reward

    def check_done(self, gameLost=False):
        if gameLost:
            self.mineField, self.zeroListXY, self.revealedBoxes, self.markedMines = ms.gameSetup()
            #ms.terminate()
            return True
        if ms.gameWon(self.revealedBoxes, self.mineField):
            return True
        return False