

class Agent:
    def __init__(self) -> None:
        self.action_space = None
        self.games_played = 0
        self.games_won = 0
        self.type = 'random'

    def act(self, observation):
        return self.action_space.sample()