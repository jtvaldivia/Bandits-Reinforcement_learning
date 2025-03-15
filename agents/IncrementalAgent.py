import random

from agents.BaseAgent import BaseAgent


class IncrementalAgent(BaseAgent):
    """
    This agent learns from incremental rewards.
    """

    def __init__(self, num_of_actions: int, epsilon: float = 0.1):
        self.num_of_actions = num_of_actions
        self.__rewards = [0.0] * num_of_actions
        self.__counts = [0] * num_of_actions
        self.__epsilon = epsilon

    def get_action(self) -> int:
        if random.random() < self.__epsilon:
            return random.randrange(self.num_of_actions)
        return self.__counts.index(max(self.__counts))

    def learn(self, action, reward) -> None:
        self.__counts[action] += 1
        self.__rewards[action] += (1/self.__counts[action])* (reward - self.__rewards[action])