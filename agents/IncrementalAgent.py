import random

from agents.BaseAgent import BaseAgent


class IncrementalAgent(BaseAgent):
    """
    This agent does learn. Incremental bandit
    """

    def __init__(self, num_of_actions: int, epsilon:float ):
        self.num_of_actions = num_of_actions
        self.q_values = [0] * num_of_actions
        self.action_counts = [0] * num_of_actions
        self.epsilon = epsilon

    def get_action(self) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.num_of_actions)
        return self.q_values.index(max(self.q_values))

    def learn(self, action, reward) -> None:
        self.action_counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]
