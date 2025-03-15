import random

from agents.IncrementalAgent import IncrementalAgent


class NonStationaryAgent(IncrementalAgent):
    """
    This agent does learn. Incremental and Non stationary bandit
    """
    
    def __init__(self, num_of_actions: int, epsilon: float = 0.1, alpha: float = 0.1):
        super().__init__(num_of_actions, epsilon)
        self.alpha = alpha

    def learn(self, action, reward) -> None:
        self.action_counts[action] += 1
        self.q_values[action] += self.alpha * (reward - self.q_values[action])