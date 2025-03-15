import random
from math import exp
from agents.BaseAgent import BaseAgent


class GradientBanditAlgorithm(BaseAgent):
    """
    This agent learns from incremental rewards.
    """

    def __init__(self, num_of_actions: int, alpha: float, baseline: bool):
        self.num_of_actions = num_of_actions
        self.__total_rewards = 0.0
        self.__average_rewards = 0.0
        self.__counts = [0] * num_of_actions
        self.preferencias = [0.0] * num_of_actions
        self.__alpha = alpha
        self.baseline = baseline


    def get_action(self) -> int:
        probs = self.get_prob_action()
        return random.choices(range(self.num_of_actions), weights=probs)[0]
        
        
    def learn(self, action, reward) -> None:
        self.__counts[action] += 1
        self.update_average_rewards(reward)
        for i in range(self.num_of_actions):
            if self.baseline:
                if i==action:
                    self.preferencias[i] += self.__alpha * (reward) * (1 - self.get_prob_action()[i])
                else:
                    self.preferencias[i] -= self.__alpha * (reward) * self.get_prob_action()[i]
            elif i==action:
                self.preferencias[i] += self.__alpha * (reward - self.__average_rewards) * (1 - self.get_prob_action()[i])
            else:
                self.preferencias[i] -= self.__alpha * (reward - self.__average_rewards) * self.get_prob_action()[i]
    
    def get_prob_action(self):
        softmax = [0.0] * self.num_of_actions
        sum_exp = sum([exp(p) for p in self.preferencias])
        for i in range(self.num_of_actions):
            softmax[i] = exp(self.preferencias[i])/sum_exp
        return softmax
    
    def update_average_rewards(self, reward):
        self.__total_rewards += reward
        total_counts = sum(self.__counts) 
        if total_counts > 0:
            self.__average_rewards = self.__total_rewards / total_counts
