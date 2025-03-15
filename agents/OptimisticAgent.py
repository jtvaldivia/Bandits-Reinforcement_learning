from agents.NonStationaryAgent import NonStationaryAgent

class OptimisticAgent(NonStationaryAgent):
    """
    This agent does learn. Optimistic agent
    """
    
    def __init__(self, num_of_actions: int, epsilon: float = 0.1, alpha: float = 0.1, initial_q: float = 5):
        super().__init__(num_of_actions, epsilon, alpha)
        self.q_values = [initial_q] * num_of_actions
