from agents.IncrementalAgent import IncrementalAgent

class OptimisticAgent(IncrementalAgent):
    """
    This agent does learn. Optimistic agent
    """
    
    def __init__(self, num_of_actions: int, epsilon: float = 0.1, initial_q: float = 5):
        super().__init__(num_of_actions, epsilon)
        self.q_values = [initial_q] * num_of_actions
