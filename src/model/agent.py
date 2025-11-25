import numpy as np
from numpy.typing import NDArray
from typing import Callable, Optional, List


class Agent:

    def __init__(
            self, 
            index: int,
            opinion: float,
            influence_of_others: NDArray[np.float64],
            influence_change_functions: Optional[List[Callable[..., float]]] = None
            ) -> None:
        self.index = index
        self.opinion = opinion
        self.influence_of_others = influence_of_others
        self.influence_change_functions = influence_change_functions

        self.iteration = 0
        self.agents_in_network = influence_of_others.shape[0]
        self.normalize_influence()
    
    def get_opinion(self) -> float:
        return self.opinion
    
    def set_opinion(self, new_opinion: float) -> None:
        self.opinion = new_opinion

    def get_influence_of_others(self) -> NDArray[np.float64]:
        return self.influence_of_others
    
    def add_iteration(self) -> None:
        self.iteration += 1

    def normalize_influence(self, epsilon: float = 1e-8) -> None:
        """
        This method ensures self.influence_of_others sums to 1.

        Assumptions:
        - self.influence_of_others is a 1-D numpy array of non-negative floats.
        - self.index is the agent's own position in that vector, and it is a valid index in self.influence_of_others.

        Steps:
        1) Compute total = sum(arr).
        2) If total is very close to 1 (within rel_tol), do nothing.
        3) If total > eps, divide the vector by total (this makes the sum exactly 1).
        4) If total <= eps (zero or near-zero), create a fallback vector where the agent has full influence on itself.
        """
        total = self.influence_of_others.sum()

        if abs(total - 1.0) <= epsilon:
            return
        
        elif total > epsilon:
            self.influence_of_others /= total
        
        else:
            fallback = np.zeros(self.agents_in_network, dtype=np.float64)
            fallback[self.index] = 1.0
            self.influence_of_others = fallback

    def update_influence_of_others(self, **kwargs) -> None:
        """
        This method updates the agent's influence_of_others vector by applying the corresponding influence change functions.
        
        Steps:
        1) Check if a vector of influence change functions is provided. If not, do nothing.
        2) For each agent in the network, check if there is a corresponding influence change function (The vector may contain None entries, which indicates that the influence of that other agent on this agent never changes).
        3) If there is a function, update the influence of that agent on this agent by calling the function with the current iteration number, the current influence value, and any additional keyword arguments provided.
        4) After updating all influences, normalize the influence_of_others vector to ensure it sums to 1.
        """
        if self.influence_change_functions is not None:
            for i in range(self.agents_in_network):
                func = self.influence_change_functions[i]
                if func is not None:
                    self.influence_of_others[i] = func(
                        iteration=self.iteration,
                        current_influence=self.influence_of_others[i],
                        **kwargs
                    )
            self.normalize_influence()
            