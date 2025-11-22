import numpy as np
from numpy.typing import NDArray
from typing import Callable, Optional


class Agent:

    def __init__(
            self, 
            index: int,
            opinion: float,
            influence_of_others: NDArray[np.float64],
            influence_change_function: Optional[Callable[..., NDArray[np.float64]]] = None
            ) -> None:
        self.index = index
        self.opinion = opinion
        self.influence_of_others = influence_of_others
        self.influence_change_function = influence_change_function

        self.iteration = 0
        self.agents_in_network = influence_of_others.shape[0]
        self.normalize_influence()

    def normalize_influence(self, epsilon: float = 1e-8) -> None:
        """
        Ensure self.influence_of_others sums to 1.

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