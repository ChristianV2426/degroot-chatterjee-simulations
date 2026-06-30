import numpy as np
from typing import Callable, List, Optional

from .precision import (
    matrix,
    mp,
    mp_column,
    mp_copy,
    mp_from_numpy,
    mp_sum,
    mp_zeros,
    mpf,
)


class Agent:
    def __init__(
            self,
            index: int,
            opinion,
            influence_of_others,
            influence_change_functions: Optional[List[Callable[..., float]]] = None
            ) -> None:
        """
        Represent one agent in the social network.

        The agent stores its current opinion, the weights that determine how
        other agents influence it, and optional functions for time-varying
        influence. Numeric values are stored with mpmath precision.
        """
        self.index = index
        self.opinion = mpf(opinion)

        if isinstance(influence_of_others, np.ndarray):
            self.influence_of_others = mp_from_numpy(influence_of_others)
        elif isinstance(influence_of_others, matrix):
            self.influence_of_others = mp_copy(influence_of_others)
        else:
            self.influence_of_others = mp_column(influence_of_others)

        if self.influence_of_others.cols != 1:
            if self.influence_of_others.rows == 1:
                self.influence_of_others = self.influence_of_others.T
            else:
                raise ValueError(
                    "influence_of_others must be 1-D (row or column matrix)"
                )

        self.influence_change_functions = influence_change_functions
        self.iteration = 0
        self.agents_in_network = self.influence_of_others.rows
        self.normalize_influence()

    def get_index(self) -> int:
        return self.index

    def get_opinion(self) -> mpf:
        return self.opinion

    def set_opinion(self, new_opinion) -> None:
        self.opinion = mpf(new_opinion)

    def get_influence_of_others(self) -> matrix:
        return self.influence_of_others

    def get_influence_change_functions(self) -> Optional[List[Callable[..., float]]]:
        return self.influence_change_functions

    def set_influence_change_functions(self, functions: List[Callable[..., float]]) -> None:
        self.influence_change_functions = functions

    def get_iteration(self) -> int:
        return self.iteration

    def add_iteration(self) -> None:
        self.iteration += 1

    def normalize_influence(self, epsilon=None) -> None:
        """
        Normalize the influence vector so its entries sum to 1.

        If the vector is numerically zero at the current precision, the agent
        falls back to full self-influence.
        """
        if epsilon is None:
            epsilon = mpf(10) ** -(mp.dps - 4)

        total = mp_sum(self.influence_of_others)

        if total > epsilon:
            self.influence_of_others = self.influence_of_others / total
        else:
            fallback = mp_zeros(self.agents_in_network)
            fallback[self.index, 0] = mpf(1)
            self.influence_of_others = fallback

    def update_influence_of_others(self, **kwargs) -> None:
        """
        Update this agent's influence vector with explicit change functions.

        Each non-empty function receives the current iteration, this agent's
        index, the other agent's index, the current influence value, and any
        additional keyword arguments. This method is useful for recreating
        results such as those stated in Chatterjee and Seneta (1977) for
        open-minded agents.
        """
        if self.influence_change_functions is not None:
            for i in range(self.agents_in_network):
                func = self.influence_change_functions[i]
                if func is not None:
                    self.influence_of_others[i, 0] = func(
                        iteration=self.iteration,
                        own_index=self.index,
                        other_agent_index=i,
                        current_influence=self.influence_of_others[i, 0],
                        **kwargs
                    )
            self.normalize_influence()

    def update_influence_of_others_v2(self, distribution_function, last_opinion_vector) -> None:
        """
        Update influence weights from opinion differences.

        Existing zero entries remain zero. Non-zero entries are recomputed by
        applying the distribution function to the absolute difference between
        this agent's previous opinion and each neighbor's previous opinion.
        This method is useful for implementing dynamic influence mechanisms
        driven by homophily.
        """
        epsilon = mpf(10) ** -(mp.dps - 4)

        if abs(self.influence_of_others[self.index, 0] - mpf(1)) <= epsilon:
            return

        if distribution_function is not None:
            distribution_vector = mp_zeros(self.agents_in_network)

            for i in range(self.agents_in_network):
                if self.influence_of_others[i, 0] != 0:
                    difference_opinion = abs(last_opinion_vector[i, 0] - last_opinion_vector[self.index, 0])
                    distribution_vector[i, 0] = distribution_function(difference_opinion)

            self.influence_of_others = distribution_vector / mp_sum(distribution_vector)
            self.normalize_influence()

    @staticmethod
    def generate_random_agent(index: int, n_agents: int, seed: Optional[int] = None) -> 'Agent':
        """
        Generate an agent with a random opinion and random influence vector.

        The beta distribution gives initial opinions and weights close to the
        extremes, which is useful for polarized starting conditions.
        """
        if seed is not None:
            np.random.seed(seed + index)

        opinion = np.random.beta(0.5, 0.5)
        influence_of_others = np.random.beta(0.5, 0.5, size=n_agents).astype(np.float64)

        agent = Agent(
            index=index,
            opinion=opinion,
            influence_of_others=influence_of_others
        )
        return agent
