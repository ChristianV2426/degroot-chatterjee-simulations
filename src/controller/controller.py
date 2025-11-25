from model import SocialNetwork, Agent
import numpy as np
from typing import List


class Controller:
    def __init__(self, social_network: SocialNetwork, n_iterations: int) -> None:
        self.social_network = social_network
        self.n_iterations = n_iterations
        self.opinion_history: List[np.ndarray] = []
        self.influence_matrix_history: List[np.ndarray] = []
        self.backward_product: np.ndarray = None
        self.backward_product_history: List[np.ndarray] = []

        # Initialize backward_product and histories with initial state
        self.append_opinion_vector()
        self.append_influence_matrix()
        self.backward_product = np.eye(self.social_network.n_agents, dtype=np.float64)
    
    def append_opinion_vector(self) -> None:
        self.opinion_history.append(self.social_network.get_opinion_vector().copy())

    def append_influence_matrix(self) -> None:
        self.influence_matrix_history.append(self.social_network.get_influence_matrix().copy())

    def append_backward_product(self) -> None:
        self.backward_product_history.append(self.backward_product.copy())
    
    def run_simulation(self) -> None:
        """
        This method runs the DeGroot simulation for n_iterations.
        At each iteration, it updates the backward product, which is a piece of information that belongs to the Controller, not to the SocialNetwork.
        Then, it updates the opinions and influences in the SocialNetwork, and appends the new states to the histories.
        """
        for iteration in range(self.n_iterations):
            self.backward_product = self.social_network.get_influence_matrix() @ self.backward_product
            self.append_backward_product()

            self.social_network.update_opinions()
            self.social_network.update_influences()
            self.social_network.update_graph()

            self.append_opinion_vector()
            self.append_influence_matrix()

    def get_first_opinion_vector(self) -> np.ndarray:
        return self.opinion_history[0]
    
    def get_last_opinion_vector(self) -> np.ndarray:
        return self.opinion_history[-1]
    
    def get_opinion_history(self) -> List[np.ndarray]:
        return self.opinion_history
    
    def get_first_influence_matrix(self) -> np.ndarray:
        return self.influence_matrix_history[0]
    
    def get_last_influence_matrix(self) -> np.ndarray:
        return self.influence_matrix_history[-1]
    
    def get_influence_matrix_history(self) -> List[np.ndarray]:
        return self.influence_matrix_history
    
    def get_final_opinion_via_backward_product(self) -> np.ndarray:
        """
        It is possible to compute the final opinion vector by multiplying the initial opinion vector by the backward product.
        This is just an easy and alternative way to verify that the simulation was run correctly.
        """
        return self.backward_product @ self.get_first_opinion_vector()

    def print_current_network_graph(self, include_self_loops: bool = True) -> None:
        self.social_network.print_network_graph(include_self_loops=include_self_loops)

    def print_opinion_history(self) -> None:
        for i, opinion_vector in enumerate(self.opinion_history):
            print(f"Iteration {i}:")
            print(opinion_vector)
    
    def print_influence_matrix_history(self) -> None:
        for i, influence_matrix in enumerate(self.influence_matrix_history):
            print(f"Iteration {i}:")
            print(influence_matrix)

    def print_backward_product_history(self) -> None:
        for i, backward_product in enumerate(self.backward_product_history):
            print(f"Iteration {i}:")
            print(backward_product)
