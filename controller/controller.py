from collections import deque
from typing import List

from model import SocialNetwork
from model.precision import matrix, mp, mp_copy, mp_eye
from view import View


class Controller:
    def __init__(self, social_network: SocialNetwork, n_iterations: int) -> None:
        """
        Coordinate the simulation between the model and the view.

        The controller stores opinion history, recent influence matrices, graph
        snapshots for animation, and backward products for analytical checks.
        """
        self.social_network = social_network
        self.n_iterations = n_iterations
        self.opinion_history: List[matrix] = []
        self.influence_matrix_history: deque = deque(maxlen=10)
        self.backward_product: matrix = None
        self.network_graphs: List = []
        self.backward_product_history: List[matrix] = []

        self.append_opinion_vector()
        self.append_influence_matrix()
        self._initial_influence_matrix: matrix = self.influence_matrix_history[0]
        self.append_network_graph()
        self.backward_product = mp_eye(self.social_network.n_agents)

    def append_opinion_vector(self) -> None:
        """Store a copy of the current opinion vector."""
        self.opinion_history.append(mp_copy(self.social_network.get_opinion_vector()))

    def append_influence_matrix(self) -> None:
        """Store a copy of the current influence matrix."""
        self.influence_matrix_history.append(mp_copy(self.social_network.get_influence_matrix()))

    def append_network_graph(self) -> None:
        """Store a copy of the current visualization graph."""
        self.network_graphs.append(self.social_network.get_graph().copy())

    def append_backward_product(self) -> None:
        """Store a copy of the current backward product."""
        self.backward_product_history.append(mp_copy(self.backward_product))

    def _format_influence_matrix(self, mat: matrix) -> str:
        """Format an influence matrix using the current mpmath precision."""
        rows = []
        for i in range(self.social_network.n_agents):
            row = [
                mp.nstr(mat[i, j], 200)
                for j in range(self.social_network.n_agents)
            ] 
            rows.append(" ".join(row))
        return "\n".join(rows)

    def run_simulation(self) -> None:
        """
        Run the standard DeGroot simulation.

        Opinions are updated from the current influence matrix, then influence
        weights are updated through each agent's explicit change functions. This
        method is useful for recreating results such as those stated in
        Chatterjee and Seneta (1977) for open-minded agents.
        """
        for _ in range(self.n_iterations):
            # # At each iteration, we can update the backward product, which is a piece of information that belongs to the Controller, not to the SocialNetwork. This is useful for analytical checks, but it is not strictly necessary for the simulation itself.
            # self.backward_product = self.social_network.get_influence_matrix() * self.backward_product
            # self.append_backward_product()

            self.social_network.update_opinions()
            self.social_network.update_influences()
            self.social_network.update_graph()

            self.append_opinion_vector()
            self.append_influence_matrix()
            self.append_network_graph()

    def run_simulation_v2(self) -> None:
        """
        Run the DeGroot simulation with dynamic influence weights.

        Opinions are updated first. Then non-zero influence weights are
        recomputed from the previous opinion vector and the network's
        distribution function. This method is useful for implementing dynamic
        influence mechanisms driven by homophily.
        """
        for _ in range(self.n_iterations):
            # # At each iteration, we can update the backward product, which is a piece of information that belongs to the Controller, not to the SocialNetwork. This is useful for analytical checks, but it is not strictly necessary for the simulation itself.
            # self.backward_product = self.social_network.get_influence_matrix() * self.backward_product
            # self.append_backward_product()

            self.social_network.update_opinions()
            self.social_network.update_influences_v2(self.opinion_history[-1])
            self.social_network.update_graph()

            self.append_opinion_vector()
            self.append_influence_matrix()
            self.append_network_graph()

    def plot_opinion_history(self, show_labels: bool = True) -> None:
        """Plot the stored opinion history."""
        View.plot_opinion_history(self.opinion_history, self.social_network.n_agents, show_labels=show_labels)

    def display_network_graphs_animation(self, include_self_loops: bool = False, filter_vertex: int = None) -> None:
        """Display an interactive animation of the stored network graphs."""
        View.display_network_graphs_animation(
            self.network_graphs,
            self.social_network,
            include_self_loops=include_self_loops,
            filter_vertex=filter_vertex,
        )

    def get_first_opinion_vector(self) -> matrix:
        return self.opinion_history[0]

    def get_last_opinion_vector(self) -> matrix:
        return self.opinion_history[-1]

    def get_opinion_history(self) -> List[matrix]:
        return self.opinion_history

    def get_first_influence_matrix(self) -> matrix:
        return self._initial_influence_matrix

    def get_last_influence_matrix(self) -> matrix:
        return self.influence_matrix_history[-1]

    def get_influence_matrix_history(self) -> List[matrix]:
        return list(self.influence_matrix_history)

    def get_backward_product_history(self) -> List[matrix]:
        return self.backward_product_history

    def get_final_opinion_via_backward_product(self) -> matrix:
        """
        Compute the final opinion vector from the initial vector and backward product.

        This is an analytical cross-check for the standard simulation path.
        """
        return self.backward_product * self.get_first_opinion_vector()

    def print_current_network_graph(self, include_self_loops: bool = False) -> None:
        """Display the last stored network graph."""
        graph = self.network_graphs[-1]
        position = self.social_network.get_node_positions()
        View.display_network_graph(graph, position, include_self_loops)

    def print_opinion_history(self) -> None:
        """Print the stored opinion vector for each iteration."""
        for i, opinion_vector in enumerate(self.opinion_history):
            print(f"Iteration {i}:")
            print(opinion_vector)

    def print_influence_matrix_history(self) -> None:
        """Print the stored influence matrices."""
        for i, influence_matrix in enumerate(self.influence_matrix_history):
            print(f"Iteration {i}:")
            print(self._format_influence_matrix(influence_matrix))
            print()

    def print_last_influence_matrices(self, n: int = 5) -> None:
        """Print the last n stored influence matrices."""
        entries = list(self.influence_matrix_history)[-n:]
        last_iter = len(self.opinion_history) - 1
        first_shown_iter = last_iter - len(entries) + 1
        for i, influence_matrix in enumerate(entries):
            print(f"Iteration {first_shown_iter + i}:")
            print(self._format_influence_matrix(influence_matrix))
            print()

    def print_backward_product_history(self) -> None:
        """Print the stored backward product matrices."""
        for i, backward_product in enumerate(self.backward_product_history):
            print(f"Iteration {i}:")
            print(backward_product)
