import networkx as nx
import numpy as np
from typing import Callable, List, Optional

from .agent import Agent
from .precision import matrix, mp_zeros_matrix, to_float64_matrix


class SocialNetwork:
    def __init__(self, agents: List[Agent], distribution_function: Callable = None) -> None:
        """
        Represent a social network of agents and influence relationships.

        The network owns the opinion vector, influence matrix, and directed
        graph used for visualization. The distribution function is used by the
        dynamic influence update to assign weights from opinion differences.
        """
        self.agents = agents
        self.distribution_function = distribution_function

        self.n_agents = len(agents)
        self.graph = None
        self.node_positions = None

        self.sort_agents_by_index()
        self.init_opinion_vector()
        self.init_influence_matrix()
        self.update_graph()
        self.node_positions = nx.spring_layout(self.graph, seed=42)

    def sort_agents_by_index(self) -> None:
        self.agents.sort(key=lambda agent: agent.index)

    def init_opinion_vector(self) -> None:
        """Build the n x 1 opinion vector from the agents' current opinions."""
        self.opinion_vector = mp_zeros_matrix(self.n_agents, 1)
        for i, agent in enumerate(self.agents):
            self.opinion_vector[i, 0] = agent.get_opinion()

    def init_influence_matrix(self) -> None:
        """
        Build the influence matrix from each agent's influence vector.

        Row i contains the weights used to update agent i's opinion.
        """
        self.influence_matrix = mp_zeros_matrix(self.n_agents, self.n_agents)
        for i, agent in enumerate(self.agents):
            agent_row = agent.get_influence_of_others()
            for j in range(self.n_agents):
                self.influence_matrix[i, j] = agent_row[j, 0]

    def update_graph(self) -> None:
        """
        Build the directed graph used for visualization.

        The influence matrix is transposed because DeGroot rows represent how an
        agent is influenced by others, while NetworkX edges point from
        influencer to influenced agent.
        """
        float_matrix = to_float64_matrix(self.influence_matrix)
        self.graph = nx.from_numpy_array(float_matrix.T, create_using=nx.DiGraph)
        self.graph = nx.relabel_nodes(self.graph, lambda i: self.agents[i].index, copy=False)

    def get_influence_matrix(self) -> matrix:
        return self.influence_matrix

    def get_opinion_vector(self) -> matrix:
        return self.opinion_vector

    def get_graph(self) -> nx.DiGraph:
        return self.graph

    def get_node_positions(self) -> dict:
        return self.node_positions

    def set_influence_functions_of_agents(self, functions: List[Callable[..., float]]) -> None:
        """Assign the same influence-change function list to every agent."""
        for agent in self.agents:
            agent.set_influence_change_functions(functions)

    def set_distribution_function(self, distribution_function: Callable) -> None:
        """Set the function used by the dynamic influence update."""
        self.distribution_function = distribution_function

    def update_opinions(self) -> None:
        """
        Update opinions according to the DeGroot model.

        The influence matrix multiplies the current opinion vector. The result
        is stored in the network and copied back to each agent.
        """
        self.opinion_vector = self.influence_matrix * self.opinion_vector

        for i, agent in enumerate(self.agents):
            agent.set_opinion(self.opinion_vector[i, 0])
            agent.add_iteration()

    def update_influences(self, **kwargs) -> None:
        """
        Update the influence matrix using agents' explicit change functions.

        It is useful for recreating results such as those
        stated in Chatterjee and Seneta (1977) for open-minded agents.
        """
        for i, agent in enumerate(self.agents):
            agent.update_influence_of_others(**kwargs)
            agent_row = agent.get_influence_of_others()
            for j in range(self.n_agents):
                self.influence_matrix[i, j] = agent_row[j, 0]

    def update_influences_v2(self, last_opinion_vector: matrix) -> None:
        """
        Update the influence matrix from the previous opinion vector.

        Each agent recomputes its non-zero influence weights using the network's
        distribution function. This method is useful for implementing dynamic
        influence mechanisms driven by homophily.
        """
        for i, agent in enumerate(self.agents):
            agent.update_influence_of_others_v2(self.distribution_function, last_opinion_vector)
            agent_row = agent.get_influence_of_others()
            for j in range(self.n_agents):
                self.influence_matrix[i, j] = agent_row[j, 0]

    @staticmethod
    def generate_random_social_network(n_agents: int, seed: Optional[int] = None) -> 'SocialNetwork':
        """Generate a social network with random agents."""
        if seed is None:
            seed = np.random.randint(0, 1_000_000_000)
            print(f"Generated seed for social network: {seed}")

        agents = [Agent.generate_random_agent(index=i, n_agents=n_agents, seed=seed) for i in range(n_agents)]
        return SocialNetwork(agents=agents)
