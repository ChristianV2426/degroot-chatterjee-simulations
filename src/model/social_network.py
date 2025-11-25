import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from model import Agent


class SocialNetwork:
    def __init__(self, agents: List[Agent]) -> None:
        self.agents = agents

        self.n_agents = len(agents)
        self.graph = None

        self.sort_agents_by_index()
        self.init_opinion_vector()
        self.init_influence_matrix()
        self.update_graph()

    def sort_agents_by_index(self) -> None:
        self.agents.sort(key=lambda agent: agent.index)

    def init_opinion_vector(self) -> None:
        self.opinion_vector = np.zeros(self.n_agents, dtype=np.float64).reshape((-1, 1))
        self.opinion_vector[:, 0] = np.fromiter(
            (agent.get_opinion() for agent in self.agents),
            dtype=self.opinion_vector.dtype,
            count=self.n_agents            
        )

    def init_influence_matrix(self) -> None:
        self.influence_matrix = np.zeros((self.n_agents, self.n_agents), dtype=np.float64)
        self.influence_matrix[:] = np.vstack([agent.get_influence_of_others() for agent in self.agents])

    def update_graph(self) -> None:
        """
        This method creates a directed graph from the influence matrix. It's necessary to transpose the matrix because in DeGroot model,
        rows represent how an agent is influenced by others, while in NetworkX, edges go from influencer to influencee.
        """
        self.graph = nx.from_numpy_array(self.influence_matrix.T, create_using=nx.DiGraph)
        self.graph = nx.relabel_nodes(self.graph, lambda i: self.agents[i].index, copy=False)

    def get_influence_matrix(self) -> np.ndarray:
        return self.influence_matrix
    
    def get_opinion_vector(self) -> np.ndarray:
        return self.opinion_vector

    def print_network_graph(self, include_self_loops: bool = True) -> None:
        """
        This method prints the network graph using Matplotlib and NetworkX.
        It is necesaary to make sure when two agents influence each other, the edges are drawn with curves to avoid overlapping.
        In the case of one-directional influence, straight edges are drawn.
        It is also possible to exclude self-loops from the visualization by setting include_self_loops to False.
        """
        fig, ax = plt.subplots()

        # Define the positions of nodes using a layout algorithm
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Get edge weights for labeling
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')

        # Get edges to draw
        if include_self_loops:
            edges_to_draw = self.graph.edges()
        else:
            # Exclude self-loops
            self_loops = set(nx.selfloop_edges(self.graph))
            edges_to_draw = [edge for edge in self.graph.edges() if edge not in self_loops]

        # Draw nodes and their labels
        nx.draw_networkx_nodes(self.graph, pos, ax=ax)
        nx.draw_networkx_labels(self.graph, pos, ax=ax, font_size=10)

        # Draw curved edges and their labels
        curved_edges = [edge for edge in edges_to_draw if (edge[1], edge[0]) in edges_to_draw]
        curved_edges_labels = {edge: label for edge, label in edge_labels.items() if edge in curved_edges}
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=curved_edges,
            connectionstyle='arc3, rad=0.15',
            ax=ax
        )
        nx.draw_networkx_edge_labels(
            self.graph,
            pos,
            edge_labels=curved_edges_labels,
            connectionstyle='arc3, rad=0.15',
            font_size=8,
            label_pos=0.5,
            ax=ax
        )
        
        # Draw straight edges and their labels
        straight_edges = list(set(edges_to_draw) - set(curved_edges))
        straight_edges_labels = {edge: label for edge, label in edge_labels.items() if edge in straight_edges}
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=straight_edges,
            ax=ax
        )
        nx.draw_networkx_edge_labels(
            self.graph,
            pos,
            edge_labels=straight_edges_labels,
            font_size=8,
            label_pos=0.5,
            ax=ax
        )

        plt.tight_layout()
        plt.show()
    

    def update_opinions(self) -> None:
        """
        This method updates the opinion vector according to the DeGroot model:
        opinion_vector = influence_matrix @ opinion_vector
        After updating the opinion vector, which resides in the SocialNetwork, each agent is informed of its new opinion.
        """
        self.opinion_vector = self.influence_matrix @ self.opinion_vector

        # Let each agent know what their new opinion is
        for i, agent in enumerate(self.agents):
            agent.set_opinion(self.opinion_vector[i, 0])
            agent.add_iteration()
        
    
    def update_influences(self, **kwargs) -> None:
        """
        This method updates the influence matrix by asking each agent to update its influence_of_others vector.
        After updating each agent's influence_of_others, the influence matrix is reconstructed and the graph
        representation of the social network is updated.
        """
        for i, agent in enumerate(self.agents):
            agent.update_influence_of_others(**kwargs)
            self.influence_matrix[i, :] = agent.get_influence_of_others()
        self.update_graph()
        
