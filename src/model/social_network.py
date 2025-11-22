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
        self.init_influence_matrix()
        self.init_opinion_vector()
        self.update_graph()


    def sort_agents_by_index(self) -> None:
        self.agents.sort(key=lambda agent: agent.index)


    def init_influence_matrix(self) -> None:
        self.influence_matrix = np.zeros((self.n_agents, self.n_agents), dtype=np.float64)
        self.influence_matrix[:] = np.vstack([agent.influence_of_others for agent in self.agents])
    

    def init_opinion_vector(self) -> None:
        self.opinion_vector = np.zeros(self.n_agents, dtype=np.float64).reshape((-1, 1))
        self.opinion_vector[:, 0] = np.fromiter(
            (agent.opinion for agent in self.agents),
            dtype=self.opinion_vector.dtype,
            count=self.n_agents            
        )


    def update_graph(self) -> None:
        self.graph = nx.from_numpy_array(self.influence_matrix.T, create_using=nx.DiGraph)
        self.graph = nx.relabel_nodes(self.graph, lambda i: self.agents[i].index, copy=False)


    def get_influence_matrix(self) -> np.ndarray:
        return self.influence_matrix
    

    def get_opinion_vector(self) -> np.ndarray:
        return self.opinion_vector


    def print_network_graph(self, include_self_loops: bool = True) -> None:
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
