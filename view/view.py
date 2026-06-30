import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from model.precision import matrix, mp, to_float64_matrix
from model.social_network import SocialNetwork


class View:
    """
    Handle visualization for simulations and derived data.

    The view plots opinion histories, renders network graphs, and displays graph
    animations. It does not modify simulation state.
    """

    @staticmethod
    def plot_opinion_history(opinion_history: List[matrix], n_agents: int, show_labels: bool = True) -> None:
        """
        Plot opinion trajectories for all agents.

        Opinion history is stored as mpmath matrices, so values are converted to
        float64 at the Matplotlib boundary.
        """
        opinions = np.hstack([to_float64_matrix(col) for col in opinion_history])
        n_iterations = len(opinion_history)
        iteration_range = range(n_iterations)

        plt.figure()
        for agent_index in range(n_agents):
            plt.plot(
                iteration_range,
                opinions[agent_index, :],
                label=f"Agent {agent_index}"
            )

        plt.xlim(0, n_iterations - 1)
        plt.ylim(0, 1)
        plt.xlabel("Time")
        plt.ylabel("Opinion")
        plt.title("Opinion Evolution Over Time")
        if show_labels:
            plt.legend(loc='best')
        plt.grid(True)

        plt.show()

    @staticmethod
    def plot_minimum_influence_history(minimum_influence_history: List, log_scale: bool = False) -> None:
        """
        Plot the smallest positive influence recorded at each iteration.

        Log scale is useful when the minimum influence becomes very small.
        """
        iteration_range = range(len(minimum_influence_history))

        if log_scale:
            values = [
                float(mp.log10(value)) if value is not None and value > 0 else np.nan
                for value in minimum_influence_history
            ]
            ylabel = "log10(Minimum positive influence)"
        else:
            values = [
                float(value) if value is not None else np.nan
                for value in minimum_influence_history
            ]
            ylabel = "Minimum positive influence"

        plt.figure()
        plt.plot(iteration_range, values)
        plt.xlabel("Time")
        plt.ylabel(ylabel)
        plt.title("Minimum Influence Over Time")
        plt.grid(True)
        plt.show()

    @staticmethod
    def display_network_graph(graph: nx.DiGraph, position: dict, include_self_loops: bool = False, filter_vertex: int = None, ax=None) -> None:
        """
        Draw a directed influence graph.

        Bidirectional edges are curved to reduce overlap. Self-loops can be
        hidden, and the graph can be filtered to show only edges directed toward
        a selected vertex.
        """
        if ax is None:
            fig, ax = plt.subplots()

        edge_labels = {edge: round(
            weight, 10) for edge, weight in nx.get_edge_attributes(graph, 'weight').items()}

        if include_self_loops:
            edges_to_draw = graph.edges()
        else:
            self_loops = set(nx.selfloop_edges(graph))
            edges_to_draw = [
                edge for edge in graph.edges() if edge not in self_loops]

        if filter_vertex is not None:
            edges_to_draw = [
                edge for edge in edges_to_draw
                if edge[1] == filter_vertex
            ]
            edge_labels = {
                edge: label for edge, label in edge_labels.items()
                if edge[1] == filter_vertex
            }

        nx.draw_networkx_nodes(graph, position, ax=ax)
        nx.draw_networkx_labels(graph, position, ax=ax, font_size=10)

        curved_edges = [edge for edge in edges_to_draw if (
            edge[1], edge[0]) in edges_to_draw]
        curved_edges_labels = {edge: label for edge,
                               label in edge_labels.items() if edge in curved_edges}
        nx.draw_networkx_edges(
            G=graph,
            pos=position,
            edgelist=curved_edges,
            connectionstyle='arc3, rad=0.15',
            ax=ax
        )
        nx.draw_networkx_edge_labels(
            G=graph,
            pos=position,
            edge_labels=curved_edges_labels,
            connectionstyle='arc3, rad=0.15',
            font_size=8,
            label_pos=0.5,
            ax=ax
        )

        straight_edges = list(set(edges_to_draw) - set(curved_edges))
        straight_edges_labels = {
            edge: label for edge, label in edge_labels.items() if edge in straight_edges}
        nx.draw_networkx_edges(
            G=graph,
            pos=position,
            edgelist=straight_edges,
            ax=ax
        )
        nx.draw_networkx_edge_labels(
            G=graph,
            pos=position,
            edge_labels=straight_edges_labels,
            font_size=8,
            label_pos=0.5,
            ax=ax
        )

    @staticmethod
    def display_network_graphs_animation(network_graphs: List[nx.DiGraph], social_network: SocialNetwork, include_self_loops: bool = False, filter_vertex: int = None) -> None:
        """
        Display an interactive graph animation.

        The left and right arrow keys navigate through the stored graph
        snapshots by iteration.
        """
        fig, ax = plt.subplots()
        position = social_network.get_node_positions()
        total_graphs = len(network_graphs)
        current_index = [0]

        def draw_graph(index: int) -> None:
            ax.clear()
            View.display_network_graph(
                graph=network_graphs[index],
                position=position,
                include_self_loops=include_self_loops,
                filter_vertex=filter_vertex,
                ax=ax
            )
            ax.set_title(f"Iteration {index}")
            fig.canvas.draw_idle()

        def on_key(event) -> None:
            if event.key == 'right':
                current_index[0] = (current_index[0] + 1) % total_graphs
            elif event.key == 'left':
                current_index[0] = (current_index[0] - 1) % total_graphs

            draw_graph(current_index[0])

        fig.canvas.mpl_connect('key_press_event', on_key)
        draw_graph(current_index[0])
        plt.show()
