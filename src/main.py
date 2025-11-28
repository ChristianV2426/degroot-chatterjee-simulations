from model import SocialNetwork, Agent
from controller import Controller
import numpy as np


if __name__ == "__main__":
    def equal_influence_function(iteration: int, current_influence: float, *args, **kwargs) -> float:
        """
        Adjust the influence of an agent so that it gradually converges to 1/n_agents over time.

        Parameters:
        - iteration (int): The current iteration number of the simulation.
        - current_influence (float): The current influence of another agent on the agent.
        - *args: Positional arguments (not used here).
        - **kwargs: Keyword arguments containing 'n_agents' and 'alpha'.
            - n_agents (int): Total number of agents in the social network.
            - alpha (float) is the adjustment rate (controls how quickly the influence converges to 1/n_agents).
        """
        n_agents = kwargs.get('n_agents')
        alpha = kwargs.get('alpha', 0.1)

        target_influence = 1.0 / n_agents
        adjustment = alpha * (target_influence - current_influence)
        return current_influence + adjustment
    
    n_agents = 3
    
    influence_chage_functions = [
        lambda iteration, current_influence, **kwargs: equal_influence_function(
            iteration=iteration, 
            current_influence=current_influence, 
            n_agents=n_agents,
            alpha=0.1
        ) 
        for _ in range(n_agents)
    ]

    # influence_chage_functions = None

    agent0 = Agent(
        index=0,
        opinion=1,
        influence_of_others=np.array([0.9, 0.2, 0.1], dtype=np.float64),
        influence_change_functions=influence_chage_functions)
    
    agent1 = Agent(
        index=1,
        opinion=0.8,
        influence_of_others=np.array([1, 1, 5], dtype=np.float64),
        influence_change_functions=influence_chage_functions)
    
    agent2 = Agent(
        index=2,
        opinion=0,
        influence_of_others=np.array([0.6, 0.7, 0.3], dtype=np.float64),
        influence_change_functions=influence_chage_functions)
    
    agents = [agent0, agent1, agent2]

    social_network = SocialNetwork(agents=agents)

    controller = Controller(social_network=social_network, n_iterations=10)

    # controller.print_current_network_graph()

    controller.run_simulation()
    controller.display_network_graphs_animation(include_self_loops=True)

    # print("Opinion history:",)
    # controller.print_opinion_history()

    # print("\n\nFinal opinion via backward product:\n", controller.get_final_opinion_via_backward_product())

    # print("\n\nInfluence matrix history:",)
    # controller.print_influence_matrix_history() 

    # print("\n\nBackward product history:",)
    # controller.print_backward_product_history()

    # controller.print_current_network_graph()

    controller.plot_opinion_history()
