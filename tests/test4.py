from model import SocialNetwork, Agent
from controller import Controller
import numpy as np


if __name__ == "__main__":
    def stubborn_influence_function(iteration: int, current_influence: float, own_index: int, other_agent_index: int, *args, **kwargs) -> float:
        """
        Adjust the influence of an agent to make it stubborn, maintaining high self-influence and low influence from others.

        Parameters:
        - iteration (int): The current iteration number of the simulation.
        - current_influence (float): The current influence of another agent on the agent.
        - own_index (int): The index of the agent whose influence is being adjusted.
        - other_agent_index (int): The index of the other agent whose influence is being considered.
        - args: Positional arguments (not used here).
        - kwargs: Keyword arguments containing 'alpha'.
            - alpha (float) is the adjustment rate (controls how quickly the influence approaches the stubborn values).
        
        When own_index == other_agent_index, the target influence is 1.0 (self-influence). If the current_influence is already 1.0, it decreases slightly to allow the asymptotic approach to happen.
        When own_index != other_agent_index, the target influence is 0.0 (influence from others). If the current_influence is already 0.0, it increases slightly to allow the asymptotic approach to happen.
        """
        alpha = kwargs.get('alpha', 0.1)

        if own_index == other_agent_index:
            if current_influence == 1.0:
                adjustment = -0.009
            else:
                adjustment = alpha * (1.0 - current_influence)
        else:
            if current_influence == 0.0:
                adjustment = 0.009
            else:
                adjustment = alpha * (0.0 - current_influence)
        
        return current_influence + adjustment
    
    n_agents = 3

    influence_chage_functions = [
        lambda iteration, current_influence, own_index, other_agent_index, **kwargs: stubborn_influence_function(
            iteration=iteration, 
            current_influence=current_influence, 
            own_index=own_index,
            other_agent_index=other_agent_index,
            n_agents=n_agents,
            alpha=0.001
        ) 
        for _ in range(n_agents)
    ]
    # influence_chage_functions = None
    
    agent0 = Agent(
        index=0,
        opinion=0.1,
        influence_of_others=np.array([1.0, 0.0, 0.0], dtype=np.float64)
    )

    agent1 = Agent(
        index=1,
        opinion=0.5,
        influence_of_others=np.array([0, 1, 0.0], dtype=np.float64)
    )

    agent2 = Agent(
        index=2,
        opinion=0.9,
        influence_of_others=np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )

    social_network = SocialNetwork(agents=[agent0, agent1, agent2])
    social_network.set_influence_functions_of_agents(influence_chage_functions)
    controller = Controller(social_network=social_network, n_iterations=150)
    controller.run_simulation()
    controller.display_network_graphs_animation(include_self_loops=True)
    controller.plot_opinion_history()
    print(controller.get_last_opinion_vector())