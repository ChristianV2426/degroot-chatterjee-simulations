from model import SocialNetwork, Agent
from controller import Controller
import numpy as np


if __name__ == "__main__":
    agent0 = Agent(
        index=0,
        opinion=1,
        influence_of_others=np.array([0.9, 0, 0.1], dtype=np.float64))
    
    agent1 = Agent(
        index=1,
        opinion=0.8,
        influence_of_others=np.array([1, 0, 0], dtype=np.float64))
    
    agent2 = Agent(
        index=2,
        opinion=0,
        influence_of_others=np.array([0, 0.7, 0.3], dtype=np.float64))
    
    agents = [agent0, agent1, agent2]

    social_network = SocialNetwork(agents=agents)

    controller = Controller(social_network=social_network, n_iterations=10)

    controller.print_current_network_graph()

    controller.run_simulation()

    print("Opinion history:",)
    controller.print_opinion_history()

    print("\n\nFinal opinion via backward product:\n", controller.get_final_opinion_via_backward_product())

    print("\n\nInfluence matrix history:",)
    controller.print_influence_matrix_history() 

    print("\n\nBackward product history:",)
    controller.print_backward_product_history()
