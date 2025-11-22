from model import SocialNetwork, Agent
import numpy as np


if __name__ == "__main__":
    agent0 = Agent(
        index=0,
        opinion=0.5,
        influence_of_others=np.array([0.1, 0.1, 0.6], dtype=np.float64))
    
    agent1 = Agent(
        index=1,
        opinion=0.8,
        influence_of_others=np.array([0.5, 0.5, 0], dtype=np.float64))
    
    agent2 = Agent(
        index=2,
        opinion=0.2,
        influence_of_others=np.array([0.6, 0.6, 0.6], dtype=np.float64))
    
    agents = [agent0, agent1, agent2]

    social_network = SocialNetwork(agents=agents)
    print(social_network.get_influence_matrix())
    print(social_network.get_opinion_vector())
    social_network.print_network_graph(include_self_loops=True)
