import os
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from src.environment_route import RouteEnv
from src.model import DQN
from src.build_dataset_for_training import sample_episode_places


# places = [
#     {"destination_id": "p0", "lat": 21.03, "lng": 105.85},
#     {"destination_id": "p1", "lat": 21.01, "lng": 105.84},
#     {"destination_id": "p2", "lat": 21.05, "lng": 105.86},
#     {"destination_id": "p3", "lat": 21.00, "lng": 105.83},
#     {"destination_id": "p4", "lat": 21.04, "lng": 105.82}
# ]
source_data_path = "data/cleaned_data.jsonl"
_, test_places = sample_episode_places(source_data_path, k=5)

def optimize_route(test_places, checkpoint_path = "agent3_optimize_route/deep_q_learning/dqn_route_checkpoints.pt"):

    def select_action(state, visited, epsilon):
        if np.random.rand() < epsilon:
            valid = np.where(visited == 0)[0]
            return np.random.choice(valid)

        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = policy_net(state).detach().numpy()[0]
        q_values[visited == 1] = -1e9
        return np.argmax(q_values)


    env = RouteEnv(test_places)    
    STATE_DIM = 15
    ACTION_DIM = 5

    policy_net = DQN(STATE_DIM, ACTION_DIM)
    # Load trained model
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        policy_net.load_state_dict(checkpoint["model"])

    # Test thử model đã train
    state = env.reset()
    done = False

    while not done:
        action = select_action(state, env.visited, epsilon=0.0)
        state, _, done, _ = env.step(action)

    route = [test_places[i]["destination_id"] for i in env.route]
    print("Optimal route:", route)


optimize_route(test_places)