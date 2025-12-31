import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
from build_dataset_for_training import sample_episode_places

def select_action(state, visited, epsilon):
    if np.random.rand() < epsilon:
        valid = np.where(visited == 0)[0]
        return np.random.choice(valid)

    state = torch.FloatTensor(state).unsqueeze(0)
    q_values = policy_net(state).detach().numpy()[0]
    q_values[visited == 1] = -1e9
    return np.argmax(q_values)

from collections import deque
import random
from environment_route import RouteEnv
from model import DQN

STATE_DIM = 15
ACTION_DIM = 5
source_data_path = "data/cleaned_data.jsonl"

policy_net = DQN(STATE_DIM, ACTION_DIM)
target_net = DQN(STATE_DIM, ACTION_DIM)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
memory = deque(maxlen=20000)

gamma = 0.99
epsilon_min = 0.05
epsilon_decay = 0.995
batch_size = 64
target_update = 200

import os

checkpoint_path = "agent3_optimize_route/deep_q_learning/dqn_route_checkpoints.pt"

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    policy_net.load_state_dict(checkpoint["model"])
    target_net.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    epsilon = checkpoint["epsilon"]
    start_episode = checkpoint["episode"] + 1

    print(f"Resume training from episode {start_episode}")
else:
    epsilon = 1.0
    start_episode = 0
    print("Initial training")

def optimize():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    s, a, r, s2, d = zip(*batch)

    s = torch.FloatTensor(s)
    a = torch.LongTensor(a).unsqueeze(1)
    r = torch.FloatTensor(r)
    s2 = torch.FloatTensor(s2)
    d = torch.FloatTensor(d)

    q = policy_net(s).gather(1, a).squeeze()
    with torch.no_grad():
        q_next = target_net(s2).max(1)[0]
        q_target = r + gamma * q_next * (1 - d)

    loss = nn.MSELoss()(q, q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for episode in range(start_episode, start_episode + 1000):
    
    province_id, episode_places = sample_episode_places(
        source_data_path, k=5
    )
    
    env = RouteEnv(episode_places)

    state = env.reset()
    done = False

    while not done:
        action = select_action(state, env.visited, epsilon)
        next_state, reward, done, _ = env.step(action)

        memory.append((state, action, reward, next_state, done))
        state = next_state

        optimize()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

torch.save({
    "episode": episode,
    "model": policy_net.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epsilon": epsilon
}, checkpoint_path)

print(f"Saved checkpoint at episode {episode}")

# Test thử model đã train
policy_net.eval()

_, test_places = sample_episode_places(source_data_path, k=5)
env = RouteEnv(test_places)

state = env.reset()
done = False

while not done:
    action = select_action(state, env.visited, epsilon=0.0)
    state, _, done, _ = env.step(action)

route = [test_places[i]["destination_id"] for i in env.route]
print("Optimal route:", route)
