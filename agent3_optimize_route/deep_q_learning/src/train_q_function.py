import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

places = [
    {"id": "p0", "lat": 21.03, "lng": 105.85},
    {"id": "p1", "lat": 21.01, "lng": 105.84},
    {"id": "p2", "lat": 21.05, "lng": 105.86},
    {"id": "p3", "lat": 21.00, "lng": 105.83},
    {"id": "p4", "lat": 21.04, "lng": 105.82}
]

env = RouteEnv(places)

policy_net = DQN(env.observation_space.shape[0], env.action_space.n)
target_net = DQN(env.observation_space.shape[0], env.action_space.n)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
memory = deque(maxlen=10000)

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

# Test 
state = env.reset()
done = False

while not done:
    action = select_action(state, env.visited, epsilon=0.0)
    state, _, done, _ = env.step(action)

route = [places[i]["id"] for i in env.route]
print("Optimal route:", route)

