import math
import random
from collections import defaultdict

places = [
    {"id": "p0", "lat": 21.03, "lng": 105.85},
    {"id": "p1", "lat": 21.01, "lng": 105.84},
    {"id": "p2", "lat": 21.05, "lng": 105.86},
    {"id": "p3", "lat": 21.00, "lng": 105.83},
    {"id": "p4", "lat": 21.04, "lng": 105.82}
]


def distance(i, j):
    p1, p2 = places[i], places[j]
    return math.sqrt((p1["lat"] - p2["lat"])**2 + (p1["lng"] - p2["lng"])**2)

def get_valid_actions(visited_mask):
    return [i for i in range(5) if not (visited_mask & (1 << i))]

def is_done(visited_mask):
    return visited_mask == (1 << 5) - 1

Q = defaultdict(lambda: [0.0] * 5)

alpha = 0.1      # learning rate
gamma = 0.95     # discount factor
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
episodes = 5000

for ep in range(episodes):
    current = 0
    visited_mask = 1 << current

    while not is_done(visited_mask):
        state = (current, visited_mask)
        valid_actions = get_valid_actions(visited_mask)

        # epsilon-greedy
        if random.random() < epsilon:
            action = random.choice(valid_actions)
        else:
            q_vals = Q[state]
            action = max(valid_actions, key=lambda a: q_vals[a])

        next_mask = visited_mask | (1 << action)
        reward = -distance(current, action)

        next_state = (action, next_mask)

        # Q-learning update
        best_next = max(Q[next_state][a] for a in get_valid_actions(next_mask)) \
                    if not is_done(next_mask) else 0

        Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])

        current = action
        visited_mask = next_mask

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

route = [0]
current = 0
visited_mask = 1 << current

while not is_done(visited_mask):
    state = (current, visited_mask)
    valid_actions = get_valid_actions(visited_mask)
    action = max(valid_actions, key=lambda a: Q[state][a])

    route.append(action)
    visited_mask |= (1 << action)
    current = action

route_ids = [places[i]["id"] for i in route]
print("Optimal route:", route_ids)
