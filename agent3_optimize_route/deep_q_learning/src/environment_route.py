import gymnasium as gym
import numpy as np
from gymnasium import spaces
import math

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

class RouteEnv(gym.Env):
    def __init__(self, places):
        super().__init__()
        self.places = places
        self.n = len(places)

        self.action_space = spaces.Discrete(self.n)

        # state = [current_lat, current_lng, visited_mask(5)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 + self.n,)
        )

        self.reset()

    def reset(self):
        self.current = 0
        self.visited = np.zeros(self.n)
        self.visited[0] = 1
        self.route = [0]
        return self._get_state()

    def _get_state(self):
        coords = []
        for p in self.places:
            coords.extend([p["lat"], p["lng"]])

        return np.array(coords + self.visited.tolist(), dtype=np.float32)


    def step(self, action):
        if self.visited[action] == 1:
            return self._get_state(), -10, False, {}

        prev = self.places[self.current]
        nxt = self.places[action]

        dist = distance(
            (prev["lat"], prev["lng"]),
            (nxt["lat"], nxt["lng"])
        )

        reward = -dist

        self.current = action
        self.visited[action] = 1
        self.route.append(action)

        done = self.visited.sum() == self.n
        return self._get_state(), reward, done, {}
