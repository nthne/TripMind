from fastapi import FastAPI
import torch
from optimize_route import optimize_route
app = FastAPI()

checkpoint_path = "agent3_optimize_route/deep_q_learning/dqn_route_checkpoints.pt"

@app.post("/optimize_route")
def optimize_route(list_places):
    return optimize_route(list_places, checkpoint_path)

