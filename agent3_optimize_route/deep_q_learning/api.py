# agent3/api.py
from fastapi import FastAPI, Body
import uvicorn
from optimize_route import optimize_route

app = FastAPI()

@app.post("/optimize")
def optimize(list_places: list = Body(...)):
    optimized_data = optimize_route(list_places)
    return optimized_data

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)

