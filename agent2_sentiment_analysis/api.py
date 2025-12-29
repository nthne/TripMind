from fastapi import FastAPI
import torch
from src.model import LabelReviewModel
from label_review import ranking_place
app = FastAPI()

ckpt = torch.load("agent2_sentiment_analysis/sentiment_checkpoint.pt", map_location="cpu")
vocab = ckpt["vocab"]

model = LabelReviewModel(len(vocab))
model.load_state_dict(ckpt["model_state"])
model.eval()

@app.post("/ranking")
def ranking(list_places):
    return ranking_place(list_places)

