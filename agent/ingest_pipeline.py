import chromadb
import json
from src.model import TripMindEncoder
from src.utils import get_semantic_vector
import torch
import pickle

# Load assets
with open("weights/assets.pkl", "rb") as f:
    assets = pickle.load(f)

encoder = TripMindEncoder(len(assets['word2idx']), 128, 128, 128)
encoder.load_state_dict(torch.load("weights/encoder_weights.pth"))
encoder.to(DEVICE).eval()

client = chromadb.PersistentClient(path="./tripmind_vector_db")
collection = client.get_or_create_collection("tripmind_reviews")

# Ingest (Sử dụng logic ingest_to_chromadb bạn đã có)
# Nhớ ép kiểu metadata về string/float/int