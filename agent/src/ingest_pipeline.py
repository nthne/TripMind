import json
import chromadb
import torch
import pickle
import warnings
import os
from sklearn.exceptions import InconsistentVersionWarning
from model import TripMindEncoder
from utils import get_semantic_vector

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"--- Đang chạy trên thiết bị: {device} ---")

# 1. Load assets & model
ASSETS_PATH = "/Users/trannguyenmyanh/Documents/TripMind/agent/weights/assets.pkl"
WEIGHTS_PATH = "/Users/trannguyenmyanh/Documents/TripMind/agent/weights/encoder_weights.pth"

with open(ASSETS_PATH, "rb") as f:
    assets = pickle.load(f)
WORD2IDX = assets['word2idx']
vocab_size = assets['vocab_size']

encoder = TripMindEncoder(vocab_size, d_model=128, nhead=8, num_layers=4)
encoder.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
encoder.to(device).eval()

# 2. Khởi tạo ChromaDB
DB_PATH = "/Users/trannguyenmyanh/Documents/TripMind/agent/tripmind_vector_db"
client = chromadb.PersistentClient(path=DB_PATH)

# Xóa collection cũ để tránh rác dữ liệu nếu cần làm lại từ đầu
try:
    client.delete_collection("tripmind_reviews")
    print("--- Đã xóa collection cũ để làm mới ---")
except:
    pass

collection = client.create_collection(
    name="tripmind_reviews", 
    metadata={"hnsw:space": "cosine"}
)

def ingest_data(file_path: str):
    print(f"Đang đọc dữ liệu từ: {file_path}...")
    
    batch_size = 100
    ids, docs, metas, embs = [], [], [], []

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            
            review_text = data.get('text', '')
            if not review_text: continue

            # Xử lý ID tỉnh (đảm bảo format "00", "01")
            p_id = str(data.get("province_id", "")).zfill(2)
            
            # Xử lý Trip Type để Agent sau này có thể lọc ($where)
            # 1. Lấy dữ liệu trip thô
            trip_raw = data.get("trip")
            
            # 2. Xử lý an toàn: Chuyển từ String/None sang Dictionary
            trip_data = {}
            if trip_raw: # Nếu không phải None hoặc chuỗi rỗng
                if isinstance(trip_raw, str):
                    try:
                        trip_data = json.loads(trip_raw)
                    except json.JSONDecodeError:
                        trip_data = {}
                elif isinstance(trip_raw, dict):
                    trip_data = trip_raw

            # 3. Bây giờ gọi .get() sẽ không bao giờ lỗi nữa
            # Đảm bảo trip_data luôn là dict, nếu không thì dùng dict trống
            if not isinstance(trip_data, dict): 
                trip_data = {}
                
            trip_type = trip_data.get("trip_type", "any")
            if not trip_type: trip_type = "any"

            # Tạo metadata CHUẨN để phân biệt địa danh
            metadata = {
                "province_id": p_id,
                "destination_id": str(data.get("destination_id")), # CỰC KỲ QUAN TRỌNG
                "name": data.get("name", "Unknown"),
                "province_name": data.get("new_province_x", "Unknown"),
                "rating": float(data.get("rating_x", 0)),
                "trip_type": trip_type.lower() # Dùng để lọc trong Agent
            }

            # Vector hóa
            vector = get_semantic_vector(review_text, encoder, assets, device)

            ids.append(str(data.get('id_review', i)))
            docs.append(review_text)
            metas.append(metadata)
            embs.append(vector)

            # Thêm theo batch để tối ưu memory
            if len(ids) >= batch_size:
                collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
                print(f"Đã nạp {i+1} dòng...")
                ids, docs, metas, embs = [], [], [], []

    # Nạp nốt phần còn lại
    if ids:
        collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    
    print(f"✅ Hoàn thành! Tổng cộng: {collection.count()} bản ghi.")

if __name__ == "__main__":
    DATA_PATH = "/Users/trannguyenmyanh/Documents/TripMind/data/cleaned_data.jsonl"
    ingest_data(DATA_PATH)