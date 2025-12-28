import json
import torch
import chromadb
from tqdm import tqdm
from src.utils import get_semantic_vector  # Import từ utils.py

# 1. Khởi tạo ChromaDB Persistent Client
# Dữ liệu sẽ được lưu vào thư mục này để không bị mất khi tắt chương trình
client = chromadb.PersistentClient(path="./tripmind_vector_db")
collection = client.get_or_create_collection(
    name="tripmind_reviews",
    metadata={"hnsw:space": "cosine"} # Đảm bảo sử dụng độ tương đồng Cosine
)

def ingest_to_chromadb(file_path, model, word2idx, device):
    """
    Nạp dữ liệu từ JSONL vào ChromaDB.
    Mỗi review trở thành một vector độc lập.
    """
    model.eval()
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Ingesting to ChromaDB"):
        record = json.loads(line)
        text = record.get("text", "")
        if not text: continue

        # --- Xử lý Trip Type từ chuỗi JSON ---
        # Dữ liệu thực tế: "{\"stay_date\": \"...\", \"trip_type\": \"family\"}"
        trip_raw = record.get("trip", "{}")
        try:
            trip_data = json.loads(trip_raw) if isinstance(trip_raw, str) else trip_raw
            trip_type = trip_data.get("trip_type", "unknown")
        except:
            trip_type = "unknown"

        # --- Xử lý Categories ---
        # Dữ liệu thực tế: "[{\"id\":11091,\"name\":\"...\"}]"
        cat_raw = record.get("categories", "[]")
        try:
            cat_list = json.loads(cat_raw) if isinstance(cat_raw, str) else cat_raw
            cat_id = str(cat_list[0]['id']) if cat_list and len(cat_list) > 0 else "unknown"
        except:
            cat_id = "unknown"

        # --- Tạo Semantic Vector bằng Bi-LSTM ---
        vector = get_semantic_vector(text, model, word2idx, device)

        # --- Nạp vào ChromaDB với Metadata đầy đủ ---
        collection.add(
            embeddings=[vector],
            documents=[text],
            metadatas=[{
                "province_id": str(record.get("province_id")),
                "category_id": cat_id,
                "name": record.get("name"),
                "trip_type": trip_type, 
                "rating": float(record.get("rating_x", 0))
            }],
            ids=[str(record.get("id_review"))]
        )

def agent_1_query(user_query, model, word2idx, device, province_id, trip_type):
    """
    Thực hiện truy vấn ngữ nghĩa kết hợp lọc Metadata.
    """
    # 1. Chuyển query thành vector bằng chính model đã train
    query_vector = get_semantic_vector(user_query, model, word2idx, device)
    
    # 2. Truy vấn kết hợp lọc Metadata
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=10,
        where={
            "$and": [
                {"province_id": {"$eq": str(province_id)}},
                {"trip_type": {"$eq": trip_type}}
            ]
        }
    )
    return results