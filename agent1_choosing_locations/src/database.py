import json
import chromadb
from typing import List, Dict, Optional
import logging
from utils import get_semantic_vector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "/Users/trannguyenmyanh/Documents/TripMind/agent/tripmind_vector_db"

def get_provinces_stats():
    """
    Thống kê số lượng review và số địa điểm duy nhất cho mỗi tỉnh.
    Sử dụng destination_id để phân biệt các địa danh.
    """
    try:
        import os
        # 1. Đảm bảo dùng đường dẫn tuyệt đối
        DB_PATH = "/Users/trannguyenmyanh/Documents/TripMind/agent/tripmind_vector_db"
        
        if not os.path.exists(DB_PATH):
            logger.error(f"❌ Không tìm thấy thư mục DB tại: {DB_PATH}")
            return {}

        client = chromadb.PersistentClient(path=DB_PATH)
        target = "tripmind_reviews"
        
        # 2. Kiểm tra collection
        col_names = [c.name for c in client.list_collections()]
        if target not in col_names:
            logger.warning(f"⚠️ Collection '{target}' chưa tồn tại.")
            return {}

        collection = client.get_collection(name=target)
        
        # 3. Lấy toàn bộ metadata (Chỉ lấy metadata để tiết kiệm RAM)
        results = collection.get(include=['metadatas'])
        metadatas = results.get('metadatas', [])
        
        if not metadatas:
            return {}

        stats = {}
        for meta in metadatas:
            # Lấy province_id và ép kiểu string chuẩn ("00", "01"...)
            p_id = str(meta.get('province_id', '')).zfill(2)
            # Lấy destination_id để đếm địa điểm duy nhất
            d_id = meta.get('destination_id')
            
            if p_id and p_id != "34": # Hoặc xử lý mã "34" tùy logic của bạn
                if p_id not in stats:
                    stats[p_id] = {
                        "total_reviews": 0, 
                        "unique_places": set() # Dùng set để tự động loại bỏ trùng lặp
                    }
                
                stats[p_id]["total_reviews"] += 1
                if d_id:
                    stats[p_id]["unique_places"].add(str(d_id))

        # 4. Chuyển đổi set() thành con số cụ thể để trả về JSON
        final_result = {}
        for pid, data in stats.items():
            final_result[pid] = {
                "total_reviews": data["total_reviews"],
                "unique_places": len(data["unique_places"]) # Số lượng địa danh thực tế
            }
        
        logger.info(f"✅ Thống kê xong: {len(final_result)} tỉnh thành.")
        return final_result

    except Exception as e:
        logger.error(f"❌ Lỗi khi tính stats: {str(e)}")
        return {}
    
def ingest_to_chromadb(file_path, model, assets, device, batch_size=100):
    from utils import get_semantic_vector
    import os
    
    # 1. Dùng đường dẫn tuyệt đối để tránh đọc nhầm file rác ở thư mục khác
    client = chromadb.PersistentClient(path="/Users/trannguyenmyanh/Documents/TripMind/agent/tripmind_vector_db")
    
    try:
        client.delete_collection("tripmind_reviews")
        logger.info("♻️ Đã xóa collection cũ để làm mới hoàn toàn")
    except:
        pass
    
    collection = client.create_collection(
        name="tripmind_reviews",
        metadata={"hnsw:space": "cosine"}
    )
    
    model.eval()
    
    # 2. Đọc file theo dạng generator để tiết kiệm RAM và tránh lỗi đọc file
    valid_count = 0
    skipped_count = 0
    embeddings_batch, documents_batch, metadatas_batch, ids_batch = [], [], [], []
    
    # Danh sách kiểm tra tỉnh (Set này sẽ giúp ta debug xem có bao nhiêu tỉnh thực tế)
    detected_provinces = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                record = json.loads(line.strip())
                text = record.get("text", "").strip()
                
                if not text or len(text) < 10:
                    skipped_count += 1
                    continue

                # ÉP KIỂU VÀ KIỂM TRA ID
                p_id = str(record.get("province_id", ""))
                d_id = str(record.get("destination_id", ""))
                
                # CƠ CHẾ CHỐNG SAI LỆCH: 
                # ID tỉnh thường ngắn (6 số), ID địa danh thường dài hơn.
                # Nếu p_id > 7 chữ số, khả năng cao là dữ liệu bị lệch cột.
                if not p_id or not d_id or len(p_id) > 7:
                    skipped_count += 1
                    continue

                detected_provinces.add(p_id)

                # 2. XỬ LÝ TRIP TYPE (Giữ logic cũ nhưng thêm clean data)
                trip_raw = record.get("trip", "{}")
                try:
                    trip_data = json.loads(trip_raw) if isinstance(trip_raw, str) else trip_raw
                    trip_type = trip_data.get("trip_type") if trip_data else "any"
                    trip_type = trip_type.lower().strip() if trip_type else "any"
                    if trip_type in ["", "none", "null"]: trip_type = "any"
                except:
                    trip_type = "any"

                # 3. XỬ LÝ CATEGORIES
                cat_raw = record.get("categories", "[]")
                try:
                    cat_list = json.loads(cat_raw) if isinstance(cat_raw, str) else cat_raw
                    if isinstance(cat_list, list) and len(cat_list) > 0:
                        cat_id = str(cat_list[0].get('id', '0'))
                        cat_name = cat_list[0].get('name', 'Chưa phân loại')
                    else:
                        cat_id = "0"
                        cat_name = "Chưa phân loại"
                except:
                    cat_id = "0"
                    cat_name = "Chưa phân loại"

                # 4. TẠO SEMANTIC VECTOR (Sử dụng assets và device đã sửa cho M1)
                vector = get_semantic_vector(text, model, assets, device)

                metadatas_batch.append({
                    "province_id": p_id,
                    "destination_id": d_id,
                    "name": record.get("name", "").strip(),
                    "rating": float(record.get("rating_x", 0)),
                    "trip_type": trip_type, # Từ logic xử lý của bạn
                    "place_key": f"{p_id}_{d_id}"
                })
                
                embeddings_batch.append(vector)
                documents_batch.append(text)
                ids_batch.append(str(record.get("id_review", f"rev_{idx}")))
                
                valid_count += 1

                if len(embeddings_batch) >= batch_size:
                    collection.add(embeddings=embeddings_batch, documents=documents_batch, 
                                   metadatas=metadatas_batch, ids=ids_batch)
                    embeddings_batch, documents_batch, metadatas_batch, ids_batch = [], [], [], []
                
            except Exception as e:
                skipped_count += 1
                continue
    
    # Insert batch cuối
    if embeddings_batch:
        collection.add(embeddings=embeddings_batch, documents=documents_batch, 
                       metadatas=metadatas_batch, ids=ids_batch)

    logger.info(f"✅ Hoàn tất! Phát hiện thực tế: {len(detected_provinces)} tỉnh.")
    logger.info(f"✅ Danh sách ID tỉnh: {sorted(list(detected_provinces))}")
    return collection


def agent_1_output(user_query, model, word2idx, device, province_id, trip_type='any', n_places=10, max_reviews_per_place=5):
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection("tripmind_reviews")
        
        # 1. Tạo Query Vector
        query_vector = get_semantic_vector(user_query, model, {"word2idx": word2idx}, device)
        
        # 2. Xây dựng Filter "lỏng" hơn để tránh ra 0 kết quả
        p_id_str = str(province_id).zfill(2)
        
        # Luôn lọc theo tỉnh
        where_filter = {"province_id": p_id_str}
        
        # Chỉ lọc trip_type nếu khách yêu cầu cụ thể và không phải 'any'
        if trip_type and trip_type.lower() not in ["any", "all", "none"]:
            where_filter = {
                "$and": [
                    {"province_id": p_id_str},
                    {"trip_type": trip_type.lower()}
                ]
            }

        # 3. Query
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=50, # Lấy nhiều một chút để lọc trùng
            where=where_filter
        )

        if not results['documents'] or not results['documents'][0]:
            return []

        places_dict = {}
        for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
            # DEBUG: Bạn có thể bỏ comment dòng dưới để xem metadata thật sự có gì
            # print(f"DEBUG META: {meta}") 

            p_id = meta.get("destination_id") or meta.get("place_key") or "unknown_id"
            p_name = meta.get("name") or meta.get("place_name") or "Địa danh chưa xác định"
            
            if p_id not in places_dict:
                places_dict[p_id] = {
                    "place_id": p_id,
                    "name": p_name,
                    "reviews": [doc],
                    "score": dist
                }
            elif len(places_dict[p_id]["reviews"]) < max_reviews_per_place:
                places_dict[p_id]["reviews"].append(doc)

        # Sắp xếp theo khoảng cách (càng nhỏ càng giống)
        sorted_output = sorted(places_dict.values(), key=lambda x: x['score'])
        return sorted_output[:n_places]

    except Exception as e:
        logger.error(f"Lỗi Agent: {e}")
        return {"message": "No province data available."}