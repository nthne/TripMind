from flask import Flask, request, jsonify
import requests
import torch
import pickle
import os
import logging
from model import TripMindEncoder
from database import get_provinces_stats, agent_1_output

# Cấu hình Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- GLOBAL VARIABLES ---
MODEL = None
WORD2IDX = None
ASSETS = None
# Tự động xác định thiết bị (MPS cho Mac, CUDA cho GPU, hoặc CPU)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
PROVINCE_STATS = None
AGENT_2_URL = "http://localhost:8000/ranking"

def load_system():
    global MODEL, ASSETS, WORD2IDX, PROVINCE_STATS
    try:
        logger.info(f"🚀 Khởi tạo hệ thống trên thiết bị: {DEVICE}...")
        weights_path = "/Users/trannguyenmyanh/Documents/TripMind/agent/weights"
        
        # 1. Load Assets
        with open(os.path.join(weights_path, "assets.pkl"), "rb") as f:
            ASSETS = pickle.load(f)
        
        WORD2IDX = ASSETS['word2idx']
        vocab_size = ASSETS['vocab_size']
        
        # Lấy số lượng category từ cat_encoder
        num_categories = len(ASSETS['cat_encoder'].classes_)
        
        # 2. Khởi tạo Model Multi-task
        # Đảm bảo d_model=256 và num_layers khớp với lúc train trên Kaggle
        MODEL = TripMindEncoder(
            vocab_size=vocab_size,
            num_categories=num_categories,
            d_model=256,   
            nhead=8,
            num_layers=4   
        ).to(DEVICE)
        
        # 3. Load trọng số
        weights_file = os.path.join(weights_path, "encoder_weights.pth")
        state_dict = torch.load(weights_file, map_location=DEVICE)
        MODEL.load_state_dict(state_dict)
        MODEL.eval()
        
        # 4. Load thống kê tỉnh thành
        PROVINCE_STATS = get_provinces_stats()
        
        logger.info("✅ Hệ thống Transformer Multi-task và Database đã sẵn sàng!")
        
    except Exception as e:
        logger.error(f"❌ Lỗi khởi động hệ thống: {str(e)}")
        raise e

@app.route('/api/v1/recommend', methods=['POST'])
def recommend_places():
    try:
        data = request.get_json()
        query = data.get('query')
        province_id = data.get('province_id')
        
        if not query or province_id is None:
            return jsonify({"success": False, "error": "Missing query or province_id"}), 400
        
        # Chuẩn hóa province_id thành dạng chuỗi "00", "01"...
        p_id_str = str(province_id).zfill(2) 
        trip_type = data.get('trip_type', 'any')
        n_places = min(int(data.get('n_places', 10)), 50)
        max_reviews = int(data.get('max_reviews_per_place', 5))
        
        # Ghi log debug query
        logger.info(f"🔍 Query: '{query}' | Tỉnh: {p_id_str}")

        # GỌI AGENT 1 (Recall)
        # Agent 1 sẽ sử dụng model Transformer để tạo vector và tìm kiếm trong ChromaDB
        candidates = agent_1_output(
            user_query=query,
            model=MODEL,
            word2idx=WORD2IDX,
            assets=ASSETS,
            device=DEVICE,
            province_id=p_id_str,
            trip_type=trip_type,
            n_places=10, 
            max_reviews_per_place=max_reviews
        )
        print(f"Agent 1 tìm thấy: {len(candidates) if candidates else 'None'} ứng viên")
        
        if not candidates:
            return jsonify({
                "success": True, 
                "data": [], 
                "message": f"Không tìm thấy địa điểm nào tại tỉnh {p_id_str}"
            }), 200

        # GỌI AGENT 2 (Ranking)
        try:
            response = requests.post(AGENT_2_URL, json=candidates, timeout=10)
            if response.status_code == 200:
                final_results = response.json()[:n_places]
            else:
                final_results = candidates[:n_places]
        except Exception as e:
            logger.error(f"Agent 2 Error: {e}")
            final_results = candidates[:n_places]

        return jsonify({
            "success": True,
            "data": final_results,
            "metadata": {
                "province_id": p_id_str,
                "candidates_found": len(candidates),
                "device_used": str(DEVICE)
            }
        }), 200
    
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/v1/provinces', methods=['GET'])
def get_provinces():
    if PROVINCE_STATS:
        provinces = [{"province_id": pid, **stats} for pid, stats in sorted(PROVINCE_STATS.items(), key=lambda x: x[1]['total_reviews'], reverse=True)]
        return jsonify({"success": True, "total_provinces": len(provinces), "provinces": provinces}), 200
    return jsonify({"success": False, "error": "Stats not available"}), 500

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "device": str(DEVICE), "agent2_link": AGENT_2_URL}), 200

if __name__ == "__main__":
    load_system()
    app.run(host='0.0.0.0', port=5000, debug=False)