from flask import Flask, request, jsonify
import torch
import pickle
import os
import logging
from model import TripMindEncoder
# Chuyển import lên đây để tránh overhead khi gọi API
from database import get_provinces_stats, agent_1_output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
MODEL = None
WORD2IDX = None
DEVICE = None
PROVINCE_STATS = None

def load_system(weights_path="/Users/trannguyenmyanh/Documents/TripMind/agent/weights"):
    global MODEL, WORD2IDX, DEVICE, PROVINCE_STATS
    
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"🚀 Khởi tạo trên {DEVICE}...")

    print("--- DEBUG: Đang gọi get_provinces_stats() ---")
    
    try:
        # 1. Load Assets
        with open(os.path.join(weights_path, "assets.pkl"), "rb") as f:
            assets = pickle.load(f)
        WORD2IDX = assets['word2idx']
        vocab_size = assets['vocab_size']
        
        # 2. Load Model
        MODEL = TripMindEncoder(vocab_size, 128, 128, 128)
        MODEL.load_state_dict(
            torch.load(os.path.join(weights_path, "encoder_weights.pth"), map_location=DEVICE)
        )
        MODEL.to(DEVICE)
        MODEL.eval()
        
        # 3. Load Stats từ DB
        PROVINCE_STATS = get_provinces_stats()
        logger.info(f"✓ Loaded {len(PROVINCE_STATS)} provinces (New ID format: 00-33)")
        print("--- DEBUG: Đã load PROVINCE_STATS ---", PROVINCE_STATS)
        
    except Exception as e:
        logger.error(f"❌ Lỗi khởi động hệ thống: {e}")
        raise e

@app.route('/api/v1/recommend', methods=['POST'])
def recommend_places():
    try:
        data = request.get_json()
        
        # Validate input
        query = data.get('query')
        province_id = data.get('province_id')
        
        if not query or province_id is None:
            return jsonify({"success": False, "error": "Missing query or province_id"}), 400
        
        # Chuẩn hóa province_id sang string (để khớp với format "00", "01")
        # Nếu client gửi 0, nó sẽ thành "00"
        p_id_str = str(province_id).zfill(2) 
        
        trip_type = data.get('trip_type', 'any')
        n_places = min(int(data.get('n_places', 10)), 50)
        max_reviews = int(data.get('max_reviews_per_place', 5))
        
        # Gọi xử lý
        results = agent_1_output(
            user_query=query,
            model=MODEL,
            word2idx=WORD2IDX,
            device=DEVICE,
            province_id=p_id_str, # Dùng ID đã chuẩn hóa
            trip_type=trip_type,
            n_places=n_places,
            max_reviews_per_place=max_reviews
        )
        
        province_info = PROVINCE_STATS.get(p_id_str, {})
        
        return jsonify({
            "success": True,
            "data": results,
            "metadata": {
                "province_id": p_id_str,
                "province_stats": province_info
            }
        }), 200
        
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/v1/provinces', methods=['GET'])
def get_provinces():
    """Lấy danh sách tỉnh thành có trong database"""
    try:
        if PROVINCE_STATS:
            provinces = [
                {
                    "province_id": pid,
                    "total_reviews": stats['total_reviews'],
                    "unique_places": stats['unique_places']
                }
                for pid, stats in sorted(PROVINCE_STATS.items(), 
                                        key=lambda x: x[1]['total_reviews'], 
                                        reverse=True)
            ]
            
            return jsonify({
                "success": True,
                "total_provinces": len(provinces),
                "provinces": provinces
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": "Province stats not available"
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "agent": "Agent 1 - Place Recommendation",
        "device": str(DEVICE),
        "provinces_loaded": len(PROVINCE_STATS) if PROVINCE_STATS else 0,
        "model_loaded": MODEL is not None
    }), 200


if __name__ == "__main__":
    load_system()
    app.run(host='0.0.0.0', port=5000, debug=False)