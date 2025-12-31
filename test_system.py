import requests
import json
import time

# URL của Agent 1 (Cửa ngõ chính)
API_URL = "http://127.0.0.1:5000/api/v1/recommend"

def test_full_flow(query, province_id, trip_type="any"):
    print("\n" + "="*80)
    print(f"🔍 TRUY VẤN: {query}")
    print(f"📍 TỈNH: {province_id} | 👥 LOẠI HÌNH: {trip_type}")
    print("="*80)

    payload = {
        "query": query,
        "province_id": province_id,
        "trip_type": trip_type,
        "n_places": 5  # Yêu cầu Top 5 cuối cùng
    }

    try:
        start_time = time.time()
        response = requests.post(API_URL, json=payload, timeout=15)
        elapsed = time.time() - start_time

        if response.status_code == 200:
            res_data = response.json()
            results = res_data.get('data', [])
            meta = res_data.get('metadata', {})

            print(f"✅ Thành công! Thời gian xử lý tổng cộng: {elapsed:.2f}s")
            print(f"📊 Agent 1 tìm thấy: {meta.get('candidates_retrieved')} ứng viên")
            print(f"🏆 Agent 2 đã lọc và xếp hạng xong.")
            print("-" * 40)

            for i, place in enumerate(results, 1):
                # Kiểm tra xem có final_score từ Agent 2 không
                score = place.get('final_score', 'N/A')
                print(f"{i}. 🏛️ {place['name']} (ID: {place['destination_id']})")
                print(f"   🌟 ĐIỂM AGENT 2: {score}")
                if place.get('reviews'):
                    print(f"   💬 Review tiêu biểu: \"{place['reviews'][0][:100]}...\"")
                print("-" * 40)
        else:
            print(f"❌ Lỗi hệ thống: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"❌ Không thể kết nối tới Agent 1: {e}")

if __name__ == "__main__":
    # Test 1: An Giang - Tâm linh
    test_full_flow("Tôi muốn đi chùa cầu bình an", "00", "family")
    
    # Test 2: Đắk Lắk - Cà phê/Văn hóa
    test_full_flow("Thưởng thức cà phê đặc sản và bảo tàng", "31", "friends")