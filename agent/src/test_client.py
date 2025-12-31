import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000/api/v1"

def print_header(title):
    print("\n" + "="*80)
    print(f" {title.upper()} ".center(80, "="))
    print("="*80)

def test_health():
    try:
        response = requests.get(f"{BASE_URL}/health")
        return response.json()
    except:
        return None

def test_recommend(query, province_id, trip_type="any", n_places=10):
    print_header(f"🧪 Test Recommend: {query}")
    
    payload = {
        "query": query,
        "province_id": province_id,
        "trip_type": trip_type,
        "n_places": n_places
    }
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/recommend", json=payload)
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        results = data.get('data', [])
        print(f"✅ Thành công ({elapsed:.2f}s) - Tìm thấy {len(results)} địa điểm")
        print(f"📍 Province: {data['metadata']['province_id']} | 👥 Loại hình: {trip_type}")
        
        for i, place in enumerate(results, 1):
            # SỬA LỖI TẠI ĐÂY: Dùng 'destination_id' thay vì 'place_id'
            d_id = place.get('destination_id', 'N/A')
            name = place.get('name', 'Unknown')
            reviews = place.get('reviews', [])
            
            print(f"\n{i}. 🏛️ {name} (ID: {d_id})")
            print(f"   📊 Số lượng review tìm được: {len(reviews)}")
            
            # In ra tối đa 2 review tiêu biểu
            for j, rev_text in enumerate(reviews[:2], 1):
                # Cắt bớt nếu review quá dài để dễ nhìn log
                short_rev = (rev_text[:120] + '...') if len(rev_text) > 120 else rev_text
                print(f"   💬 Review {j}: \"{short_rev}\"")
    else:
        print(f"❌ Lỗi {response.status_code}: {response.text}")

if __name__ == "__main__":
    # Khởi động test
    health = test_health()
    if not health:
        print("❌ API chưa chạy. Hãy chạy 'python api.py' trước.")
        exit()

    # Test với ID 15 (Hải Phòng/Quảng Ninh tùy data của bạn)
    test_recommend(
        query="Tôi thích tham quan các địa điểm có cảnh đẹp thiên nhiên", 
        province_id="15", 
        trip_type="family",
        n_places=10 # Đã tăng lên 10 theo yêu cầu của bạn
    )