import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000/api/v1"

def print_header(title):
    print("\n" + "="*80)
    print(f" {title.upper()} ".center(80, "="))
    print("="*80)

def test_health():
    print_header("🧪 Test: Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        return response.json()
    except Exception as e:
        print(f"❌ Lỗi kết nối API: {e}")
        return None

def test_get_provinces():
    print_header("🧪 Test: Get Provinces List")
    response = requests.get(f"{BASE_URL}/provinces")
    data = response.json()
    if data['success']:
        print(f"✅ Tìm thấy {data['total_provinces']} tỉnh thành.")
        # In ra 5 tỉnh đầu tiên có nhiều review nhất
        for p in data['provinces'][:5]:
            print(f" - ID {p['province_id']}: {p['total_reviews']} reviews, {p['unique_places']} địa điểm")
    return data

def test_recommend(query, province_id, trip_type="any", n_places=10):
    print_header(f"🧪 Test Recommend: {query} (ID: {province_id})")
    
    payload = {
        "query": query,
        "province_id": province_id,
        "trip_type": trip_type,
        "n_places": n_places,
        "max_reviews_per_place": 3
    }
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/recommend", json=payload)
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        results = data.get('data', [])
        print(f"✅ Thành công ({elapsed:.2f}s) - Tìm thấy {len(results)} địa điểm")
        print(f"📍 Province: {data['metadata']['province_id']}")
        print(f"👥 Trip Type: {trip_type}")
        
        for i, place in enumerate(results, 1):
            print(f"\n{i}. 🏛️ {place['name']} (ID: {place['place_id']})")
            print(f"   💬 Review: \"{place['reviews'][0][:100]}...\"")
    else:
        print(f"❌ Lỗi {response.status_code}: {response.text}")

if __name__ == "__main__":
    # 1. Kiểm tra trạng thái hệ thống
    health = test_health()
    if not health:
        exit()

    # 2. Lấy danh sách tỉnh để biết ID nào đang có dữ liệu
    test_get_provinces()

    # 3. Chạy các kịch bản test thực tế
    # Test 1: Truy vấn cơ bản về An Giang (ID 00)
    test_recommend(
        query="Tôi thích tham quan các địa điểm có cảnh đẹp thiên nhiên", 
        province_id="15", 
        trip_type="family"
    )

    # # Test 2: Truy vấn về du lịch tâm linh tại An Giang
    # test_recommend(
    #     query="đi chùa bà chúa xứ cầu may", 
    #     province_id="00", 
    #     trip_type="family"
    # )

    # # Test 3: Test một tỉnh khác (ví dụ Đắk Lắk - ID 31 dựa trên snippet của bạn)
    # test_recommend(
    #     query="bảo tàng dân tộc học", 
    #     province_id="31", 
    #     trip_type="friends"
    # )

    # # Test 4: Trường hợp lỗi - Thiếu tham số
    # print_header("🧪 Test: Edge Case - Missing Query")
    # bad_response = requests.post(f"{BASE_URL}/recommend", json={"province_id": "00"})
    # print(f"Kết quả mong đợi (Lỗi 400): {bad_response.status_code} - {bad_response.json().get('error')}")

    print("\n" + "="*80)
    print(" CÁC BÀI KIỂM TRA HOÀN TẤT ".center(80, "="))
    print("="*80)