import json

def load_raw_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

from collections import defaultdict

def group_by_provinces(path):
    all_places = load_raw_data(path)

    places_by_province = defaultdict(list)

    for p in all_places:
        temp = {"destination_id": p["destination_id"], "province_id": p["province_id"], "lat": float(p["tọa độ"].split(",")[0].strip()), "lng": float(p["tọa độ"].split(",")[1].strip())}
        places_by_province[p["province_id"]].append(temp)

    return places_by_province

import random

def sample_episode_places(path, k=5):
    # chỉ lấy province có >= k places
    places_by_province = group_by_provinces(path)
    
    valid_provinces = [
        pid for pid, ps in places_by_province.items()
        if len(ps) >= k
    ]

    province_id = random.choice(valid_provinces)
    places = random.sample(places_by_province[province_id], k)

    return province_id, places



