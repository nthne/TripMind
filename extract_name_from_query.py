import json

data_path = "TripMind/data/cleaned_data.jsonl"
data_list_provinces_path = "TripMind/data/list_provinces.json"

provinces = []
with open(data_list_provinces_path, "r", encoding='utf-8') as f:
    list_provinces = json.load(f)

def extract_city_from_query(query):
    for name, id in list_provinces:
        if name in query:
            return id
        