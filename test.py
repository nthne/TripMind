import pickle
with open("/Users/trannguyenmyanh/Documents/TripMind/agent/weights/assets.pkl", "rb") as f:
    assets = pickle.load(f)
    w2i = assets['word2idx']
    # print(f"Kiểm tra 'chùa': {'chùa' in w2i}")
    # print(f"Kiểm tra 'đi_chùa': {'đi_chùa' in w2i}")
    print(f"Tổng số từ trong từ điển: {len(w2i)}")
    print(f"Ví dụ từ điển: {list(w2i.items())[:100]}")