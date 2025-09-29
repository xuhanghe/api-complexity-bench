import json

with open("server_scores.json", "r") as f:
    data = json.load(f)

data = data["server_scores"]
hard_count = 0
for entry in data:
    if entry["level"] == "Hard":
        hard_count += 1
    if entry["factor"] == 50:
        print(entry)
        break
    
# print(f"Number of hard entries: {hard_count}")
# print(f"Total entries: {len(data)}")