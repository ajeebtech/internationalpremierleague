import json
import torch
import os
import pandas as pd
from tqdm import tqdm

# Load player names from CSV
players_df = pd.read_csv("players.csv")
player_names = players_df["player"].dropna().unique()

# Directory where all match JSONs are stored
root_dir = "/Users/jatin/Documents/python/multipremierleague"

for bowler_name in tqdm(player_names, desc="Processing bowlers"):
    overs_data = []

    # Walk inside multipremierleague looking for folders ending with _json
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        if os.path.isdir(subfolder_path) and subfolder.endswith("_json"):
            # Now go through all JSON files in that subfolder (and its subfolders)
            for root, dirs, files in os.walk(subfolder_path):
                for file in files:
                    if file.endswith(".json"):
                        json_path = os.path.join(root, file)
                        try:
                            with open(json_path, "r") as f:
                                match = json.load(f)
                        except Exception:
                            continue  # Skip malformed files

                        for inning in match.get("innings", []):
                            for over in inning.get("overs", []):
                                balls = []
                                for delivery in over.get("deliveries", []):
                                    if delivery.get("bowler") != bowler_name:
                                        continue

                                    if "wickets" in delivery and delivery["wickets"]:
                                        outcome = -1
                                    else:
                                        outcome = delivery.get("runs", {}).get("batter", 0)
                                        if outcome not in [0, 1, 2, 3, 4, 6]:
                                            outcome = 0  # Normalize extras

                                    balls.append(outcome)

                                if balls:
                                    balls = (balls + [0] * 6)[:6]  # Pad or trim to 6
                                    overs_data.append(balls)

    if overs_data:
        tensor = torch.tensor(overs_data)
        sanitized_name = bowler_name.replace("/", "-")
        torch.save(tensor, f"bowlers/{sanitized_name}.pt")
    # You can also log bowler_name and shape if needed

print("✅ Done.")