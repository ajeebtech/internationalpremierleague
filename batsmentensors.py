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

for batter_name in tqdm(player_names, desc="Processing batsmen"):
    runs_data = []

    for subfolder in sorted(os.listdir(root_dir)):  # Alphabetical order for folders
        subfolder_path = os.path.join(root_dir, subfolder)
        if os.path.isdir(subfolder_path) and subfolder.endswith("_json"):
            for root, dirs, files in os.walk(subfolder_path):
                dirs.sort()   # Ensure os.walk is deterministic (sort subdirs)
                for file in sorted(files):  # Alphabetical order for files
                    if file.endswith(".json"):
                        json_path = os.path.join(root, file)
                        try:
                            with open(json_path, "r") as f:
                                match = json.load(f)
                        except Exception:
                            continue
                        for inning in match.get("innings", []):
                            for over in inning.get("overs", []):
                                for delivery in over.get("deliveries", []):
                                    if delivery.get("batter") != batter_name:
                                        continue  # Only when batter is on strike
                                    if "wickets" in delivery:
                                        for w in delivery["wickets"]:
                                            if w.get("player_out") == batter_name:
                                                run = -1  # Dismissal
                                                break
                                        else:
                                            run = delivery.get("runs", {}).get("batter", 0)
                                    else:
                                        run = delivery.get("runs", {}).get("batter", 0)

                                    runs_data.append(run)

    # Split runs_data into groups of 6 balls
    if runs_data:
        six_ball_overs = [runs_data[i:i+6] for i in range(0, len(runs_data), 6) if len(runs_data[i:i+6]) == 6]
        if six_ball_overs:
            tensor = torch.tensor(six_ball_overs)
            sanitized_name = batter_name.replace("/", "-")
            os.makedirs("batsmen", exist_ok=True)
            torch.save(tensor, f"batsmen/{sanitized_name}.pt")

print("✅ Done.")