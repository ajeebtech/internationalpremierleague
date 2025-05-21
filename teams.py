import os
import json
import csv

# Root directory containing the folders
root_dir = '/Users/jatin/Documents/python/multipremierleague'
output_file = 'teams.csv'

# Use a set to avoid duplicates
team_folder_pairs = set()

# Traverse all folders and files
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.json'):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    players = data.get('info', {}).get('players', {})
                    foldername = os.path.basename(root)
                    for team in players.keys():
                        team_folder_pairs.add((team, foldername.replace('_json', '').upper()))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# Write to CSV
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['team', 'league'])
    writer.writerows(sorted(team_folder_pairs))  # optional: sorted for readability

print(f"Saved unique team-folder pairs to {output_file}")