import csv
import json

# Read the CSV file
with open('teams.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    teams_list = []
    
    for row in reader:
        team_name = row['team'].strip()
        league = row['league'].strip()
        stadium = row['stadium'].strip()
        
        # Generate shortName (first letters of each word in team name)
        short_name = ''.join([word[0] for word in team_name.split()]).upper()
        
        team_entry = {
            "name": team_name,
            "shortName": short_name,
            "league": league,
            "players": [],
            "CanBuyOverseas": True,
            "wins": 0,
            "points": 0,
            "CanBuy": True,
            "HomeGround": stadium,
            "bowlers": [],
            "wicketkeepers": 0,
            "overseas": 0
        }
        
        teams_list.append(team_entry)

# Write to JSON file
with open('teams.json', 'w', encoding='utf-8') as outfile:
    json.dump(teams_list, outfile, indent=4)