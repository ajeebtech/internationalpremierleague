import json
import pandas as pd
data = {}    # all this data is for match reports
match_no = 0

# Load the teams.csv file into a DataFrame
df = pd.read_csv('teams.csv')
df = df.set_index('team')

# Load the randomized_team_matches.json file
with open('randomized_team_matches.json', 'r') as f:
    matches = json.load(f)
with open('teams.json', 'r') as f:
    teams = json.load(f)

stadium = df.loc[matches[0].get('team1'),'stadium']
print('Stadium for the first match:', stadium)

data['stadium'] = stadium
data['match_number'] = match_no
data['match_number'] += 1

team1_name = matches[0].get('team1')
team1_data = next((team for team in teams if team['name'].lower() == team1_name.lower()), None)
team2_name = matches[0].get('team2')
team2_data = next((team for team in teams if team['name'].lower() == team2_name.lower()), None)

if team1_data:
    team1_list = team1_data.get('players', [])


if team2_data:
    team2_list = team2_data.get('players', [])

data['players'] = {}

data['players'][team1_name] = [player['name'] for player in team1_list]
data['players'][team2_name] = [player['name'] for player in team2_list]
print(data)
data['teams'] = [team1_name, team2_name]
