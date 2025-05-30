import json
import os
import random
import sys

# --- Configuration ---
PLAYERS_DIR = "players"  # folder containing player JSONs (e.g. players/WRS Gidman.json)
TEAMS_JSON = "teams.json"  # teams file (updated in place)
# --- End Configuration ---

# --- Helper Functions ---

def load_player_json(players_dir):
    """Loads all player JSONs from players_dir and returns a list of player dicts (each with a 'name' (filename) and player stats)."""
    players = []
    for filename in os.listdir(players_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(players_dir, filename)
            try:
                with open(file_path, "r") as f:
                    player = json.load(f)
                    # Use filename (without .json) as player name
                    player["name"] = os.path.splitext(filename)[0]
                    players.append(player)
            except Exception as e:
                print(f"Error reading {file_path}: {e}", file=sys.stderr)
    return players


def categorize_players(players):
    """Categorizes players into (in_league, overseas) and (bowler, batsman) buckets.
       A bowler is defined as a player with wickets >= 50.
       Returns a dict (e.g. { "in_league": { "bowler": [player1, ...], "batsman": [player1, ...] }, "overseas": { "bowler": [...], "batsman": [...] } }.
    """
    in_league = {}
    overseas = {}
    for p in players:
        league = p.get("league", "").strip()
        wickets = p.get("wickets", 0)
        is_bowler = (wickets >= 50)
        if league not in in_league:
            in_league[league] = {"bowler": [], "batsman": []}
        if "" not in overseas:
            overseas[""] = {"bowler": [], "batsman": []}
        if is_bowler:
            in_league[league]["bowler"].append(p)
            overseas[""]["bowler"].append(p) if league == "" else None
        else:
            in_league[league]["batsman"].append(p)
            overseas[""]["batsman"].append(p) if league == "" else None
    return in_league, overseas


def draft_players_for_team(team, all_players, draft_log, used_players):
    league = team.get("league", "").strip()
    if not league:
        draft_log.append(f"Team {team['name']} has no league; skipping draft.")
        return []
    draft_log.append(f"Team {team['name']} (league: {league}) draft log:")
    first_player = (team.get("players", []) or [None])[0]
    if first_player is None:
        draft_log.append("  No first player found; skipping draft.")
        return []
    first_name = first_player.get("name", "").strip()
    if not first_name:
        draft_log.append("  First player has no name; skipping draft.")
        return []
    
    # Build in-league and overseas pools for this team
    in_league_bowlers = [p for p in all_players if p.get("league", "").strip() == league and p.get("wickets", 0) >= 50 and p["name"] not in used_players and p["name"] != first_name]
    in_league_batsmen = [p for p in all_players if p.get("league", "").strip() == league and p.get("wickets", 0) < 50 and p["name"] not in used_players and p["name"] != first_name]
    overseas_bowlers = [p for p in all_players if p.get("league", "").strip() != league and p.get("league", "").strip() != "" and p.get("wickets", 0) >= 50 and p["name"] not in used_players and p["name"] != first_name]
    overseas_batsmen = [p for p in all_players if p.get("league", "").strip() != league and p.get("league", "").strip() != "" and p.get("wickets", 0) < 50 and p["name"] not in used_players and p["name"] != first_name]
    
    # Get first player's league and bowler status
    first_is_bowler = (first_player.get("wickets", 0) >= 50)
    first_is_in_league = (first_player.get("league", "").strip() == league)
    first_is_overseas = not first_is_in_league and first_player.get("league", "").strip() != ""
    
    # Initialize counts
    in_league_count = 1 if first_is_in_league else 0
    overseas_count = 1 if first_is_overseas else 0
    bowler_count = 1 if first_is_bowler else 0
    batsman_count = 1 if not first_is_bowler else 0
    
    drafted = [first_player]
    used_players.add(first_name)
    
    # Debug logging for first player
    draft_log.append(f"  First player: {first_name}")
    draft_log.append(f"  First player league: {first_player.get('league', '')}")
    draft_log.append(f"  First player is in league: {first_is_in_league}")
    draft_log.append(f"  First player is overseas: {first_is_overseas}")
    draft_log.append(f"  Initial counts - in_league: {in_league_count}, overseas: {overseas_count}")
    
    # First, ensure we have exactly 4 overseas players by drafting exactly 3 more if first player is overseas
    # or exactly 4 more if first player is not overseas
    overseas_needed = 4 - overseas_count
    draft_log.append(f"  Need to draft {overseas_needed} more overseas players")
    
    while overseas_count < 4:
        pool = []
        if overseas_bowlers:
            pool += ["bowler"] * len(overseas_bowlers)
        if overseas_batsmen:
            pool += ["batsman"] * len(overseas_batsmen)
        if not pool:
            draft_log.append(f"  Not enough overseas players available to meet quota. Current overseas count: {overseas_count}")
            break
        pick_type = random.choice(pool)
        if pick_type == "bowler":
            p = random.choice(overseas_bowlers)
            drafted.append(p)
            used_players.add(p["name"])
            overseas_bowlers.remove(p)
            overseas_count += 1
            bowler_count += 1
            draft_log.append(f"  Drafted overseas bowler: {p['name']} (league: {p.get('league', '')}, wickets: {p.get('wickets', 0)}). Overseas count now: {overseas_count}")
        else:
            p = random.choice(overseas_batsmen)
            drafted.append(p)
            used_players.add(p["name"])
            overseas_batsmen.remove(p)
            overseas_count += 1
            draft_log.append(f"  Drafted overseas batsman: {p['name']} (league: {p.get('league', '')}, runs: {p.get('runs', 0)}). Overseas count now: {overseas_count}")
    
    # Then fill remaining slots with in-league players, prioritizing bowlers if needed
    while len(drafted) < 11:
        # If we need more bowlers, prioritize them
        if bowler_count < 5 and in_league_bowlers:
            p = random.choice(in_league_bowlers)
            drafted.append(p)
            used_players.add(p["name"])
            in_league_bowlers.remove(p)
            in_league_count += 1
            bowler_count += 1
            draft_log.append(f"  Drafted in-league bowler: {p['name']} (league: {p.get('league', '')}, wickets: {p.get('wickets', 0)})")
        elif in_league_batsmen:
            p = random.choice(in_league_batsmen)
            drafted.append(p)
            used_players.add(p["name"])
            in_league_batsmen.remove(p)
            in_league_count += 1
            draft_log.append(f"  Drafted in-league batsman: {p['name']} (league: {p.get('league', '')}, runs: {p.get('runs', 0)})")
        else:
            draft_log.append("  Not enough in-league players available to complete team.")
            break
    
    # Final verification of overseas players
    overseas_players = [p for p in drafted if p.get("league", "").strip() != league and p.get("league", "").strip() != ""]
    draft_log.append(f"  Final verification - Overseas players: {[(p['name'], p.get('league', '')) for p in overseas_players]}")
    draft_log.append(f"  Final counts - in_league: {in_league_count}, overseas: {len(overseas_players)}, bowlers: {bowler_count}, total: {len(drafted)}")
    
    # Verify we have exactly 4 overseas players
    if len(overseas_players) != 4:
        draft_log.append(f"  WARNING: Team has {len(overseas_players)} overseas players instead of 4!")
        # If we don't have 4 overseas players, try to fix it by replacing an in-league player
        if len(overseas_players) < 4 and (overseas_bowlers or overseas_batsmen):
            draft_log.append("  Attempting to fix overseas player count...")
            # Remove an in-league player
            for i, p in enumerate(drafted):
                if p.get("league", "").strip() == league:
                    removed = drafted.pop(i)
                    used_players.remove(removed["name"])
                    draft_log.append(f"  Removed in-league player: {removed['name']}")
                    break
            # Add an overseas player
            if overseas_bowlers:
                p = random.choice(overseas_bowlers)
                drafted.append(p)
                used_players.add(p["name"])
                draft_log.append(f"  Added overseas bowler: {p['name']} (league: {p.get('league', '')})")
            elif overseas_batsmen:
                p = random.choice(overseas_batsmen)
                drafted.append(p)
                used_players.add(p["name"])
                draft_log.append(f"  Added overseas batsman: {p['name']} (league: {p.get('league', '')})")
            # Verify again
            overseas_players = [p for p in drafted if p.get("league", "").strip() != league and p.get("league", "").strip() != ""]
            draft_log.append(f"  After fix - Overseas players: {[(p['name'], p.get('league', '')) for p in overseas_players]}")
    
    return drafted


# --- Main ---

if __name__ == "__main__":
    # --- 1. Load all player JSONs ---
    print("Reading player JSONs from folder: " + PLAYERS_DIR)
    players = load_player_json(PLAYERS_DIR)
    print(f"Loaded {len(players)} players.")
    # --- 2. Categorize players ---
    in_league, overseas = categorize_players(players)
    print("Categorization summary:")
    for league, data in in_league.items():
        print(f"  in_league {league} bowler: " + str(len(data["bowler"])))
        print(f"  in_league {league} batsman: " + str(len(data["batsman"])))
    for league, data in overseas.items():
        print(f"  overseas {league} bowler: " + str(len(data["bowler"])))
        print(f"  overseas {league} batsman: " + str(len(data["batsman"])))
    # --- 3. Load teams.json ---
    print("Reading teams from: " + TEAMS_JSON)
    with open(TEAMS_JSON, "r") as f:
         teams = json.load(f)
    print(f"Loaded {len(teams)} teams.")
    # --- 4. Draft players for each team ---
    draft_log = []
    used_players = set()  # to avoid picking the same player for multiple teams
    for team in teams:
         drafted = draft_players_for_team(team, players, draft_log, used_players)
         if drafted:
              team["players"] = drafted
    # --- 5. Write updated teams back to teams.json ---
    print("Writing updated teams (with drafted players) to: " + TEAMS_JSON)
    with open(TEAMS_JSON, "w") as f:
         json.dump(teams, f, indent=4)
    print("Draft log (summary):")
    for log in draft_log:
         print(log) 