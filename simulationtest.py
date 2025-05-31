import json
import pandas as pd
import random
import torch
import os
from player_predictions import predict_next_over, format_over, format_sequence
import time
from datetime import datetime, timedelta

class MatchSimulator:
    def __init__(self, team1_name=None, team2_name=None, toss_winner=None, toss_decision=None):
        # Load team data
        with open('teams.json', 'r') as f:
            teams = json.load(f)
        
        # Initialize match date (starting from May 31st, 2024)
        self.start_date = datetime(2024, 5, 31)
        self.current_match_date = self.start_date
        
        # Initialize points table
        self.points_table = {}
        for team in teams:
            self.points_table[team['name']] = {
                'played': 0,
                'won': 0,
                'lost': 0,
                'points': 0,
                'net_run_rate': 0.0,
                'runs_for': 0,
                'runs_against': 0,
                'overs_played': 0
            }
        
        # If specific match details provided, initialize for single match
        if team1_name and team2_name:
            self.team1_name = team1_name
            self.team2_name = team2_name
            self.toss_winner = toss_winner
            self.toss_decision = toss_decision
            
            # Get team data
            self.team1_data = next((team for team in teams if team['name'].lower() == team1_name.lower()), None)
            self.team2_data = next((team for team in teams if team['name'].lower() == team2_name.lower()), None)
            
            # Initialize match state
            self.first_batting_team = team1_name if toss_decision == 'bat' else team2_name
            self.first_bowling_team = team2_name if toss_decision == 'bat' else team1_name
            
            # Load and sort players
            self._load_players()
            
            # Initialize match data
            self.match_data = {
                'first_innings': None,
                'second_innings': None,
                'target': None,
                'result': None
            }
            
            # Cache for model predictions
            self.prediction_cache = {}

    def _load_players(self):
        """Load and sort players for both teams."""
        self.batting_orders = {}
        self.bowlers = {}
        
        for team_name, team_data in [(self.team1_name, self.team1_data), 
                                   (self.team2_name, self.team2_data)]:
            if not team_data:
                continue
                
            # Load player stats
            player_stats_list = []
            for player in team_data.get('players', []):
                player_name = player['name']
                stats = self._load_player_stats(player_name)
                if stats is not None:
                    # Add name to stats
                    stats['name'] = player_name
                    # Ensure we have a runs field, default to 0 if missing
                    if 'runs' not in stats:
                        stats['runs'] = 0
                    player_stats_list.append(stats)
            
            # Sort batting order strictly by runs (highest to lowest)
            player_stats_list.sort(key=lambda p: p['runs'], reverse=True)
            print(f"\nBatting order for {team_name} (sorted by runs):")
            for p in player_stats_list:
                print(f"{p['name']}: {p['runs']} runs")
            self.batting_orders[team_name] = [p['name'] for p in player_stats_list]
            
            # Sort and store top 5 bowlers (by wickets)
            player_stats_list.sort(key=lambda p: p.get('wickets', 0), reverse=True)
            self.bowlers[team_name] = [p['name'] for p in player_stats_list[:5]]
            print(f"\nBowlers for {team_name} (sorted by wickets):")
            for p in player_stats_list[:5]:
                print(f"{p['name']}: {p.get('wickets', 0)} wickets")

    def _load_player_stats(self, player_name):
        """Load player stats from JSON file."""
        filename = f"players/{player_name}.json"
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Player file not found: {filename}")
            return None

    def _get_cached_prediction(self, player_name, context_overs):
        """Get or create cached prediction for a player."""
        # Convert each over in context_overs to a tuple, then make the whole context a tuple
        context_tuple = tuple(tuple(over) for over in context_overs)
        cache_key = (player_name, context_tuple)
        
        if cache_key not in self.prediction_cache:
            result = predict_next_over(player_name, num_overs=4)
            if result:
                self.prediction_cache[cache_key] = result
            else:
                return None
        return self.prediction_cache[cache_key]

    def simulate_delivery(self, context_overs):
        """Simulate a single delivery using cached predictions."""
        batsman = self.current_innings['current_batsmen'][0]  # Striker
        bowler = self.current_innings['current_bowler']
        
        # Get predictions from cache
        batsman_result = self._get_cached_prediction(batsman, context_overs)
        bowler_result = self._get_cached_prediction(bowler, context_overs)
        
        if not (batsman_result and bowler_result):
            # If prediction fails, use a fallback random outcome
            print(f"Warning: Prediction failed for {batsman} vs {bowler}, using fallback outcome")
            outcomes = ['0', '1', '2', '4', '6', 'W']
            weights = [0.4, 0.3, 0.1, 0.15, 0.03, 0.02]
            outcome = random.choices(outcomes, weights=weights)[0]
            return -1 if outcome == 'W' else int(outcome)
        
        # Get logits for first delivery
        batsman_logits = torch.tensor(batsman_result["logits"][0])
        bowler_logits = torch.tensor(bowler_result["logits"][0])
        
        # Add the logits together to get combined prediction
        # This represents the interaction between batsman and bowler
        combined_logits = batsman_logits + bowler_logits
        
        # Get the outcome with highest combined logit
        outcome_idx = torch.argmax(combined_logits).item()
        
        # Convert index to outcome
        idx_to_label = {0: 'W', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '6'}
        outcome = idx_to_label[outcome_idx]
        
        # Print the battle details for debugging
        print(f"\nDelivery battle: {batsman} vs {bowler}")
        print("Batsman logits:")
        for i, logit in enumerate(batsman_logits):
            print(f"  {idx_to_label[i]}: {logit.item():.3f}")
        print("\nBowler logits:")
        for i, logit in enumerate(bowler_logits):
            print(f"  {idx_to_label[i]}: {logit.item():.3f}")
        print("\nCombined logits:")
        for i, logit in enumerate(combined_logits):
            print(f"  {idx_to_label[i]}: {logit.item():.3f}")
        print(f"\nFinal outcome: {outcome}")
        
        # Convert outcome to numeric value (W becomes -1)
        if outcome == 'W':
            return -1
        return int(outcome)

    def handle_wicket(self):
        """Handle a wicket by bringing in the next batsman."""
        if self.current_innings['next_batsman_idx'] < len(self.batting_orders[self.current_innings['batting_team']]):
            # Replace the striker with next batsman
            self.current_innings['current_batsmen'][0] = self.batting_orders[self.current_innings['batting_team']][self.current_innings['next_batsman_idx']]
            self.current_innings['next_batsman_idx'] += 1
            return True
        return False  # No more batsmen available

    def change_bowler(self):
        """Change the bowler after every over."""
        # Get the next bowler in rotation
        next_bowler = self.bowlers[self.current_innings['bowling_team']][self.current_innings['next_bowler_idx']]
        self.current_innings['current_bowler'] = next_bowler
        
        # Update next bowler index (cycle through available bowlers)
        self.current_innings['next_bowler_idx'] = (self.current_innings['next_bowler_idx'] + 1) % len(self.bowlers[self.current_innings['bowling_team']])
        
        # Update bowler's overs count
        self.current_innings['bowler_overs'][next_bowler] += 1
        
        print(f"\nNew bowler: {next_bowler} (Overs bowled: {self.current_innings['bowler_overs'][next_bowler]})")

    def _initialize_innings(self, batting_team, bowling_team):
        """Initialize innings state for a team."""
        return {
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'overs': [],
            'score': 0,
            'wickets': 0,
            'current_batsmen': self.batting_orders[batting_team][:2],
            'current_bowler': self.bowlers[bowling_team][0],
            'next_batsman_idx': 2,
            'bowler_overs': {bowler: 0 for bowler in self.bowlers[bowling_team]},
            'next_bowler_idx': 1,
            'stats': {
                'total_deliveries': 0,
                'model_failures': 0,
                'forced_overs': 0
            }
        }

    def _save_match_report(self, match_id):
        """Save detailed match report in JSON format."""
        report = {
            "match_id": match_id,
            "teams": {
                "team1": self.team1_name,
                "team2": self.team2_name
            },
            "toss": {
                "winner": self.toss_winner,
                "decision": self.toss_decision
            },
            "innings": []
        }
        
        # Add first innings
        first_innings = {
            "team": self.first_batting_team,
            "overs": []
        }
        
        for over_num, over in enumerate(self.match_data['first_innings']['overs']):
            over_data = {
                "over": over_num,
                "deliveries": []
            }
            
            # Get the batsmen and bowler for this over
            # We need to reconstruct this from the match state
            current_batsmen = self.match_data['first_innings']['current_batsmen']
            current_bowler = self.match_data['first_innings']['current_bowler']
            
            for ball_num, runs in enumerate(over):
                delivery = {
                    "batter": current_batsmen[0],  # Striker
                    "bowler": current_bowler,
                    "non_striker": current_batsmen[1],
                    "runs": {
                        "batter": runs if runs != -1 else 0,  # Convert wicket (-1) to 0 runs
                        "extras": 0,
                        "total": runs if runs != -1 else 0
                    },
                    "wicket": runs == -1
                }
                over_data["deliveries"].append(delivery)
                
                # Update batsmen for next delivery if needed
                if runs % 2 == 1:  # Odd runs
                    current_batsmen = current_batsmen[::-1]
            
            first_innings["overs"].append(over_data)
        
        # Add second innings
        second_innings = {
            "team": self.first_bowling_team,
            "target": self.match_data['target'],
            "overs": []
        }
        
        for over_num, over in enumerate(self.match_data['second_innings']['overs']):
            over_data = {
                "over": over_num,
                "deliveries": []
            }
            
            current_batsmen = self.match_data['second_innings']['current_batsmen']
            current_bowler = self.match_data['second_innings']['current_bowler']
            
            for ball_num, runs in enumerate(over):
                delivery = {
                    "batter": current_batsmen[0],
                    "bowler": current_bowler,
                    "non_striker": current_batsmen[1],
                    "runs": {
                        "batter": runs if runs != -1 else 0,
                        "extras": 0,
                        "total": runs if runs != -1 else 0
                    },
                    "wicket": runs == -1
                }
                over_data["deliveries"].append(delivery)
                
                if runs % 2 == 1:  # Odd runs
                    current_batsmen = current_batsmen[::-1]
            
            second_innings["overs"].append(over_data)
        
        # Add innings to report
        report["innings"].append(first_innings)
        report["innings"].append(second_innings)
        
        # Add match result
        report["result"] = {
            "winner": self.first_batting_team if "won by" in self.match_data['result'] and "runs" in self.match_data['result'] else self.first_bowling_team,
            "margin": self.match_data['result'],
            "first_innings_score": f"{self.match_data['first_innings']['score']}/{self.match_data['first_innings']['wickets']}",
            "second_innings_score": f"{self.match_data['second_innings']['score']}/{self.match_data['second_innings']['wickets']}"
        }
        
        # Add model statistics
        report["model_stats"] = {
            "first_innings": {
                "total_deliveries": self.match_data['first_innings']['stats']['total_deliveries'],
                "model_failures": self.match_data['first_innings']['stats']['model_failures'],
                "forced_overs": self.match_data['first_innings']['stats']['forced_overs']
            },
            "second_innings": {
                "total_deliveries": self.match_data['second_innings']['stats']['total_deliveries'],
                "model_failures": self.match_data['second_innings']['stats']['model_failures'],
                "forced_overs": self.match_data['second_innings']['stats']['forced_overs']
            }
        }
        
        # Save to file
        filename = f"match reports/match_{match_id}.json"
        os.makedirs("match reports", exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nMatch report saved to {filename}")
        return filename

    def simulate_match(self):
        """Simulate both innings of the match."""
        print(f"\nMatch: {self.team1_name} vs {self.team2_name}")
        print(f"Date: {self.current_match_date.strftime('%B %d, %Y')}")
        print(f"Toss: {self.toss_winner} won and chose to {self.toss_decision}")
        
        # Generate match ID from date
        match_id = self.current_match_date.strftime('%Y%m%d')
        
        # First innings
        print("\n=== FIRST INNINGS ===")
        self.current_innings = self._initialize_innings(self.first_batting_team, self.first_bowling_team)
        first_innings_result = self.simulate_innings()
        # Store complete innings data including stats from current_innings
        self.match_data['first_innings'] = {
            'batting_team': first_innings_result['batting_team'],
            'bowling_team': first_innings_result['bowling_team'],
            'overs': first_innings_result['overs'],
            'score': first_innings_result['score'],
            'wickets': first_innings_result['wickets'],
            'current_batsmen': first_innings_result['current_batsmen'],
            'current_bowler': first_innings_result['current_bowler'],
            'stats': self.current_innings['stats']  # Get stats from current_innings
        }
        
        # Calculate target
        self.match_data['target'] = first_innings_result['score'] + 1
        
        # Second innings
        print("\n=== SECOND INNINGS ===")
        self.current_innings = self._initialize_innings(self.first_bowling_team, self.first_batting_team)
        print(f"Target: {self.match_data['target']}")
        second_innings_result = self.simulate_innings()
        # Store complete innings data including stats from current_innings
        self.match_data['second_innings'] = {
            'batting_team': second_innings_result['batting_team'],
            'bowling_team': second_innings_result['bowling_team'],
            'overs': second_innings_result['overs'],
            'score': second_innings_result['score'],
            'wickets': second_innings_result['wickets'],
            'current_batsmen': second_innings_result['current_batsmen'],
            'current_bowler': second_innings_result['current_bowler'],
            'stats': self.current_innings['stats']  # Get stats from current_innings
        }
        
        # Determine result
        if second_innings_result['wickets'] >= 10:
            self.match_data['result'] = f"{self.first_batting_team} won by {self.match_data['target'] - second_innings_result['score'] - 1} runs"
        elif second_innings_result['score'] >= self.match_data['target']:
            wickets_left = 10 - second_innings_result['wickets']
            self.match_data['result'] = f"{self.first_bowling_team} won by {wickets_left} wickets"
        else:
            self.match_data['result'] = f"{self.first_batting_team} won by {self.match_data['target'] - second_innings_result['score'] - 1} runs"
        
        print(f"\n=== MATCH RESULT ===")
        print(f"First innings: {self.first_batting_team} {first_innings_result['score']}/{first_innings_result['wickets']}")
        print(f"Second innings: {self.first_bowling_team} {second_innings_result['score']}/{second_innings_result['wickets']}")
        print(f"Result: {self.match_data['result']}")
        
        # Save match report
        report_file = self._save_match_report(match_id)
        
        return self.match_data

    def simulate_innings(self):
        """Simulate a complete innings (max 20 overs)."""
        context_overs = [[0, 0, 0, 0, 0, 0] for _ in range(4)]  # Initialize with dot balls
        
        print(f"\nStarting innings: {self.current_innings['batting_team']} vs {self.current_innings['bowling_team']}")
        print(f"Opening batsmen: {self.current_innings['current_batsmen'][0]} and {self.current_innings['current_batsmen'][1]}")
        print(f"Opening bowler: {self.current_innings['current_bowler']}")
        print("\nBowling order:")
        for i, bowler in enumerate(self.bowlers[self.current_innings['bowling_team']]):
            print(f"{i+1}. {bowler}")
        
        min_overs = 5  # Ensure at least 5 overs are played unless all out or target reached
        overs_played = 0
        
        for over_num in range(20):  # Maximum 20 overs
            if self.current_innings['wickets'] >= 10:  # All out
                print("\nAll out!")
                break
                
            # Check if target is reached (for second innings)
            if self.match_data['target'] and self.current_innings['score'] >= self.match_data['target']:
                print("\nTarget reached!")
                break
                
            print(f"\nOver {over_num + 1}: {self.current_innings['current_batsmen'][0]} facing {self.current_innings['current_bowler']}")
            if self.match_data['target']:
                print(f"Target: {self.match_data['target']}, Current: {self.current_innings['score']}, Need: {self.match_data['target'] - self.current_innings['score']}")
            
            over = []
            valid_deliveries = 0  # Count valid deliveries in this over
            
            # Simulate each delivery in the over
            for ball in range(6):
                if self.current_innings['wickets'] >= 10:  # Check for all out after each delivery
                    break
                    
                # Check if target is reached (for second innings)
                if self.match_data['target'] and self.current_innings['score'] >= self.match_data['target']:
                    break
                    
                outcome = self.simulate_delivery(context_overs)
                if outcome is None:
                    continue  # Skip this delivery but continue the over
                    
                over.append(outcome)
                valid_deliveries += 1
                
                # Handle wicket
                if outcome == -1:
                    self.current_innings['wickets'] += 1
                    if not self.handle_wicket():
                        break  # All out
                else:
                    # Rotate strike for odd runs (1, 3, 5)
                    if outcome % 2 == 1:
                        self.current_innings['current_batsmen'].reverse()
                    self.current_innings['score'] += outcome
                    
                    # Check if target is reached after each run (for second innings)
                    if self.match_data['target'] and self.current_innings['score'] >= self.match_data['target']:
                        break
                
                # Update context overs for next delivery
                context_overs = context_overs[1:] + [over]
            
            # Only count the over if we had valid deliveries
            if valid_deliveries > 0:
                self.current_innings['overs'].append(over)
                overs_played += 1
                
                print(f"Over: {format_over(over)}")
                print(f"Score: {self.current_innings['score']}/{self.current_innings['wickets']}")
                print(f"Current batsmen: {self.current_innings['current_batsmen'][0]}* and {self.current_innings['current_batsmen'][1]}")
                
                # Change bowler after every over
                self.change_bowler()
                
                # Rotate strike at the end of the over (except for the last over)
                if over_num < 19 and not (self.match_data['target'] and self.current_innings['score'] >= self.match_data['target']):
                    self.current_innings['current_batsmen'].reverse()
            
            # Check if we've played minimum overs and can end innings
            if overs_played >= min_overs:
                if self.match_data['target'] and self.current_innings['score'] >= self.match_data['target']:
                    break
                if self.current_innings['wickets'] >= 10:
                    break
        
        # Ensure we have a valid innings
        if not self.current_innings['overs']:
            print("Warning: No valid overs were played, forcing at least one over")
            # Force at least one over with random outcomes
            forced_over = []
            for _ in range(6):
                outcomes = ['0', '1', '2', '4', '6', 'W']
                weights = [0.4, 0.3, 0.1, 0.15, 0.03, 0.02]
                outcome = random.choices(outcomes, weights=weights)[0]
                forced_over.append(-1 if outcome == 'W' else int(outcome))
            self.current_innings['overs'].append(forced_over)
            self.current_innings['score'] = sum(ball for ball in forced_over if ball != -1)
            self.current_innings['wickets'] = forced_over.count(-1)
        
        print(f"\nInnings complete: {self.current_innings['score']}/{self.current_innings['wickets']} in {len(self.current_innings['overs'])} overs")
        print("\nBowling summary:")
        for bowler, overs in self.current_innings['bowler_overs'].items():
            print(f"{bowler}: {overs} overs")
        return self.current_innings

    def _update_points_table(self, match_result):
        """Update points table based on match result."""
        # Extract teams and scores from match result
        first_team = match_result['first_innings']['batting_team']
        second_team = match_result['second_innings']['batting_team']
        
        first_score = match_result['first_innings']['score']
        first_wickets = match_result['first_innings']['wickets']
        second_score = match_result['second_innings']['score']
        second_wickets = match_result['second_innings']['wickets']
        
        # Update matches played
        self.points_table[first_team]['played'] += 1
        self.points_table[second_team]['played'] += 1
        
        # Update runs and overs
        first_overs = len(match_result['first_innings']['overs'])
        second_overs = len(match_result['second_innings']['overs'])
        
        self.points_table[first_team]['runs_for'] += first_score
        self.points_table[first_team]['runs_against'] += second_score
        self.points_table[first_team]['overs_played'] += first_overs
        
        self.points_table[second_team]['runs_for'] += second_score
        self.points_table[second_team]['runs_against'] += first_score
        self.points_table[second_team]['overs_played'] += second_overs
        
        # Determine winner and update points
        if "won by" in match_result['result']:
            if first_team in match_result['result']:
                # First team won
                self.points_table[first_team]['won'] += 1
                self.points_table[first_team]['points'] += 2
                self.points_table[second_team]['lost'] += 1
            else:
                # Second team won
                self.points_table[second_team]['won'] += 1
                self.points_table[second_team]['points'] += 2
                self.points_table[first_team]['lost'] += 1
        
        # Update net run rates
        for team in [first_team, second_team]:
            if self.points_table[team]['overs_played'] > 0:
                runs_per_over = self.points_table[team]['runs_for'] / self.points_table[team]['overs_played']
                runs_against_per_over = self.points_table[team]['runs_against'] / self.points_table[team]['overs_played']
                self.points_table[team]['net_run_rate'] = runs_per_over - runs_against_per_over

    def _save_points_table(self):
        """Save current points table to a JSON file."""
        # Sort teams by points (descending) and then by net run rate (descending)
        sorted_teams = sorted(
            self.points_table.items(),
            key=lambda x: (-x[1]['points'], -x[1]['net_run_rate'])
        )
        
        # Create a list of team standings
        standings = []
        for position, (team, stats) in enumerate(sorted_teams, 1):
            standings.append({
                'position': position,
                'team': team,
                **stats
            })
        
        # Save to file
        filename = "match reports/points_table.json"
        with open(filename, 'w') as f:
            json.dump(standings, f, indent=2)
        
        print(f"\nPoints table saved to {filename}")
        return filename

    def simulate_all_matches(self):
        """Simulate all matches from randomized_team_matches.json."""
        # Load matches
        with open('randomized_team_matches.json', 'r') as f:
            matches = json.load(f)
        
        print(f"\nSimulating {len(matches)} matches...")
        
        for match_num, match in enumerate(matches, 1):
            print(f"\n=== Match {match_num}/{len(matches)} ===")
            print(f"Date: {self.current_match_date.strftime('%B %d, %Y')}")
            
            # Get match details
            team1_name = match['team1']
            team2_name = match['team2']
            toss_winner = random.choice([team1_name, team2_name])
            toss_decision = random.choice(['bat', 'bowl'])
            
            # Create new simulator instance for this match
            simulator = MatchSimulator(team1_name, team2_name, toss_winner, toss_decision)
            simulator.current_match_date = self.current_match_date  # Set the date for this match
            
            # Simulate match
            match_result = simulator.simulate_match()
            
            # Update points table
            self._update_points_table(match_result)
            
            # Save points table after each match
            self._save_points_table()
            
            # Increment date for next match
            self.current_match_date += timedelta(days=1)
        
        print("\n=== TOURNAMENT COMPLETE ===")
        print("\nFinal Points Table:")
        sorted_teams = sorted(
            self.points_table.items(),
            key=lambda x: (-x[1]['points'], -x[1]['net_run_rate'])
        )
        for position, (team, stats) in enumerate(sorted_teams, 1):
            print(f"{position}. {team}: {stats['points']} points (W: {stats['won']}, L: {stats['lost']}, NRR: {stats['net_run_rate']:.3f})")

if __name__ == "__main__":
    # Load match data
    with open('randomized_team_matches.json', 'r') as f:
        matches = json.load(f)

    # Get first match details
    match = matches[0]  # Get the first match
    team1_name = match['team1']
    team2_name = match['team2']
    toss_winner = random.choice([team1_name, team2_name])
    toss_decision = random.choice(['bat', 'bowl'])

    print(f"\nSimulating single match: {team1_name} vs {team2_name}")
    print(f"Toss: {toss_winner} won and chose to {toss_decision}")

    # Create and run simulation
    simulator = MatchSimulator(team1_name, team2_name, toss_winner, toss_decision)
    match_result = simulator.simulate_match()