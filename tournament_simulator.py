import json
import pandas as pd
import random
from simulationtest import MatchSimulator
import os
import time
from tqdm import tqdm

class TournamentSimulator:
    def __init__(self):
        # Load match schedule
        with open('randomized_team_matches.json', 'r') as f:
            self.matches = json.load(f)
        
        # Initialize points table
        self.points_table = {}
        self.initialize_points_table()
        
        # Create directory for match reports
        os.makedirs("match reports", exist_ok=True)
        
    def initialize_points_table(self):
        """Initialize points table for all teams."""
        # Get unique teams from matches
        teams = set()
        for match in self.matches:
            teams.add(match['team1'])
            teams.add(match['team2'])
        
        # Initialize points for each team
        for team in teams:
            self.points_table[team] = {
                'played': 0,
                'won': 0,
                'lost': 0,
                'points': 0,
                'net_run_rate': 0.0,
                'runs_for': 0,
                'runs_against': 0,
                'overs_for': 0,
                'overs_against': 0
            }
    
    def update_points_table(self, match_result):
        """Update points table based on match result."""
        first_innings = match_result['first_innings']
        second_innings = match_result['second_innings']
        
        # Get teams
        team1 = first_innings['batting_team']
        team2 = second_innings['batting_team']
        
        # Update matches played
        self.points_table[team1]['played'] += 1
        self.points_table[team2]['played'] += 1
        
        # Update runs and overs
        self.points_table[team1]['runs_for'] += first_innings['score']
        self.points_table[team1]['runs_against'] += second_innings['score']
        self.points_table[team1]['overs_for'] += len(first_innings['overs'])
        self.points_table[team1]['overs_against'] += len(second_innings['overs'])
        
        self.points_table[team2]['runs_for'] += second_innings['score']
        self.points_table[team2]['runs_against'] += first_innings['score']
        self.points_table[team2]['overs_for'] += len(second_innings['overs'])
        self.points_table[team2]['overs_against'] += len(first_innings['overs'])
        
        # Determine winner and update points
        if "won by" in match_result['result']:
            if "runs" in match_result['result']:
                # Team batting first won
                self.points_table[team1]['won'] += 1
                self.points_table[team1]['points'] += 2
                self.points_table[team2]['lost'] += 1
            else:
                # Team batting second won
                self.points_table[team2]['won'] += 1
                self.points_table[team2]['points'] += 2
                self.points_table[team1]['lost'] += 1
        
        # Calculate net run rates
        for team in [team1, team2]:
            if self.points_table[team]['overs_for'] > 0 and self.points_table[team]['overs_against'] > 0:
                run_rate_for = self.points_table[team]['runs_for'] / self.points_table[team]['overs_for']
                run_rate_against = self.points_table[team]['runs_against'] / self.points_table[team]['overs_against']
                self.points_table[team]['net_run_rate'] = run_rate_for - run_rate_against
    
    def print_points_table(self):
        """Print the current points table."""
        print("\n=== POINTS TABLE ===")
        print("Team\t\tP\tW\tL\tPts\tNRR")
        print("-" * 50)
        
        # Sort teams by points (descending) and then by net run rate (descending)
        sorted_teams = sorted(self.points_table.items(), 
                            key=lambda x: (-x[1]['points'], -x[1]['net_run_rate']))
        
        for team, stats in sorted_teams:
            print(f"{team:<15}\t{stats['played']}\t{stats['won']}\t{stats['lost']}\t{stats['points']}\t{stats['net_run_rate']:.3f}")
    
    def save_points_table(self):
        """Save points table to JSON file."""
        filename = "match reports/points_table.json"
        with open(filename, 'w') as f:
            json.dump(self.points_table, f, indent=2)
        print(f"\nPoints table saved to {filename}")
    
    def simulate_all_matches(self):
        """Simulate all matches in the tournament."""
        print("Starting tournament simulation...")
        
        # Create progress bar
        pbar = tqdm(self.matches, desc="Simulating matches", unit="match")
        
        for i, match in enumerate(pbar):
            # Update progress bar description with current match
            pbar.set_description(f"Match: {match['team1']} vs {match['team2']}")
            
            # Simulate match
            toss_winner = random.choice([match['team1'], match['team2']])
            toss_decision = random.choice(['bat', 'bowl'])
            
            simulator = MatchSimulator(match['team1'], match['team2'], toss_winner, toss_decision)
            match_result = simulator.simulate_match()
            
            # Update points table
            self.update_points_table(match_result)
            
            # Print updated points table after each match
            self.print_points_table()
            
            # Add a 5-minute break after every 100 games
            if (i + 1) % 100 == 0 and i + 1 < len(self.matches):
                print(f"\nCompleted {i + 1} matches. Taking a 5-minute break...")
                time.sleep(300)  # 5 minutes = 300 seconds
        
        # Save final points table
        self.save_points_table()
        print("\nTournament simulation complete!")

if __name__ == "__main__":
    # Create and run tournament simulation
    tournament = TournamentSimulator()
    tournament.simulate_all_matches() 