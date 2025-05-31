from player_predictions import get_player_predictions, update_players_with_predictions

# Example 1: Get predictions for a single player
def test_single_player():
    # Create a player dictionary (similar to what's in teams.json)
    player = {
        "name": "JJ Bumrah",  # Make sure this matches a .pt file in bowler_models/
        "league": "IPL",
        "wickets": 145,  # This makes it a bowler (>= 50 wickets)
        "runs": 100
    }
    
    # Get predictions
    predictions = get_player_predictions(player)
    if predictions:
        print(f"\nPredictions for {player['name']}:")
        print(f"Predicted wickets: {predictions['predicted_wickets']}")
        print(f"Predicted runs conceded: {predictions['predicted_runs']}")
        print(f"Predicted economy rate: {predictions['predicted_economy']:.2f}")
        print(f"Predicted average: {predictions['predicted_average']:.2f}")
    else:
        print(f"\nNo model found for {player['name']}")

# Example 2: Get predictions for multiple players
def test_multiple_players():
    # Create a list of players
    players = [
        {
            "name": "JJ Bumrah",
            "league": "IPL",
            "wickets": 145,
            "runs": 100
        },
        {
            "name": "Virat Kohli",  # This would be a batsman
            "league": "IPL",
            "wickets": 0,
            "runs": 7263
        }
    ]
    
    # Update all players with their predictions
    updated_players = update_players_with_predictions(players)
    
    print("\nPredictions for multiple players:")
    for player in updated_players:
        if "predicted_wickets" in player:
            print(f"\n{player['name']}:")
            print(f"Predicted wickets: {player['predicted_wickets']}")
            print(f"Predicted runs conceded: {player['predicted_runs']}")
            print(f"Predicted economy rate: {player['predicted_economy']:.2f}")
            print(f"Predicted average: {player['predicted_average']:.2f}")
        else:
            print(f"\nNo model found for {player['name']}")

if __name__ == "__main__":
    print("Testing single player predictions...")
    test_single_player()
    
    print("\nTesting multiple player predictions...")
    test_multiple_players() 