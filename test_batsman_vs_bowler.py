from player_predictions import predict_next_over, format_over, format_sequence
import torch

def test_batsman_vs_bowler():
    # Test with a batsman and bowler who have both data and models
    batsman_name = "V Kohli"
    bowler_name = "JJ Bumrah"
    
    print(f"\nSimulating {batsman_name} facing {bowler_name}:")
    
    # Get predictions for both players
    batsman_result = predict_next_over(batsman_name, num_overs=4)
    bowler_result = predict_next_over(bowler_name, num_overs=4)
    
    if batsman_result and bowler_result:
        print("\nInput sequence (4 overs):")
        print(format_sequence(batsman_result["input_sequence"]))
        
        # Get logits for first delivery from both players
        batsman_logits = torch.tensor(batsman_result["logits"][0])
        bowler_logits = torch.tensor(bowler_result["logits"][0])
        
        # Print raw logits for both players
        print(f"\nBatsman ({batsman_name}) logits for next delivery:")
        print(batsman_logits)
        print(f"\nBowler ({bowler_name}) logits for next delivery:")
        print(bowler_logits)
        
        # Compare logits to determine outcome
        # For each possible outcome (W,0,1,2,3,4,6), take the higher logit
        combined_logits = torch.maximum(batsman_logits, bowler_logits)
        outcome_idx = torch.argmax(combined_logits).item()
        
        # Convert index to outcome
        idx_to_label = {0: 'W', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '6'}
        outcome = idx_to_label[outcome_idx]
        
        print(f"\nOutcome determined by highest logit: {outcome}")
        print(f"Batsman's highest logit: {batsman_logits[outcome_idx]:.2f}")
        print(f"Bowler's highest logit: {bowler_logits[outcome_idx]:.2f}")
    else:
        print(f"Could not get predictions for both players")

if __name__ == "__main__":
    test_batsman_vs_bowler() 