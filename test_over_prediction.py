from player_predictions import predict_next_over, format_over, format_sequence
import torch

def test_bowler_over():
    # Test with a bowler who has both data and model
    bowler_name = "JJ Bumrah"  # Make sure this bowler has both data in bowlers/ and model in bowler_models/
    
    print(f"\nPredicting next delivery for {bowler_name} based on 4 consecutive overs from their data:")
    result = predict_next_over(bowler_name, num_overs=4)
    
    if result:
        print("\nInput sequence (4 overs):")
        print(format_sequence(result["input_sequence"]))
        
        # Print raw logits for the first delivery
        ball = result["predicted_over"][0]
        logits = result["logits"][0]
        ball_str = "W" if ball == -1 else str(ball)
        print(f"\nRaw logits for next delivery (predicted {ball_str}):")
        print(logits)
    else:
        print(f"Could not predict delivery for {bowler_name}")

if __name__ == "__main__":
    test_bowler_over()