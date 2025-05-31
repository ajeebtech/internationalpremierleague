from player_predictions import predict_next_over, format_over, format_sequence
import torch

def test_batsman_delivery():
    # Test with a batsman who has both data and model
    batsman_name = "V Kohli"  # Make sure this batsman has both data in batsmen/ and model in batsmen_models/
    
    print(f"\nPredicting next delivery for {batsman_name} based on 4 consecutive overs from their data:")
    result = predict_next_over(batsman_name, num_overs=4)
    
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
        print(f"Could not predict delivery for {batsman_name}")

if __name__ == "__main__":
    test_batsman_delivery() 