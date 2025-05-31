import os
import sys
import torch
import torch.nn as nn
import random

# --- Configuration ---
BOWLER_MODELS_DIR = "bowler_models"  # directory containing bowler prediction models
BATSMEN_MODELS_DIR = "batsmen_models"  # directory containing batsmen prediction models
BOWLERS_DATA_DIR = "bowlers"  # directory containing bowler sequences
# --- End Configuration ---

# --- Model Definition ---
class BallPredictor(nn.Module):
    def __init__(self, vocab_size=7, embed_dim=16, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.rnn(x)
        return self.fc(h.squeeze(0))

# Label mapping (same as training)
label_to_idx = {'-1': 0, '0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '6': 6}
idx_to_label = {v: k for k, v in label_to_idx.items()}

def load_player_model(player_name, is_bowler=True):
    """Load a player's prediction model if available."""
    model_dir = BOWLER_MODELS_DIR if is_bowler else BATSMEN_MODELS_DIR
    model_path = os.path.join(model_dir, f"{player_name}.pt")
    if not os.path.exists(model_path):
        return None
    
    try:
        model = BallPredictor()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model for {player_name}: {e}", file=sys.stderr)
        return None

def predict_player_performance(model, num_balls=6):
    """Use the model to predict a player's performance over num_balls."""
    if model is None:
        return None
    
    # Start with a random sequence of 6 balls
    sequence = torch.tensor([random.randint(0, 6) for _ in range(6)], dtype=torch.long).unsqueeze(0)
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_balls):
            output = model(sequence)
            pred = torch.argmax(output, dim=1).item()
            predictions.append(int(idx_to_label[pred]))
            # Update sequence for next prediction
            sequence = torch.cat([sequence[:, 1:], torch.tensor([[pred]], dtype=torch.long)], dim=1)
    
    # Calculate predicted stats
    wickets = predictions.count(-1)
    runs = sum(r for r in predictions if r != -1)
    economy = (sum(r for r in predictions if r != -1) / len(predictions)) * 6 if predictions else 0
    average = runs / wickets if wickets > 0 else float('inf')
    
    return {
        "predicted_wickets": wickets,
        "predicted_runs": runs,
        "predicted_economy": economy,
        "predicted_average": average
    }

def get_player_predictions(player):
    """Get predictions for a player if they have a model available."""
    is_bowler = (player.get("wickets", 0) >= 50)
    model = load_player_model(player["name"], is_bowler=is_bowler)
    if model:
        predictions = predict_player_performance(model)
        if predictions:
            return predictions
    return None

def update_players_with_predictions(players):
    """Update a list of players with their model predictions if available."""
    for player in players:
        predictions = get_player_predictions(player)
        if predictions:
            player.update(predictions)
    return players

def format_logits(logits, top_k=3):
    """Format logits into a readable string showing probabilities for each possible outcome.
    Args:
        logits: Raw model output logits
        top_k: Number of top probabilities to show
    """
    # Convert logits to probabilities using softmax
    probs = torch.nn.functional.softmax(logits, dim=-1)
    probs = probs.squeeze().tolist()
    
    # Create list of (outcome, probability) pairs
    outcomes = []
    for idx, prob in enumerate(probs):
        outcome = idx_to_label[idx]
        if outcome == '-1':
            outcome = 'W'  # Wicket
        outcomes.append((outcome, prob))
    
    # Sort by probability
    outcomes.sort(key=lambda x: x[1], reverse=True)
    
    # Format the top k outcomes
    result = []
    for outcome, prob in outcomes[:top_k]:
        result.append(f"{outcome}: {prob:.2%}")
    
    return " | ".join(result)

def predict_next_over(bowler_name, num_overs=4):
    """Predict the next over for a bowler based on a continuous sequence of num_overs from their data.
    If there aren't enough overs, pads with zeros to make up the required number.
    Returns a list of 6 ball outcomes (runs or wicket) along with their logits."""
    # First load the bowler's data
    bowler_data_path = os.path.join(BOWLERS_DATA_DIR, f"{bowler_name}.pt")
    if not os.path.exists(bowler_data_path):
        print(f"No data found for {bowler_name}", file=sys.stderr)
        return None
    
    try:
        # Load the bowler's sequence data
        data = torch.load(bowler_data_path)
        if data.ndim != 2 or data.shape[1] != 6:
            print(f"Invalid data shape for {bowler_name}: {data.shape}", file=sys.stderr)
            return None
        
        # Create a padded sequence if we don't have enough overs
        if len(data) < num_overs:
            print(f"Warning: Only {len(data)} overs available for {bowler_name}, padding with zeros", file=sys.stderr)
            # Create a zero over (6 dot balls)
            zero_over = torch.zeros((1, 6), dtype=data.dtype)
            # Pad the data with zero overs
            padding = torch.cat([zero_over] * (num_overs - len(data)))
            data = torch.cat([padding, data])
        
        # Get a random continuous sequence of num_overs
        max_start = max(0, len(data) - num_overs)
        start_idx = random.randint(0, max_start)
        
        # Get the sequence of num_overs
        sequence = data[start_idx:start_idx + num_overs]
        
        # Get the actual next over if available, otherwise None
        actual_next_over = None
        if start_idx + num_overs < len(data):
            actual_next_over = data[start_idx + num_overs].tolist()
        
        # Load the bowler's model
        model = load_player_model(bowler_name, is_bowler=True)
        if model is None:
            return None
        
        # Store the original sequence for display
        input_sequence = sequence.tolist()
        
        # Convert sequence to model input format (flatten the overs)
        sequence = sequence.flatten()  # This gives us 24 balls (4 overs * 6 balls)
        sequence = torch.tensor([label_to_idx[str(x.item())] for x in sequence], dtype=torch.long)
        
        # Predict the next over
        predictions = []
        all_logits = []  # Store logits for each ball
        
        with torch.no_grad():
            # Use the last 6 balls of the sequence to predict each ball
            for i in range(6):  # 6 balls in an over
                input_seq = sequence[-6:].unsqueeze(0)  # Take last 6 balls as input
                output = model(input_seq)
                logits = output  # Store raw logits
                pred = torch.argmax(output, dim=1).item()
                pred_value = int(idx_to_label[pred])
                predictions.append(pred_value)
                all_logits.append(logits.squeeze().tolist())  # Store logits for this ball
                # Update sequence for next prediction
                sequence = torch.cat([sequence, torch.tensor([pred], dtype=torch.long)])
        
        result = {
            "input_sequence": input_sequence,  # The 4 overs we used (in original format)
            "predicted_over": predictions,  # Our model's prediction
            "logits": all_logits  # Raw logits for each ball
        }
        
        # Add actual next over if available
        if actual_next_over is not None:
            result["actual_next_over"] = actual_next_over
        
        return result
    
    except Exception as e:
        print(f"Error predicting next over for {bowler_name}: {e}", file=sys.stderr)
        return None

def format_over(over):
    """Format an over's predictions into a readable string."""
    if not over:
        return "No prediction available"
    
    # Convert numbers to cricket notation
    def ball_to_str(ball):
        if ball == -1:
            return "W"  # Wicket
        return str(ball)
    
    balls = [ball_to_str(ball) for ball in over]
    runs = sum(ball for ball in over if ball != -1)
    wickets = over.count(-1)
    
    return f"{' '.join(balls)} | Runs: {runs} | Wickets: {wickets}"

def format_sequence(sequence):
    """Format a sequence of overs into a readable string."""
    if not sequence:
        return "No sequence available"
    
    result = []
    for over in sequence:
        result.append(format_over(over))
    return "\n".join(result)