# ball_predictor.py

import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Original overs data
overs = torch.tensor([
    [0, 4, 1, 0, 0, 4],
    [0, 0, 0, 1, 0, 1],
    [0, 2, 0, 1, 1, 0],
    [1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 4, 0],
    [1, 1, 1, 0, 0, 1],
    [1, 4, 2, 4, 1, 0],
    [0, 1, 1, 4, 4, 1],
    [0, 0, 0, 0, 1, 1],
    [1, 0, 1, 1, 0, 1],
    [4, 1, 0, 6, 0, 1]
], dtype=torch.long)

# Step 2: Flatten all overs into one sequence of balls
all_balls = overs.flatten()

# Step 3: Prepare input (X) and target (y) sequences
sequence_length = 6
X = []
y = []

for i in range(len(all_balls) - sequence_length):
    X.append(all_balls[i:i + sequence_length])
    y.append(all_balls[i + sequence_length])

X = torch.stack(X)  # shape: [num_samples, 6]
y = torch.tensor(y, dtype=torch.long)  # shape: [num_samples]

# Step 4: Define the model
class BallPredictor(nn.Module):
    def __init__(self, vocab_size=7, embed_dim=16, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)               # (batch, 6, embed_dim)
        _, h = self.rnn(x)                  # h: (1, batch, hidden)
        return self.fc(h.squeeze(0))        # (batch, vocab_size)

# Step 5: Initialize and train the model
model = BallPredictor()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Step 6: Predict the next ball given a new 6-ball sequence
# Example input: last 6 balls from training data
test_seq = all_balls[-6:].unsqueeze(0)  # shape: [1, 6]
with torch.no_grad():
    pred_logits = model(test_seq)
    pred_run = torch.argmax(pred_logits, dim=1).item()
    print(f"\nGiven sequence: {test_seq.tolist()[0]}")
    print(f"Predicted next ball run: {pred_run}")