import torch
import torch.nn as nn
import json
import pickle
import os


# Same model definition as used during training
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

# Load model
bowler_name = 'JJ Bumrah'
model_path = f'bowler_models/{bowler_name}.pt'

model = BallPredictor()
model.load_state_dict(torch.load(model_path))
model.eval()