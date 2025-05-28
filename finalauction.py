import torch
import numpy as np
from auction_env import AuctionEnv, PPOAgent

# Load the trained agent
trained_agent = PPOAgent(input_dim=2, action_dim=100)
trained_agent.load_state_dict(torch.load('trained_ppo_agent.pth'))
trained_agent.eval()

# Create 83 clones (1 + 82)
NUM_AGENTS = 83
agents = [trained_agent for _ in range(NUM_AGENTS)]

# Set up the auction environment
env = AuctionEnv(player_folder='players/', num_agents=NUM_AGENTS)

# Run a single auction
obs = env.reset()
done = False

while not done:
    obs_tensor = torch.FloatTensor(obs)
    logits, _ = trained_agent(obs_tensor)
    dist = torch.distributions.Categorical(logits)
    actions = dist.sample([NUM_AGENTS])
    bids = actions.numpy().flatten()

    # Let agents with 11 players stop bidding
    for i in range(NUM_AGENTS):
        if len(env.teams[i]) >= 11:
            bids[i] = 0

    obs, rewards, done, _ = env.step(bids)

# Print final teams and rewards
env.print_teams()
env.print_total_rewards()

# Rank agents by total rewards
print("\n=== Agent Rankings by Total Reward ===")
sorted_agents = sorted(enumerate(env.total_rewards), key=lambda x: x[1], reverse=True)
for rank, (agent_id, reward) in enumerate(sorted_agents, 1):
    print(f"{rank}. Agent {agent_id}: {reward:.2f}")