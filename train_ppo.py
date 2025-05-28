import torch
import torch.optim as optim
import numpy as np
from auction_env import AuctionEnv, PPOAgent

# Hyperparameters
NUM_AGENTS = 82  # Ensure this matches the number of teams in teams.json
NUM_EPISODES = 1000
GAMMA = 0.99
CLIP = 0.2
LR = 1e-4
EPOCHS = 5

# Environment and Agent
env = AuctionEnv(player_folder='players/', teams_file='teams.json')
obs_dim = 2  # runs, wickets
action_dim = 100  # bid values from 0 to 99

agent = PPOAgent(input_dim=obs_dim, action_dim=action_dim)
optimizer = optim.Adam(agent.parameters(), lr=LR)

# Track number of non-league players bought by each agent
non_league_count = {i: 0 for i in range(NUM_AGENTS)}

# Rollout Buffer
class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.rewards, self.values, self.log_probs, self.dones = [], [], [], [], [], []

    def clear(self):
        self.states, self.actions, self.rewards, self.values, self.log_probs, self.dones = [], [], [], [], [], []

buffer = RolloutBuffer()

# Training Loop
for episode in range(NUM_EPISODES):
    obs = env.reset()
    obs = torch.FloatTensor(obs)

    done = False
    episode_reward = 0

    while not done:
        logits, value = agent(obs)
        dist = torch.distributions.Categorical(logits)
        actions = dist.sample([NUM_AGENTS])

        actions_np = actions.numpy()

        # Adjust bids: pass if agent has already bought 4 non-league players
        for i in range(NUM_AGENTS):
            if non_league_count[i] >= 4:
                actions_np[i] = 0  # Pass (no bid)

        actions = torch.tensor(actions_np)
        log_probs = dist.log_prob(actions)

        bids = actions.numpy().flatten()
        next_obs, rewards, done, _ = env.step(bids)

        # Check if the bought player is a non-league player
        for i, reward in enumerate(rewards):
            # Assuming a positive reward means player bought
            if reward > 0:
                if env.is_non_league(i):
                    non_league_count[i] += 1

        next_obs = torch.FloatTensor(next_obs)

        # Store rollout per step
        buffer.states.append(obs)
        buffer.actions.append(actions)
        buffer.rewards.append(torch.tensor(rewards, dtype=torch.float))
        buffer.values.append(value.squeeze().detach())
        buffer.log_probs.append(log_probs)
        buffer.dones.append(done)

        obs = next_obs
        episode_reward += rewards[0]

    # Convert lists to tensors
    states = torch.stack(buffer.states)
    actions = torch.stack(buffer.actions).T
    rewards = torch.stack(buffer.rewards).T
    values = torch.stack(buffer.values)
    log_probs_old = torch.stack(buffer.log_probs).T
    dones = torch.tensor(buffer.dones, dtype=torch.float)

    # Calculate advantages
    advantages = []
    returns = []
    next_value = 0
    for t in reversed(range(len(rewards))):
        mask = 1 - dones[t]
        next_value = rewards[t] + GAMMA * next_value * mask
        returns.insert(0, next_value)
        advantage = next_value - values[t]
        advantages.insert(0, advantage)
    advantages = torch.stack(advantages).detach()
    returns = torch.stack(returns).detach()

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # PPO Update
    for _ in range(EPOCHS):
        logits, value_preds = agent(states)
        dist = torch.distributions.Categorical(logits)
        log_probs = dist.log_prob(actions)

        ratios = torch.exp(log_probs - log_probs_old.detach())
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - CLIP, 1 + CLIP) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = ((returns - value_preds.squeeze())**2).mean()

        loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    buffer.clear()
    print(f"Episode {episode + 1} Agent 0 Reward: {episode_reward}")
    env.print_teams()

# Save the trained PPO agent
torch.save(agent.state_dict(), 'trained_ppo_agent.pth')
print("Trained PPO agent saved as 'trained_ppo_agent.pth'")