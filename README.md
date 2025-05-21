# T20 Simulation Overview

**84 teams, 6,972 matches**  
An auction with 84 trained DDPG agents, 2,421 players

## Team Composition

- 11 players in each team:
  - 4 bowlers  
  - 1 all rounder *(there will be a bowling score. if there are not 4 bowlers, the best bowlers out of the batsmen and all rounders)*
  - 4 overseas *(depending on whether they don't belong to that region's team)*

## Data Preparation

- All the deliveries ever bowled will be sequenced for each player  
- All the balls ever played will be sequenced for each player  
- A random batch of 4 overs will be picked randomly, different seed every time  
- Same for the batsmen  
- All these categorical values will be converted into a one-hot encoding  

## Model and Simulation

- These 2 tensors will be processed upon and a new delivery will be predicted  
- Then cricket will be played  
- One match each for the home side, one away *(the home advantage exists)*  
- 6,972 iterations of `for` loops, 6,972 match reports

---

### Who's gonna be crowned the T20 champion of the world?
