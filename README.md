84 teams, 6,972 matches.
an auction with 84 trained DDPG agents, 2421 players
11 players in each team, 4 bowlers 1 all rounder (there will be a bowling score. if there are not 4 bowlers, the best bowlers out of the batsmen and all rounders)
, 4 overseas(depending on whether they don't belong to that region's team)
all the deliveries ever bowled will be sequenced for each player.
all the balls ever played will be sequenced for each player
a random batch of 4 overs will be picked randomly, different seed every time
same for the batsmen, all these categorical values will be converted into a one-hot encoding 
these 2 tensors will be processed upon and a new delivery will be predicted
then cricket will be played 
one match each for the home side, one away, the home advantage exists
6,972 iterations of for loops, 6,972 match reports
who's gonna be crowned the t20 champion of the world?
