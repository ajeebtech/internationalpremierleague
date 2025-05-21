import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
RG_tensor = torch.load("batsmen/RG Sharma.pt")
# JJ_tensor = torch.load("bowlers/JJ Bumrah.pt")
# # print(torch.matmul(RG_tensor, torch.transpose(JJ_tensor,0,1)))
# product = torch.matmul(RG_tensor, torch.transpose(JJ_tensor,0,1))
# print(product)
# # probabilities = F.softmax(product, dim=1)
print(RG_tensor)