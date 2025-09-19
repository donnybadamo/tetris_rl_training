# test_network.py
import torch
from train import TetrisNet
import numpy as np

print("Testing neural network...")
net = TetrisNet()

# Create fake state like your environment would
fake_state = np.zeros((2, 20, 10), dtype=np.int32)
state_tensor = torch.FloatTensor(fake_state).unsqueeze(0)

print("Running forward pass...")
policy_logits, value = net(state_tensor)
print("Forward pass worked!")

print("Testing action sampling...")
policy = torch.softmax(policy_logits, dim=-1)
action_dist = torch.distributions.Categorical(policy)
action = action_dist.sample()
log_prob = action_dist.log_prob(action)

print(f"Sampled action: {action.item()}")
print("Network test complete!")