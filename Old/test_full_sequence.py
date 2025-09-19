# test_full_sequence.py
import torch
from tetris_env import TetrisEnvironment
from train import TetrisNet

print("Testing full sequence...")
env = TetrisEnvironment()
net = TetrisNet()

print("Reset...")
state, _ = env.reset(seed=123)

print("Convert to tensor...")
state_tensor = torch.FloatTensor(state).unsqueeze(0)

print("Network forward...")
policy_logits, value = net(state_tensor)
policy = torch.softmax(policy_logits, dim=-1)

print("Sample action...")
action_dist = torch.distributions.Categorical(policy)
action = action_dist.sample()

print(f"Taking action {action.item()}...")
next_state, reward, done, _, _ = env.step(action.item())

print(f"Success! Reward: {reward}, Done: {done}")
env.close()