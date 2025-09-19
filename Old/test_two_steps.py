# test_two_steps.py
from tetris_env import TetrisEnvironment
from train import TetrisNet
import torch

print("Testing just 2 steps...")
env = TetrisEnvironment()
net = TetrisNet()

print("Step 1...")
state, _ = env.reset()
state_tensor = torch.FloatTensor(state).unsqueeze(0)
policy_logits, value = net(state_tensor)
policy = torch.softmax(policy_logits, dim=-1)
action_dist = torch.distributions.Categorical(policy)
action = action_dist.sample()

next_state, reward, done, _, _ = env.step(action.item())
print(f"Step 1 complete: action={action.item()}, reward={reward}, done={done}")

if not done:
    print("Step 2...")
    state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
    policy_logits, value = net(state_tensor)
    policy = torch.softmax(policy_logits, dim=-1)
    action_dist = torch.distributions.Categorical(policy)
    action = action_dist.sample()
    
    next_state, reward, done, _, _ = env.step(action.item())
    print(f"Step 2 complete: action={action.item()}, reward={reward}, done={done}")

env.close()