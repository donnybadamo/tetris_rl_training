# Quick test script
from tetris_env import TetrisEnvironment

print("Testing environment directly...")
env = TetrisEnvironment()

print("Resetting...")
state, _ = env.reset(seed=123)
print(f"Reset worked, state shape: {state.shape}")

print("Taking one action...")
state, reward, done, _, _ = env.step(13)
print(f"Action worked: reward={reward}, done={done}")

env.close()
print("Test complete")