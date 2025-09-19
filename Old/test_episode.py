# test_episode_only.py
from tetris_env import TetrisEnvironment
from train import TetrisTrainer

print("Testing episode collection only...")
trainer = TetrisTrainer()

print("Calling collect_episode...")
experience = trainer.collect_episode()

print(f"Episode complete: {experience['steps']} steps, reward {experience['total_reward']}")