# test_two_episodes.py  
from train import TetrisTrainer

print("Testing 2 episodes...")
trainer = TetrisTrainer()

print("Episode 1...")
exp1 = trainer.collect_episode()
print(f"Episode 1 done: {exp1['steps']} steps")

print("Episode 2...")
exp2 = trainer.collect_episode()  
print(f"Episode 2 done: {exp2['steps']} steps")