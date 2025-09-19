import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tetris_env import TetrisEnvironment
import matplotlib.pyplot as plt

class TetrisNet(nn.Module):
    def __init__(self, action_size=40):
        super().__init__()
        # Input: (2, 20, 10) -> flatten to 400
        self.flatten = nn.Flatten()
        
        # Simple feedforward network
        self.network = nn.Sequential(
            nn.Linear(400, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Policy head (action probabilities)
        self.policy_head = nn.Linear(64, action_size)
        
        # Value head (state value estimation)
        self.value_head = nn.Linear(64, 1)
        
    def forward(self, x):
        # x shape: (batch, 2, 20, 10)
        x = self.flatten(x)
        features = self.network(x)
        
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return policy_logits, value

class TetrisTrainer:
    def __init__(self):
        print("2a. Creating environment...")
        self.env = TetrisEnvironment()
        print("2b. Creating neural network...")
        self.net = TetrisNet()
        print("2c. Creating optimizer...")
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        print("2d. Initializing metrics...")
        
        # Training metrics
        self.scores = []
        self.episode_lengths = []
        print("2e. Trainer initialization complete")
        
    def collect_episode(self):
        """Play one episode and collect experience"""
        print("3a. Starting episode collection...")
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        
        print("3b. Resetting environment...")
        state, _ = self.env.reset()
        print("3c. Environment reset complete")
        done = False
        total_reward = 0
        steps = 0
        
        print("3d. Starting game loop...")
        while not done and steps < 1000:  # Max 1000 steps per episode
            print(f"3e. Step {steps}: Converting state to tensor...")
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            print(f"3f. Step {steps}: Getting network prediction...")
            # Get policy and value from network
            policy_logits, value = self.net(state_tensor)
            policy = torch.softmax(policy_logits, dim=-1)
            
            print(f"3g. Step {steps}: Sampling action...")
            # Sample action
            action_dist = torch.distributions.Categorical(policy)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            print(f"3h. Step {steps}: Taking action {action.item()}...")
            # Take action in environment
            next_state, reward, done, _, _ = self.env.step(action.item())
            
            print(f"3i. Step {steps}: Action complete, reward={reward}, done={done}")
            
            # Store experience
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if steps >= 5:  # Stop debug after 5 steps to avoid spam
                break
        
        print("3j. Closing environment...")
        # self.env.close()
        print("3k. Episode collection complete")
        
        return {
            'states': np.array(states),
            'actions': actions,
            'rewards': rewards,
            'values': values,
            'log_probs': torch.stack(log_probs),
            'total_reward': total_reward,
            'steps': steps
        }
    
    def calculate_returns(self, rewards, values, gamma=0.99):
        """Calculate discounted returns for PPO"""
        returns = []
        advantages = []
        
        # Add bootstrap value for last state (0 if terminal)
        values_with_bootstrap = values + [0]
        
        # Calculate returns and advantages
        running_return = 0
        for i in reversed(range(len(rewards))):
            running_return = rewards[i] + gamma * running_return
            advantage = running_return - values[i]
            
            returns.insert(0, running_return)
            advantages.insert(0, advantage)
            
        return returns, advantages
    
    def update_network(self, experience):
        """Update network using PPO"""
        states = torch.FloatTensor(experience['states'])
        actions = torch.LongTensor(experience['actions'])
        old_log_probs = experience['log_probs']
        
        returns, advantages = self.calculate_returns(
            experience['rewards'], 
            experience['values']
        )
        
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(4):  # 4 epochs per episode
            # Get current policy and values
            policy_logits, values = self.net(states)
            policy = torch.softmax(policy_logits, dim=-1)
            
            # Calculate new log probabilities
            action_dist = torch.distributions.Categorical(policy)
            new_log_probs = action_dist.log_prob(actions)
            
            # PPO ratio
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            
            # Policy loss (simplified PPO)
            policy_loss = -torch.min(
                ratio * advantages,
                torch.clamp(ratio, 0.8, 1.2) * advantages
            ).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
    
    def train(self, num_episodes=1000):
        """Main training loop"""
        print("Starting Tetris training...")
        
        for episode in range(num_episodes):
            # Collect episode
            experience = self.collect_episode()
            
            # Update network
            # self.update_network(experience)
            
            # Track metrics
            self.scores.append(experience['total_reward'])
            self.episode_lengths.append(experience['steps'])
            
            # Print progress
            if episode % 50 == 0:
                avg_score = np.mean(self.scores[-50:]) if len(self.scores) >= 50 else np.mean(self.scores)
                avg_length = np.mean(self.episode_lengths[-50:]) if len(self.episode_lengths) >= 50 else np.mean(self.episode_lengths)
                
                print(f"Episode {episode}: Avg Score: {avg_score:.1f}, Avg Length: {avg_length:.1f}")
                
                # Save model
                if episode > 0:
                    torch.save(self.net.state_dict(), f'tetris_model_ep{episode}.pt')
        
        self.env.close()
        self.plot_training_progress()
    
    def plot_training_progress(self):
        """Plot training metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot scores
        ax1.plot(self.scores)
        ax1.set_title('Training Scores')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        
        # Plot episode lengths
        ax2.plot(self.episode_lengths)
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.show()

if __name__ == "__main__":
    print("1. Creating trainer...")
    trainer = TetrisTrainer()
    print("2. Trainer created, starting training...")
    trainer.train(num_episodes=200)  # Start with 200 episodes