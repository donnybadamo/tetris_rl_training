# Tetris RL Training

A reinforcement learning agent trained to play Tetris using PPO with GA-tuned reward shaping.

## Problem Statement

Training RL agents on Tetris is challenging due to:
- **Sparse rewards**: Line clears happen infrequently, providing little learning signal
- **Long-term dependencies**: Good Tetris requires setting up future line clears, not just immediate ones
- **Complex state space**: 2^200 possible board configurations with 7 piece types and 4 rotations each

## Solution Approach

This implementation uses **dense reward shaping** based on classical Tetris heuristics, with weights derived from a successful genetic algorithm model.

### Architecture

- **Environment**: Custom Gym environment wrapping a pure Python Tetris engine
- **Action Space**: Direct piece placement (rotation, x-position) rather than movement commands
- **Observation**: Flattened 20x10 board state (200 values)
- **Algorithm**: PPO with tuned hyperparameters for dense rewards

### Reward Function

The reward combines six factors with weights proven effective by genetic algorithm optimization:

```python
reward = 9.9 * lines_cleared          # Prioritize line clearing
       + 0.12 * height_reduction      # Slight bonus for lowering stack
       - 9.25 * holes_created         # Heavy penalty for holes
       - 1.43 * bumpiness_increase    # Moderate penalty for jagged surface
       + 0.78 * well_depth            # Small bonus for strategic wells
       - 0.54 * max_height            # Penalty for tall stacks
```

**Key insight**: The line reward (+9.9) almost exactly balances the hole penalty (-9.25), forcing the agent to consider whether a line clear is worth creating holes.

## Files

- `engine.py` - Core Tetris game logic and state management
- `tetris_env.py` - Gym environment wrapper with action decoding
- `train.py` - PPO training script with optimized hyperparameters
- `pieces.py` - Tetris piece definitions and rotations
- `sim.py` - Physics simulation (collision detection, line clearing)

## Usage

```bash
# Install dependencies
pip install stable-baselines3[extra] gymnasium numpy

# Train the model
python train.py

# Monitor training
tensorboard --logdir ./logs
```

## Training Results

The dense reward shaping enables faster convergence compared to sparse line-clear-only rewards:

- **Sparse rewards**: Agent plateaus at ~30 moves, 0 lines cleared
- **Dense rewards**: Agent learns board management and line clearing strategies

## Key Learnings

1. **Curriculum learning transfer fails** - Skills learned on simplified boards don't transfer to normal Tetris
2. **Sparse rewards are insufficient** - Pure line-clear rewards lead to survival-only strategies  
3. **Classical heuristics provide scaffolding** - Dense feedback accelerates learning of strategic play
4. **Action space design matters** - Direct placement actions simplify the learning problem vs movement commands

## Hyperparameters

Tuned for dense reward signals:

```python
PPO(
    learning_rate=1e-4,     # Lower LR for dense rewards
    n_steps=1024,           # Smaller rollouts
    batch_size=32,          # Smaller batches  
    gamma=0.95,             # Less long-term focus
    clip_range=0.1,         # Tighter clipping
    ent_coef=0.01,          # Encourage exploration
)
```

## Future Work

- **State representation**: Add next piece information to observation
- **Action masking**: Prevent invalid actions at the environment level
- **Curriculum**: Progressive difficulty (board height, speed)
- **Multi-objective**: Balance line clearing with survival time
