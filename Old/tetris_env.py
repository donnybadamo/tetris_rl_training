import subprocess, sys
import json, select, time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TetrisEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Action space: 40 possible placements (4 rotations × 10 positions)
        self.action_space = spaces.Discrete(40)
        
        # State space: 20x10 board (current) + 20x10 board (with piece projection) = 800 values
        self.observation_space = spaces.Box(low=0, high=7, shape=(2, 20, 10), dtype=np.int32)
        
        # Start Node.js process
        self.process = None
        self._start_process()
        
    def _start_process(self):
        """Start the Node.js Tetris wrapper"""
        if self.process:
            self.process.terminate()
        self.process = subprocess.Popen(
            ['node', 'wrapper.js'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # simplest: merge stderr into stdout
            text=True,
            encoding='utf-8',
            bufsize=1,                  # line-buffered text I/O
            close_fds=True
        )
    
    def _send_command(self, command, timeout=5.0):
        print(f"DEBUG: Sending {command}")
        if self.process.poll() is not None:
            print(f"Process already dead: {self.process.poll()}")
            raise RuntimeError(f"Node died: rc={self.process.poll()}")

        line = json.dumps(command) + '\n'
        print(f"DEBUG: Writing line: {line.strip()}")
        self.process.stdin.write(line)
        self.process.stdin.flush()

        end = time.time() + timeout
        while time.time() < end:
            r, _, _ = select.select([self.process.stdout], [], [], 0.1)
            if r:
                raw = self.process.stdout.readline()
                if not raw:
                    raise RuntimeError("EOF from Node (process exited?)")

                try:
                    resp = json.loads(raw)
                except Exception as e:
                    raise RuntimeError(f"Bad JSON from Node: {raw!r}, err={e}")

                # If Node signals an error, surface it
                if 'error' in resp:
                    raise RuntimeError(f"Node error: {resp}")

                return resp

        raise TimeoutError(f"No response in {timeout}s for command={command}")
    
    def _decode_action(self, action):
        """Convert action number (0-39) to rotation and x position"""
        action = int(action)  # Convert numpy int64 to regular int
        rotation = action // 10  # 0-3
        x = action % 10          # 0-9
        return rotation, x
    
    def _encode_state(self, game_state):
        """Convert game state to neural network input"""
        # Piece ID to number mapping
        piece_map = {'I': 1, 'O': 2, 'T': 3, 'S': 4, 'Z': 5, 'J': 6, 'L': 7}
        
        board = np.array(game_state['board'], dtype=np.int32)
        
        # Create second board with current piece projection
        projected_board = board.copy()
        
        # Add current piece to projected board (if exists)
        if game_state['currentPiece'] and not game_state['done']:
            piece = game_state['currentPiece']
            shape = np.array(piece['shape'])
            x, y = piece['x'], piece['y']
            piece_id_num = piece_map.get(piece['id'], 1)  # Convert 'S' -> 4
            
            # Project piece onto board
            for py in range(shape.shape[0]):
                for px in range(shape.shape[1]):
                    if shape[py, px] == 1:
                        board_y = y + py
                        board_x = x + px
                        if 0 <= board_y < 20 and 0 <= board_x < 10:
                            projected_board[board_y, board_x] = piece_id_num
        
        # Stack the two boards: [current_board, projected_board]
        return np.stack([board, projected_board])
    
    def _calculate_reward(self, old_state, new_state):
        """Calculate reward for the action taken"""
        if new_state['done']:
            return -100  # Game over penalty
        
        lines_cleared = new_state['lines'] - old_state['lines']
        return lines_cleared * 10  # +10 per line cleared
    
    def reset(self, seed=None):
        """Reset the environment"""
        if seed is None:
            seed = np.random.randint(0, 1000000)
        
        # Restart process if it died
        if self.process is None or self.process.poll() is not None:
            print("Process died, restarting...")
            self._start_process()
        
        response = self._send_command({'action': 'reset', 'seed': seed})
        state = self._encode_state(response)
        
        self.last_game_state = response
        return state, {}
    
    def step(self, action):
        """Take an action in the environment"""
        rotation, x = self._decode_action(action)
        
        old_state = self.last_game_state
        response = self._send_command({
            'action': 'step', 
            'placement': {'rotation': rotation, 'x': x}
        })
        
        reward = self._calculate_reward(old_state, response)
        state = self._encode_state(response)
        done = response['done']
        
        self.last_game_state = response
        return state, reward, done, False, {}
    
    def close(self):
        """Clean up the Node.js process"""
        if self.process:
            self.process.terminate()
            self.process = None

# Test the environment
if __name__ == "__main__":
    env = TetrisEnvironment()
    
    print("Testing environment...")
    state, info = env.reset(seed=123)
    print(f"Initial state shape: {state.shape}")
    print(f"Board sum: {state[0].sum()} (should be 0 - empty board)")
    
    # Take a random action
    action = env.action_space.sample()
    state, reward, done, truncated, info = env.step(action)
    print(f"After action {action}: reward={reward}, done={done}")
    print(f"New board sum: {state[0].sum()} (should be > 0 - piece placed)")
    
    env.close()