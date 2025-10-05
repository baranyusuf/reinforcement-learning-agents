import numpy as np
from HW_2_Q1_1 import *

class MazeTD0(MazeEnvironment):
    def __init__(self, maze, alpha=0.1, gamma=0.95, epsilon=0.2, episodes=10000):
        super().__init__()
        self.maze = maze
        self.alpha = alpha        # learning rate
        self.gamma = gamma        # discount factor
        self.epsilon = epsilon    # exploration rate
        self.episodes = episodes
        max_return = 0
        self.utility = np.full(self.maze.shape, max_return, dtype=float)
        self.utility[self.maze == 1] = -np.inf

    def choose_action(self, state):
        """ε-greedy over neighboring utilities, bouncing invalid moves."""
        r, c = state
        # exploration
        if np.random.random() < self.epsilon:
            return np.random.choice(list(self.actions.keys()))
        # exploitation: pick action that leads to highest utility
        best_value = -np.inf
        best_action = None
        rows, cols = self.maze.shape
        for a, (dr, dc) in self.actions.items():
            nr, nc = r + dr, c + dc
            # if off-grid or obstacle, treat as staying put
            if not (0 <= nr < rows and 0 <= nc < cols) or self.maze[nr, nc] == 1:
                val = -np.inf
            else:
                val = self.utility[nr, nc]
            if best_action is None or val > best_value:
                best_value = val
                best_action = a
        if best_action is None:
            best_action = np.random.choice(list(self.actions.keys()))
        return best_action

    def update_utility_value(self, current_state, reward, new_state):
        """Apply the one-step TD(0) update."""
        cr, cc = current_state
        nr, nc = new_state
        current_value = self.utility[cr, cc]
        next_value    = self.utility[nr, nc]
        # TD target
        target = reward + self.gamma * next_value
        # update rule
        self.utility[cr, cc] += self.alpha * (target - current_value)
        self.utility[self.maze == 2] = self.trap_penalty
        self.utility[self.maze == 3] = self.goal_reward

    def run_episodes(self):
        """Run self.episodes of TD(0), return final utility grid."""
        for _ in range(self.episodes):
            state = self.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                new_state, reward, done = self.step(action)
                self.update_utility_value(state, reward, new_state)
                state = new_state
        return self.utility
# Create an instance of the TD‐0 agent on your maze
maze = layout
maze_td0 = MazeTD0(maze,
                   alpha=0.1,
                   gamma=0.95,
                   epsilon=0.2,
                   episodes=10000)

# Run learning and grab the final utility estimates
final_values = maze_td0.run_episodes()


