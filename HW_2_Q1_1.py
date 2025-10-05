import numpy as np
from matplotlib import pyplot as plt

class MazeEnvironment:
    def __init__(self):
        # Start position
        self.start_pos = (0, 0)
        self.current_pos = self.start_pos

        # Rewards
        self.state_penalty = -1
        self.trap_penalty  = -100
        self.goal_reward   = 100

        # Action deltas: up, down, left, right
        self.actions = {
            0: (-1,  0),   # up
            1: ( 1,  0),   # down
            2: ( 0, -1),   # left
            3: ( 0,  1)    # right
        }

        # For stochastic transitions
        self._opposite = {0:1, 1:0, 2:3, 3:2}
        self._perps    = {0:[2,3], 1:[2,3], 2:[0,1], 3:[0,1]}

        # Placeholder for the maze layout; set in subclass or afterwards
        self.maze = None

    def reset(self):
        """Return the start state and reset the agent."""
        self.current_pos = self.start_pos
        return self.current_pos

    def step(self, action):
        """
        Take the integer action (0–3), sample the actual move according to:
          - 0.75 intended direction
          - 0.05 opposite
          - 0.10 each perpendicular
        Bounce off walls/obstacles (stay in place), then return (state, reward, done).
        """
        if self.maze is None:
            raise ValueError("You must set `self.maze` before calling step().")

        # sample stochastic outcome
        p = np.random.random()
        if p < 0.75:
            a = action
        elif p < 0.80:
            a = self._opposite[action]
        else:
            a = np.random.choice(self._perps[action])

        # compute tentative next position
        dr, dc = self.actions[a]
        r, c = self.current_pos
        nr, nc = r + dr, c + dc

        # check bounds and obstacles
        rows, cols = self.maze.shape
        if not (0 <= nr < rows and 0 <= nc < cols) or self.maze[nr, nc] == 1:
            nr, nc = r, c  # bounce: stay in place

        # determine reward and done
        cell = self.maze[r, c]
        if cell == 3:           # goal
            reward, done = self.goal_reward, True
        elif cell == 2:         # trap
            reward, done = self.trap_penalty, True
        else:                   # normal free cell
            reward, done = self.state_penalty, False

        # update state
        self.current_pos = (nr, nc)
        return self.current_pos, reward, done

# --- now encode Figure 1 layout into a 10×10 array: ---
#    0 = free, 1 = obstacle (grey), 2 = trap (red), 3 = goal (green)

layout = np.array([
    # row 0
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    # row 1
    [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1],
    # row 2
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
    # row 3
    [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
    # row 4
    [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0],
    # row 5
    [1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0],
    # row 6
    [1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
    # row 7
    [0, 0, 1, 1, 1, 1, 2, 0, 1, 1, 0],
    # row 8
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    # row 9
    [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
    # row 10
    [1, 1, 1, 0, 2, 0, 1, 0, 0, 3, 0],
])


