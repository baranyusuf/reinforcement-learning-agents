import gymnasium as gym
import numpy as np
import torch
import os
import json
import torch.nn.functional as F
from utils import plot_learning_curves, plot_solved_episodes
import torch.nn as nn
from collections import deque
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")


# 1) Q-Network with two hidden layers
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(128,128)):
        super().__init__()
        dims = [state_dim] + list(hidden_dims) + [action_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)
        )
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


# 2) Replay memory (as given)
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# 3) DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim,
                 memory_size=50000, batch_size=64,
                 hidden_dims=(128, 128),  # ← New parameter
                 gamma=0.99, alpha=1e-3,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 target_update_freq=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.memory = ReplayMemory(memory_size)

        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.target_update_freq = target_update_freq

        self.solved_score = 200.0
        self.solved_window = 100
        self.rewards_window = deque(maxlen=self.solved_window)

    def get_action(self, state):
        # ε-greedy action selection
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            qvals = self.policy_net(state_t)
        return int(qvals.argmax(dim=1).item())

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states = torch.tensor(np.array([exp[0] for exp in batch]), dtype=torch.float32)
        actions = torch.tensor([exp[1] for exp in batch], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor([exp[2] for exp in batch], dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array([exp[3] for exp in batch]), dtype=torch.float32)
        dones = torch.tensor([exp[4] for exp in batch], dtype=torch.float32).unsqueeze(1)

        # Current Q-values
        curr_Q = self.policy_net(states).gather(1, actions)
        # Next Q-values from target network
        next_Q = self.target_net(next_states).max(dim=1)[0].unsqueeze(1)
        # TD target: r + γ·maxQ′·(1−done)
        target_Q = rewards + (self.gamma * next_Q * (1 - dones))

        # MSE loss and backprop
        loss = F.mse_loss(curr_Q, target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        # Copy weights
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        # Decay with floor
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)



# ─── 4) Experiment Runner ──────────────────────────────────────────────────────
def run_experiment(params, num_episodes=5000, save_dir="results_dqn"):
    os.makedirs(save_dir, exist_ok=True)
    # unpack hyperparameters
    α  = params["alpha"]
    γ  = params["gamma"]
    εd = params["epsilon_decay"]
    f  = params["target_update_freq"]
    h  = params["hidden_dims"]

    # make a fresh env & agent
    env = gym.make("LunarLander-v3")
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=s_dim, action_dim=a_dim,
        hidden_dims=h,
        alpha=α, gamma=γ,
        epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=εd,
        target_update_freq=f
    )

    episode_rewards = []
    average_scores = []
    solved_episode = None

    for ep in range(1, num_episodes+1):
        state, _ = env.reset()
        max_steps_per_episode = 1000
        total_reward = 0.0
        done = False

        for t in range(max_steps_per_episode):
            # 1) pick an action
            action = agent.get_action(state)

            # 2) step the env
            next_state, reward, done, _, _ = env.step(action)

            # 3) store the transition
            agent.memory.push(state, action, reward, next_state, float(done))

            # 4) update the network
            agent.train_step()

            # 5) move to the next state
            state = next_state

            # 6) accumulate reward
            total_reward += reward

            # 7) if the episode is over, stop early
            if done:
                break

        # end episode bookkeeping
        episode_rewards.append(total_reward)
        agent.decay_epsilon()
        if ep % agent.target_update_freq == 0:
            agent.update_target()

        # moving average over the last `solved_window`
        recent_rewards = episode_rewards[-agent.solved_window:]
        moving_avg = np.mean(recent_rewards)
        average_scores.append(moving_avg)
        if solved_episode is None and len(recent_rewards) == agent.solved_window and moving_avg >= agent.solved_score:
            solved_episode = ep

    # close environment after all episodes
    env.close()

    # save results to JSON
    results = {
        "episode_rewards": episode_rewards,
        "average_scores": average_scores,
        "hyperparameters": params,
        "solved_episode": solved_episode
    }
    # filename uses descriptive parameter keys
    filename = (
            f"{save_dir}/alpha{params['alpha']}_gamma{params['gamma']}_"
            f"epsdecay{params['epsilon_decay']}_freq{params['target_update_freq']}_"
            f"net" + "x".join(str(d) for d in params['hidden_dims']) + ".json"
    )
    with open(filename, 'w') as json_file:
        json.dump(results, json_file)
    return filename

# 1) Defaults (the bold values in your table)
defaults = {
    "alpha":           1e-3,
    "gamma":           0.99,
    "epsilon_decay":   0.995,
    "target_update_freq": 10,
    "hidden_dims":     (128, 128),
}

# 2) Sweep definitions
sweeps = {
    "alpha":           [1e-4, 1e-3, 5e-3],
    "gamma":           [0.98,  0.99, 0.999],
    "epsilon_decay":   [0.98,  0.99, 0.995],
    "target_update_freq": [1, 10, 50],
    # for the “Net*” experiments, replace these tuples with your actual layer sizes:
    "hidden_dims": [
        (128,),
        (64, 64),
        (128, 128),
        (128, 128, 128),
        (256, 256),
    ],
}

all_jsons = []

for param, values in sweeps.items():
    json_paths = []
    labels     = []
    for v in values:
        # build this run's hyperparameter dict
        params = defaults.copy()
        params[param] = v

        print(f">>> Running sweep {param} = {v}")
        fname = run_experiment(params,
                               num_episodes=5000,
                               save_dir="results_dqn")
        json_paths.append(fname)
        all_jsons.append(fname)

        # pretty label
        if isinstance(v, tuple):
            labels.append("x".join(map(str, v)))
        else:
            labels.append(str(v))

    # 3) Plot learning curves for this group
    out_curve = f"results_dqn/{param}_learning_curves.png"
    plot_learning_curves(json_paths,
                         labels=labels,
                         output_file=out_curve)

# 4) One bar chart of solved episodes across *all* experiments
out_solved = "results_dqn/solved_episodes.png"
plot_solved_episodes(all_jsons,
                     labels=None,
                     output_file=out_solved)

print("\nDone.  Look in results_dqn/ for JSONs and plots.")


