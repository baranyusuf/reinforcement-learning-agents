# HW_2_Q1_3.py
import os, json
import numpy as np
import matplotlib.pyplot as plt
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# import your modules from the previous parts
from HW_2_Q1_1 import layout
from HW_2_Q1_2 import MazeTD0
from utils import plot_value_function, plot_policy

# make output directory
os.makedirs("results_td0", exist_ok=True)

# hyperparameter grids
alphas   = [0.001, 0.01, 0.1, 0.5, 1.0]
gammas   = [0.10, 0.25, 0.50, 0.75, 0.95]
epsilons = [0.0, 0.2, 0.5, 0.8, 1.0]

# default (bolded in Table 1)
default = dict(alpha=0.1, gamma=0.95, epsilon=0.2)
milestones = [1, 50, 100, 1000, 5000, 10000]

def run_and_record(name, **params):
    """
    Runs TD(0) with given params, records:
      - utilities at each milestone
      - convergence diffs every episode
      then saves JSON and produces plots.
    """
    print(f"\n--- Experiment: {name} ({params}) ---")
    agent = MazeTD0(layout,
                    alpha=params['alpha'],
                    gamma=params['gamma'],
                    epsilon=params['epsilon'],
                    episodes=10000)
    prev_U = agent.utility.copy()
    diffs = []
    recorded = {}

    for ep in range(1, agent.episodes+1):
        if ep % 1 == 0:
            print(f"  → Completed episode {ep}/{agent.episodes}")
        state = agent.reset()
        done = False
        while not done:
            a = agent.choose_action(state)
            ns, r, done = agent.step(a)
            agent.update_utility_value(state, r, ns)
            state = ns

        # record utility snapshots
        if ep in milestones:
            recorded[ep] = agent.utility.copy()

        # convergence measure
        diff = np.abs(agent.utility - prev_U).sum()
        diffs.append(diff)
        prev_U = agent.utility.copy()

    # save JSON
    out = {
        "hyperparameters": params,
        "utilities_over_time": {str(k): v.tolist() for k, v in recorded.items()},
        "convergence": diffs
    }
    fname = f"results_td0/{name}.json"
    with open(fname, 'w') as f:
        json.dump(out, f)

    save_dir = f"results_td0/{name}_plots"
    os.makedirs(save_dir, exist_ok=True)

    # 2) Plot & save the 3×2 milestone grids in one shot
    print(f"Saving value‐function milestones to {save_dir}/{name}_value_milestones.png")
    plot_value_function(recorded, layout, name, params, save_dir)

    print(f"Saving policy milestones to {save_dir}/{name}_policy_milestones.png")
    plot_policy(recorded, layout, name, params, save_dir)




    # Plot convergence curve
    plt.figure(figsize=(6,4))
    plt.plot(range(1, agent.episodes+1), diffs, label=name)
    plt.title(f"Convergence ‒ {name}")
    plt.xlabel("Episode")
    plt.ylabel("∑ |Uₜ−Uₜ₋₁|")
    plt.legend()
    plt.tight_layout()

    # Save the convergence plot
    plt.savefig(f"results_td0/{name}_convergence.png")
    plt.close()

# 1) Sweep α (others at defaults)
for α in alphas:
    run_and_record(f"alpha_{α}", alpha=α,
                   gamma=default['gamma'],
                   epsilon=default['epsilon'])

# 2) Sweep γ (others at defaults)
for γ in gammas:
    run_and_record(f"gamma_{γ}", alpha=default['alpha'],
                   gamma=γ,
                   epsilon=default['epsilon'])

# 3) Sweep ε (others at defaults)
for ε in epsilons:
    run_and_record(f"epsilon_{ε}", alpha=default['alpha'],
                   gamma=default['gamma'],
                   epsilon=ε)


