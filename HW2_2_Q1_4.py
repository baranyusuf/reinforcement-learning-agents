# plot_convergence_curves.py

import os
import glob
import json
import matplotlib.pyplot as plt

# where your JSONs live
INPUT_DIR = r"C:\EE449\EE449_HW_2\results_td0"
# where to put the new convergence plots
OUTPUT_DIR = "results_convergence_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# how many early episodes to skip in the zoomed plot
SKIP_EPISODES = 10

for json_file in glob.glob(os.path.join(INPUT_DIR, "*.json")):
    name = os.path.splitext(os.path.basename(json_file))[0]
    with open(json_file, "r") as f:
        data = json.load(f)
    diffs = data["convergence"]
    episodes = list(range(1, len(diffs) + 1))

    # optionally drop the first SKIP_EPISODES points
    ep_plot = episodes[SKIP_EPISODES:]
    diffs_plot = diffs[SKIP_EPISODES:]

    plt.figure(figsize=(6, 4))
    plt.plot(ep_plot, diffs_plot, lw=1)
    plt.title(f"Convergence (skip {SKIP_EPISODES} eps) — {name}")
    plt.xlabel("Episode")
    plt.ylabel("∑ₙ |Uₙ − Uₙ₋₁|")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{name}_convergence.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")
