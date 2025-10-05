import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from  matplotlib.colors import LinearSegmentedColormap

def plot_value_function(recorded, maze, name,params, save_dir='results_td0'):
    """
    Plots value functions for multiple episodes in a 3x2 grid and saves the figure.

    recorded: dict of {episode: value_function array}
    maze: maze layout array
    name: base name for saving file
    save_dir: directory to save figure
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Create a 3x2 figure grid
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))

    param_str = ", ".join(f"{k}={v}" for k, v in params.items())

    # Iterate over axes and recorded episodes
    for ax, (ep, U) in zip(axes.flatten(), recorded.items()):
        # Mask obstacles, traps, and goal cells
        mask = np.zeros_like(U, dtype=bool)
        mask[maze == 1] = True
        mask[maze == 2] = True
        mask[maze == 3] = True

        # Plot heatmap on this axis
        cmap = LinearSegmentedColormap.from_list('rg', ['r', 'w', 'g'], N=256)
        sns.heatmap(U, mask=mask, annot=True, fmt=".1f", cmap=cmap,
                    cbar=False, linewidths=1, linecolor='black', ax=ax)

        # Highlight trap and obstacle cells
        for trap in np.argwhere(maze == 2):
            ax.add_patch(plt.Rectangle(trap[::-1], 1, 1,
                                       fill=True, edgecolor='black', facecolor='darkred'))
        for obs in np.argwhere(maze == 1):
            ax.add_patch(plt.Rectangle(obs[::-1], 1, 1,
                                       fill=True, edgecolor='black', facecolor='gray'))

        ax.set_title(f"Episode {ep}")
        ax.axis('off')

    # Overall title and save
    fig.suptitle(f"Value Function Milestones ({param_str})", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(save_dir, f'{name}_value_milestones.png'))
    plt.close(fig)


def plot_policy(recorded, maze, name, params, save_dir='results_td0'):
    """
    Plots policy maps for multiple episodes in a 3x2 grid and saves the figure.
    """

    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))

    policy_arrows = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
    actions = ['up', 'down', 'left', 'right']
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())

    for ax, (ep, U) in zip(axes.flatten(), recorded.items()):
        # Build policy grid of arrows
        policy_grid = np.full(maze.shape, '', dtype='<U2')
        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                # skip obstacles (1), traps (2), goal (3)
                if maze[i, j] in (1, 2, 3):
                    continue
                best_action, best_val = None, -np.inf
                for a in actions:
                    ni = i + (a == 'down') - (a == 'up')
                    nj = j + (a == 'right') - (a == 'left')
                    # only consider valid moves
                    if (0 <= ni < maze.shape[0] and
                        0 <= nj < maze.shape[1] and
                        maze[ni, nj] != 1):
                        if U[ni, nj] > best_val:
                            best_val, best_action = U[ni, nj], a
                # *** assign the arrow into the grid ***
                if best_action is not None:
                    policy_grid[i, j] = policy_arrows[best_action]

        # Mask out non-policy cells
        mask = np.zeros_like(U, dtype=bool)
        mask[maze == 1] = mask[maze == 2] = True

        cmap = LinearSegmentedColormap.from_list('rg', ['r', 'w', 'g'], N=256)
        sns.heatmap(
            U,
            mask=mask,
            annot=policy_grid,
            fmt="",
            cmap=cmap,
            cbar=False,
            linewidths=1,
            linecolor='black',
            annot_kws={'fontsize':16, 'color':'black'},
            ax=ax
        )

        # Highlight traps and obstacles
        for trap in np.argwhere(maze == 2):
            ax.add_patch(plt.Rectangle(trap[::-1], 1, 1,
                                       fill=True, edgecolor='black', facecolor='darkred'))
        for obs in np.argwhere(maze == 1):
            ax.add_patch(plt.Rectangle(obs[::-1], 1, 1,
                                       fill=True, edgecolor='black', facecolor='gray'))

        for goal in np.argwhere(maze == 3):
            ax.add_patch(plt.Rectangle(goal[::-1], 1, 1,
                                       fill=True, edgecolor='black', facecolor='lime'))

        ax.set_title(f"Episode {ep}")
        ax.axis('off')

    fig.suptitle(f"Policy Milestones ({param_str})", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(save_dir, f'{name}_policy_milestones.png'))
    plt.close(fig)

import json
import matplotlib.pyplot as plt

def plot_learning_curves(json_paths, labels=None,  output_file="learning_curves.png"):
    """
    Plots both episode_rewards and average_scores from multiple experiment JSON files as subplots.
    
    Parameters:
        json_paths (list of str): List of file paths to JSON result files.
        labels (list of str, optional): Labels for each experiment (for legend).
                                        If not provided, file names (without .json) are used.
        output_file (str): Filename for saving the output image (e.g., "learning_curves.png").
        
    The function creates two subplots:
      - The first subplot shows raw episode_rewards vs. Episode number.
      - The second subplot shows average_scores (e.g., a 100-episode moving average) vs. Episode number.
    The figure is saved to the specified output_file and also displayed.
    """
    # Load data from JSON files
    data_list = []
    for path in json_paths:
        with open(path, 'r') as f:
            data_list.append(json.load(f))
            
    # Generate default labels from file names if none are provided
    if labels is None:
        labels = [path.split('/')[-1].replace('.json', '') for path in json_paths]
    
    # Create figure with two subplots (vertical layout)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Subplot 1: Raw episode rewards
    axs[0].set_title("Episode Rewards")
    for idx, data in enumerate(data_list):
        if "episode_rewards" not in data:
            raise KeyError(f"Key 'episode_rewards' not found in {json_paths[idx]}.")
        rewards = data["episode_rewards"]
        episodes = range(1, len(rewards) + 1)
        axs[0].plot(episodes, rewards, label=labels[idx])
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")
    axs[0].legend()
    axs[0].grid(True)
    
    # Subplot 2: Moving average scores
    axs[1].set_title("Average Scores (Moving Average)")
    for idx, data in enumerate(data_list):
        if "average_scores" not in data:
            raise KeyError(f"Key 'average_scores' not found in {json_paths[idx]}.")
        avg_scores = data["average_scores"]
        episodes = range(1, len(avg_scores) + 1)
        axs[1].plot(episodes, avg_scores, label=labels[idx])
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Average Score")
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)  # Save the figure to an image file
    plt.show()

def plot_solved_episodes(json_paths, labels=None, output_file="solved_episodes.png"):
    """
    Plot a bar chart showing the episode number at which each experiment solved the task.
    
    Parameters:
        json_paths (list of str): List of JSON result file paths (each corresponding to an experiment).
        labels (list of str, optional): Names for each experiment (for x-axis labels). If not provided, file names will be used.
        output_file (str): Filename for saving the output image (e.g., "solved_episodes.png").
    
    Each JSON file is expected to have a 'solved_episode' entry (the episode index when the solve criterion was first met).
    Experiments that did not solve within the training duration are omitted from the chart.
    """
    solved_eps = []
    exp_names = []
    for i, path in enumerate(json_paths):
        with open(path, 'r') as f:
            data = json.load(f)
        episode = data.get("solved_episode", None)
        if episode is not None and episode is not False:
            solved_eps.append(episode)
            # Determine label for this bar
            if labels and i < len(labels):
                exp_names.append(labels[i])
            else:
                exp_names.append(path.split('/')[-1].replace('.json',''))
        # If not solved (episode is None or False/ -1), skip this experiment for the bar chart
    if not solved_eps:
        print("No experiments solved the environment to plot.")
        return
    plt.figure(figsize=(8,5))
    colors = plt.get_cmap('tab20').colors  # get a list of colors
    bars = plt.bar(exp_names, solved_eps, color=[colors[i % len(colors)] for i in range(len(solved_eps))])
    plt.ylabel('Episode Solved')
    plt.title('Solved Episode by Experiment')
    plt.xticks(rotation=45, ha='right')
    # Annotate each bar with the episode number
    for rect, ep in zip(bars, solved_eps):
        plt.text(rect.get_x() + rect.get_width()/2, ep + 2, str(ep), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(output_file)  # Save the figure to an image file
    plt.show()