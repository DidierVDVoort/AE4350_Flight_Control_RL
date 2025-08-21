import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# COMPARISON OF MOVING AVERAGES OF REWARDS AND EPSILON DECAY
# Load results (with seed 5) from both algorithms
dqn_results = torch.load("dqn_results/dqn_seed_5.pth")
qlearning_results = torch.load("qlearning_results/qlearning_seed_5.pth")

# Convert tensors back to numpy arrays for plotting (tensors were needed for security reasons of numpy)
dqn_rewards = dqn_results['episode_rewards'].numpy()
dqn_epsilons = dqn_results['episode_epsilons'].numpy()

qlearning_rewards = qlearning_results['episode_rewards'].numpy()
qlearning_epsilons = qlearning_results['episode_epsilons'].numpy()

# Create comparison plots
plt.figure(figsize=(6, 5))

# First plot: smoothed rewards (moving average)
window_size = 50

# Smooth the rewards using a moving average
dqn_smoothed = np.convolve(dqn_rewards, np.ones(window_size)/window_size, mode='valid')
qlearning_smoothed = np.convolve(qlearning_rewards, np.ones(window_size)/window_size, mode='valid')

# Create arrays with zeros for the first 50 episodes
dqn_smoothed_full = np.concatenate([np.zeros(window_size), dqn_smoothed])
qlearning_smoothed_full = np.concatenate([np.zeros(window_size), qlearning_smoothed])

plt.subplot(2, 1, 1)
plt.plot(dqn_smoothed_full, label='DQN')
plt.plot(qlearning_smoothed_full, label='Q-Learning')
plt.title(f'Smoothed Rewards (Window={window_size}) Comparison')
plt.xlabel('Episode')
plt.ylabel('Smoothed Reward')
plt.legend()
plt.grid(True)

# Second plot: epsilon decay comparison
plt.subplot(2, 1, 2)
plt.plot(dqn_epsilons, label='DQN')
plt.plot(qlearning_epsilons, label='Q-Learning')
plt.title('Epsilon Decay Comparison')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ANALYSIS OF TRAINING RESULTS FOR BOTH ALGORITHMS
def analyze_results(results_folder):
    """Analyze multiple runs with different seeds"""
    
    # Get all result files
    result_files = glob.glob(os.path.join(results_folder, "*.pth"))
    
    # Define metrics
    metrics = {
        'success_rates': [],
        'collision_rates': [],
        'steps_per_episode': [],
        'average_rewards': [],
        'training_times': []
    }
    
    # Save averages of results from each result file (i.e. each run)
    for file_path in result_files:
        results = torch.load(file_path)
            
        # Extract metrics from the last 50 episodes (last episodes only, since those have most stable performance)
        n_last_episodes = 50
            
        if n_last_episodes > 0:
            # Success rate (average of last episodes): landed planes / total planes
            success_rate = np.mean(results['episode_stats']['success_rate'][-n_last_episodes:])
            metrics['success_rates'].append(success_rate)
                
            # Collision rate (average of last episodes): number of collisions occurred
            collision_rate = np.mean(results['episode_stats']['collisions'][-n_last_episodes:])
            metrics['collision_rates'].append(collision_rate)
                
            # Steps per episode (average of last episodes)
            steps = np.mean(results['episode_stats']['steps_per_episode'][-n_last_episodes:])
            metrics['steps_per_episode'].append(steps)
                
            # Average reward (average of last episodes)
            avg_reward = np.mean(results['episode_rewards'].numpy()[-n_last_episodes:])
            metrics['average_rewards'].append(avg_reward)

            # Total training time (all episodes)
            training_time = results['training_time']
            metrics['training_times'].append(training_time)
    
    return metrics

# Analyze results
dqn_metrics = analyze_results("dqn_results")
qlearning_metrics = analyze_results("qlearning_results")

# Print DQN statistics summary
print(f"DQN performance summary ({len(dqn_metrics['success_rates'])} runs/seeds):")
print("=" * 50)
print(f"Success rate: {np.mean(dqn_metrics['success_rates']):.3f} +/- {np.std(dqn_metrics['success_rates']):.3f}")
print(f"Collision rate: {np.mean(dqn_metrics['collision_rates']):.3f} +/- {np.std(dqn_metrics['collision_rates']):.3f}")
print(f"Steps per episode: {np.mean(dqn_metrics['steps_per_episode']):.1f} +/- {np.std(dqn_metrics['steps_per_episode']):.1f}")
print(f"Average reward: {np.mean(dqn_metrics['average_rewards']):.1f} +/- {np.std(dqn_metrics['average_rewards']):.1f}")
print(f"Total training time: {np.mean(dqn_metrics['training_times']):.1f} +/- {np.std(dqn_metrics['training_times']):.1f}")

# Print Q-Learning statistics summary
print(f"\nQ-Learning performance summary ({len(qlearning_metrics['success_rates'])} runs/seeds):")
print("=" * 50)
print(f"Success rate: {np.mean(qlearning_metrics['success_rates']):.3f} +/- {np.std(qlearning_metrics['success_rates']):.3f}")
print(f"Collision rate: {np.mean(qlearning_metrics['collision_rates']):.3f} +/- {np.std(qlearning_metrics['collision_rates']):.3f}")
print(f"Steps per episode: {np.mean(qlearning_metrics['steps_per_episode']):.1f} +/- {np.std(qlearning_metrics['steps_per_episode']):.1f}")
print(f"Average reward: {np.mean(qlearning_metrics['average_rewards']):.1f} +/- {np.std(qlearning_metrics['average_rewards']):.1f}")
print(f"Total training time: {np.mean(qlearning_metrics['training_times']):.1f} +/- {np.std(qlearning_metrics['training_times']):.1f}")