import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dqn_agent import DQNAgent
from sim import PlaneSim
from tqdm import tqdm
import pandas as pd
import torch
from itertools import count
import random
from collections import defaultdict

# Function to set all seeds for reproducibility
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Ensure deterministic behavior on GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Sensitivity analysis for hyperparameters
def hyperparameter_sensitivity_analysis(n_episodes=200, seeds=[5,10,100]):
    # Fixed environment settings
    n_planes = 5
    state_size = 6
    action_size = 5
    
    # Storage for results
    results = defaultdict(list)
    
    # Test all combinations of parameters
    for param_name, param_values in hyper_param_ranges.items():
        print(f"\n=== Testing {param_name} ===")
        
        for value in param_values:
            print(f"Testing {param_name} = {value}")
            
            # Run multiple times with different seeds
            for seed in seeds:
                set_all_seeds(seed)
                
                # Initialize environment
                env = PlaneSim(n_planes=n_planes)

                # Set default parameters
                params = {
                    'state_size': state_size,
                    'action_size': action_size,
                    'max_replay_memory_size': 10000,
                    'discount_factor': 0.99,
                    'learning_rate': 0.01,
                    'initial_epsilon': 1.0,
                    'final_epsilon': 0.01,
                    'epsilon_decay': 0.95,
                    'batch_size': 64
                }
                
                # Override the current parameter
                params[param_name] = value
                
                # Initialize agent
                agent = DQNAgent(**params)
                
                # Training loop
                episode_stats = [] # store episode statistics
                for episode in tqdm(range(n_episodes)):
                    # Reset environment
                    env.reset()
                    total_reward = 0
                    states = [] # initialize states (one entry for each plane)

                    # Collect initial states for all planes
                    for i in range(env.n_planes):
                        state = env.get_dqn_state(i)
                        states.append(state)
                    
                    # Play one complete simulation (until all planes landed, collided or max. steps is reached)
                    for t in count():
                        actions = [] # reset actions every time (need to be stored for the agent.remember() function)

                        # Agent choose action for each plane (initially random, gradually more intelligent)
                        for i in range(env.n_planes):
                            action = agent.act(states[i])
                            actions.append(action)
                            
                            # Apply the action to the plane
                            if action != 0:
                                env.planes[i].heading += turn_increments[action]
                                env.planes[i].heading = (env.planes[i].heading + np.pi) % (2 * np.pi) - np.pi # clamp heading to [-pi, pi]

                        # Update the environment
                        env.step()
                        terminated = env.done or t == env.max_steps
                        
                        # Loop over all planes and update the agent with the new experiences
                        for i in range(env.n_planes):
                            next_state = env.get_dqn_state(i)
                            reward = env.get_reward(i)

                            # Store experience in replay memory
                            agent.remember(states[i], actions[i], reward, next_state, terminated)

                            # Update state
                            states[i] = next_state
                            total_reward += reward
                        
                        agent.replay() # train the model using the replay buffer
                        
                        if terminated:
                            landed = sum(1 for p in env.planes if not p.active and not env.collision)
                            episode_stats.append({
                                'reward': total_reward,
                                'landed': landed,
                                'collision': 1 if env.collision else 0,
                                'steps': t + 1
                            })
                            break

                    # Decay exploration rate at end of episode (agent becomes less random over time)
                    agent.decay_epsilon()

                    # Update target model every 10 episodes
                    if episode % 10 == 0:
                        agent.update_target_model() # sync target network

                # Calculate metrics from last 50 episodes
                average_reward = np.mean([ep['reward'] for ep in episode_stats[-50:]])
                success_rate = np.mean([ep['landed']/env.n_planes for ep in episode_stats[-50:]])
                collision_rate = np.mean([ep['collision'] for ep in episode_stats[-50:]])
                avg_steps = np.mean([ep['steps'] for ep in episode_stats[-50:]])

                # Store results
                results['parameter'].append(param_name)
                results['value'].append(value)
                results['seed'].append(seed)
                results['average_reward'].append(average_reward)
                results['success_rate'].append(success_rate)
                results['collision_rate'].append(collision_rate)
                results['avg_steps'].append(avg_steps)

    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df

# Plotting function for hyperparameter sensitivity analysis
def plot_hyperparameter_sensitivity(df, hyper_param_ranges):
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot for each parameter
    for i, param_name in enumerate(hyper_param_ranges.keys()):
        # Initialize subplot
        plt.subplot(2, 2, i+1)

        # Get DataFrame for current parameter
        param_df = df[df['parameter'] == param_name]
        
        # Plot mean with error bars showing standard deviation
        sns.lineplot(data=param_df, x='value', y='average_reward', 
                     errorbar='sd', marker='o', label='Average reward')
        
        plt.title(f'Sensitivity to {param_name} with standard deviation')
        plt.xlabel(param_name)
        plt.ylabel('Average reward of last 50 episodes')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Sensitivity analysis for environment settings
def environment_sensitivity_analysis(n_episodes=200, seeds=[5,10,100]):
    # Fixed agent settings
    state_size = 6
    action_size = 5
    agent_params = {
        'state_size': state_size,
        'action_size': action_size,
        'max_replay_memory_size': 10000,
        'discount_factor': 0.99,
        'learning_rate': 0.01,
        'initial_epsilon': 1.0,
        'final_epsilon': 0.01,
        'epsilon_decay': 0.95,
        'batch_size': 64
    }
    
    # Storage for results
    results = defaultdict(list)
    
    # Test all combinations of environment parameters
    for param_name, param_values in env_param_ranges.items():
        print(f"\n=== Testing {param_name} ===")
        
        for value in param_values:
            print(f"Testing {param_name} = {value}")
            
            # Run multiple times with different seeds
            for seed in seeds:
                set_all_seeds(seed)
                
                # Create environment with current parameter
                if param_name == 'n_planes':
                    env = PlaneSim(n_planes=value)
                elif param_name == 'size':
                    env = PlaneSim(n_planes=5)
                    env.size = value
                elif param_name == 'speed':
                    env = PlaneSim(n_planes=5)
                    env.plane_speed = value
                
                # Initialize agent
                agent = DQNAgent(**agent_params)
                
                # Training loop
                episode_stats = []
                for episode in tqdm(range(n_episodes)):
                    # Reset environment
                    env.reset()
                    total_reward = 0
                    states = [] # initialize states (one entry for each plane)
                    
                    # Collect initial states for all planes
                    for i in range(env.n_planes):
                        state = env.get_dqn_state(i)
                        states.append(state)
                    
                    # Play one complete simulation (until all planes landed, collided or max. steps is reached)
                    for t in count():
                        actions = [] # reset actions every time (need to be stored for the agent.remember() function)
                        
                        # Agent choose action for each plane (initially random, gradually more intelligent)
                        for i in range(env.n_planes):
                            action = agent.act(states[i])
                            actions.append(action)
                            
                            # Apply the action to the plane
                            if action != 0:
                                env.planes[i].heading += turn_increments[action]
                                env.planes[i].heading = (env.planes[i].heading + np.pi) % (2 * np.pi) - np.pi
                        
                        # Update the environment
                        env.step()
                        terminated = env.done or t == env.max_steps
                        
                        # Loop over all planes and update the agent with the new experiences
                        for i in range(env.n_planes):
                            next_state = env.get_dqn_state(i)
                            reward = env.get_reward(i)

                            # Store experience in replay memory
                            agent.remember(states[i], actions[i], reward, next_state, terminated)
                            
                            # Update state
                            states[i] = next_state
                            total_reward += reward
                        
                        agent.replay() # train the model using the replay buffer
                        
                        if terminated:
                            landed = sum(1 for p in env.planes if not p.active and not env.collision)
                            episode_stats.append({
                                'reward': total_reward,
                                'landed': landed,
                                'collision': 1 if env.collision else 0,
                                'steps': t + 1
                            })
                            break
                            
                    # Decay exploration rate at end of episode (agent becomes less random over time)
                    agent.decay_epsilon()

                    if episode % 10 == 0:
                        agent.update_target_model()
                
                # Calculate metrics from last 50 episodes
                average_reward = np.mean([ep['reward'] for ep in episode_stats[-50:]])
                success_rate = np.mean([ep['landed']/env.n_planes for ep in episode_stats[-50:]])
                collision_rate = np.mean([ep['collision'] for ep in episode_stats[-50:]])
                avg_steps = np.mean([ep['steps'] for ep in episode_stats[-50:]])

                # Store results
                results['parameter'].append(param_name)
                results['value'].append(value)
                results['seed'].append(seed)
                results['average_reward'].append(average_reward)
                results['success_rate'].append(success_rate)
                results['collision_rate'].append(collision_rate)
                results['avg_steps'].append(avg_steps)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df

# Plotting function for environment sensitivity analysis
def plot_environment_sensitivity(df, env_param_ranges):
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot for each parameter
    metrics = ['average_reward', 'success_rate', 'collision_rate', 'avg_steps']
    metric_names = ['Average reward', 'Success rate', 'Collision rate', 'Steps per episode']

    # Nested loops, looping over parameters and metrics
    # Loop over parameters
    for i, param_name in enumerate(env_param_ranges.keys()):
        param_df = df[df['parameter'] == param_name] # get DataFrame for current parameter

        if param_name == 'n_planes':
            param_df['average_reward'] /= param_df['value'] # divide by number of planes to get per-plane reward
        
        # Loop over metrics
        for j, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            # Initialize subplot
            plt.subplot(len(env_param_ranges), len(metrics), i*len(metrics)+j+1)

            # Plot mean with error bars showing standard deviation
            sns.lineplot(data=param_df, x='value', y=metric, 
                         errorbar='sd', marker='o')
            
            # Change title and y-label only for 1 plot (for per-plane rewards)
            if param_name == 'n_planes' and metric == 'average_reward':
                plt.title(f'{metric_name} (per plane) for varying {param_name}')
                plt.ylabel(f'{metric_name} (per plane)')
            else:
                plt.title(f'{metric_name} for varying {param_name}')
                plt.ylabel(metric_name)
            plt.xlabel(param_name)
            plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define seeds
    seeds = [5, 10, 100]

    # Possible actions for relative heading changes (0 = straight ahead, 1-4 = left/right turns)
    turn_increments = { # Define turn angles in radians
        1: -np.pi/12,   # -15deg (left)
        2: np.pi/12,    # +15deg (right)
        3: -np.pi/6,    # -30deg (sharper left)
        4: np.pi/6      # +30deg (sharper right)
    }

    print("Hyperparameter sensitivity analysis:")

    # Define parameter ranges to test
    hyper_param_ranges = {
        'learning_rate': [0.0001, 0.001, 0.01, 0.1],  # Wider range including very small and large values
        'epsilon_decay': [0.9, 0.95, 0.99],    # From fast decay to very slow decay
        'discount_factor': [0.5, 0.7, 0.99],  # From very myopic to far-sighted
        'batch_size': [32, 64, 128]          # From very small to large batches
    }

    hyper_df = hyperparameter_sensitivity_analysis(n_episodes=200, seeds=[5,10,100])
    hyper_df.to_csv('hyperparameter_sensitivity_results.csv', index=False)
    # hyper_df = pd.read_csv('sensitivity_results/hyperparameter_sensitivity_results.csv') # use this instead if you already have a csv file and only want to plot
    plot_hyperparameter_sensitivity(hyper_df, hyper_param_ranges)

    print("\nEnvironment sensitivity analysis:")

    # Define parameter ranges to test
    env_param_ranges = {
        'n_planes': [3, 5, 10],
        'size': [50, 100, 200],
        'speed': [3.0, 5.0, 10.0]
    }

    env_df = environment_sensitivity_analysis(n_episodes=200, seeds=seeds)
    env_df.to_csv('environment_sensitivity_results.csv', index=False)
    # env_df = pd.read_csv('sensitivity_results/environment_sensitivity_results.csv') # use this instead if you already have a csv file and only want to plot
    plot_environment_sensitivity(env_df, env_param_ranges)

    print("\nSensitivity analysis finished")