import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from sim import PlaneSim
from dqn_agent import DQNAgent
from matplotlib.lines import Line2D

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

def visualize_dqn_policy(model_path='policy_dqn.pth', n_planes=5, seed=5):
    """
    Visualize the trained DQN policy in the PlaneSim environment.
    Works with the dqn_agent.py and dqn_train.py
    """
    # Set all seeds for reproducibility
    set_all_seeds(seed)

    # Initialize environment
    env = PlaneSim(n_planes=n_planes)
    
    # Create DQN agent (exploration disabled)
    agent = DQNAgent(state_size=6, action_size=5) # match number of states/actions
    agent.epsilon = 0.0 # no exploration during visualization
    agent.load(model_path) # load trained weights (policy data)
    
    # Define turn increments (same as training)
    turn_increments = {
        1: -np.pi/12, # -15deg (left)
        2: np.pi/12,  # +15deg (right)
        3: -np.pi/6,  # -30deg (sharper left)
        4: np.pi/6    # +30deg (sharper right)
    }
    
    # Set up interactive plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    env.reset()

    # Initialize rewards and action names (for showing on plot)
    total_reward = 0
    step_rewards = []
    action_names = {0: "Straight", 1: "Right15", 2: "Left15", 3: "Right30", 4: "Left30"}

    # Store flight paths for each plane
    flight_paths = [[] for _ in range(n_planes)] # list to store positions for each plane (eventually forming a path)
    plane_colors = plt.cm.Set3(np.linspace(0, 1, n_planes)) # different colors for each plane
    
    try:
        # Main visualization loop
        for step in range(env.max_steps):
            # Initialization
            actions = []
            step_reward = 0
            
            # Get actions for each plane from DQN
            for plane_idx in range(env.n_planes):
                if env.planes[plane_idx].active:
                    state = env.get_dqn_state(plane_idx)
                    action = agent.act(state) # DQN selects action
                    actions.append(action)
                    
                    # Apply action to plane
                    if action != 0:
                        env.planes[plane_idx].heading += turn_increments[action]
                        env.planes[plane_idx].heading = (env.planes[plane_idx].heading + np.pi) % (2 * np.pi) - np.pi
                else:
                    actions.append(0) # inactive plane
            
            # Step the environment
            env.step()
            
            # Update flight paths for each plane
            for plane_idx, plane in enumerate(env.planes):
                if plane.active:
                    # Store current position in flight path
                    flight_paths[plane_idx].append(plane.position.copy())
            
            # Loop over the planes to get rewards
            for plane_idx in range(env.n_planes):
                if env.planes[plane_idx].active:
                    reward = env.get_reward(plane_idx)
                    step_reward += reward
                    total_reward += reward
            
            # Save rewards of current step for live history plotting
            step_rewards.append(step_reward)

            # Check if terminated
            terminated = env.done or (step == env.max_steps - 1)
            
            # Render environment
            ax.clear()

            # Render the current state
            env.render(ax=ax)
            
            # Draw flight paths for each plane
            for plane_idx, path in enumerate(flight_paths):
                if len(path) > 1: # need at least 2 points to draw a line
                    path_array = np.array(path)
                    ax.plot(path_array[:, 0], path_array[:, 1], 
                           color=plane_colors[plane_idx], 
                           linewidth=2, 
                           alpha=0.7,
                           label=f'Plane {plane_idx} path')
            
            # Display metrics on the plot
            metrics_text = (
                f"Step: {step}/{env.max_steps}\n"
                f"Total reward: {total_reward:.2f}\n"
                f"Current reward: {step_reward:.2f}\n"
                f"Collision: {env.collision}\n"
            )
            
            # Add info for each plane
            for i, plane in enumerate(env.planes):
                status = "Active" if plane.active else "Landed/Crashed"
                path_length = len(flight_paths[i])
                metrics_text += (
                    f"\nPlane {i}:\n"
                    f"  Action: {action_names.get(actions[i])}\n"
                    f"  Heading: {np.degrees(plane.heading):.1f}°\n"
                    f"  Position: ({plane.position[0]:.1f}, {plane.position[1]:.1f})\n"
                    f"  Path length: {path_length} steps\n"
                    f"  Status: {status}"
                )
            
            # Set title of the plot
            ax.set_title("Flight Control - DQN Policy Visualization")

            # Add legend for flight paths
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1))

            # Add text box to the right including the metrics
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(1.05, 0.95, metrics_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top', bbox=props)
            
            # Adjust plot layout to make room for the text
            plt.subplots_adjust(right=0.7)
            
            # Plot reward history (live updated)
            ax2 = ax.inset_axes([0.7, 0.7, 0.25, 0.25])
            ax2.plot(step_rewards, 'b-')
            ax2.set_title("Reward History")
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Reward")
            
            fig.canvas.draw()
            plt.pause(0.05)

            # Check if the figure is closed
            if not plt.fignum_exists(fig.number) or terminated:
                break
                
    except KeyboardInterrupt:
        print("Visualization stopped")

    # Draw final summary plot
    finally:
        plt.ioff() # interactive mode off
        
        # Create final summary plot with complete flight paths
        _, ax_final = plt.subplots(figsize=(6, 6))

        env.render(ax=ax_final)
        
        # Draw complete flight paths
        for plane_idx, path in enumerate(flight_paths):
            if len(path) > 1: # need at least 2 points to draw a line
                path_array = np.array(path)
                ax_final.plot(path_array[:, 0], path_array[:, 1], 
                            color=plane_colors[plane_idx], 
                            linewidth=3, 
                            alpha=0.8,
                            label=f'Plane {plane_idx} path')
                # Mark start position with a circle
                ax_final.scatter(path_array[0, 0], path_array[0, 1], 
                               color=plane_colors[plane_idx], 
                               s=100, marker='o', edgecolors='black', 
                               label=f'Plane {plane_idx} start')
                # Mark end position with a cross
                ax_final.scatter(path_array[-1, 0], path_array[-1, 1], 
                               color=plane_colors[plane_idx], 
                               s=100, marker='X', edgecolors='black',
                               label=f'Plane {plane_idx} end')
                
        # Add simpler legend only with start/end markers and plane colors (otherwise legend gets really large in case of many planes)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Start'), # path start marker
            Line2D([0], [0], marker='X', color='w', markerfacecolor='gray', markersize=10, label='End') # path end marker
        ]
        # Add plane color indicators
        for plane_idx in range(n_planes):
            legend_elements.append(Line2D([0], [0], color=plane_colors[plane_idx], linewidth=3, label=f'Plane {plane_idx}'))
        
        # Draw environment elements
        ax_final.set_title(f"DQN Strategy After 200 Episodes - Total Reward: {total_reward:.1f}")
        ax_final.legend(handles=legend_elements, loc='upper right')
        ax_final2 = ax_final.inset_axes([0.7, 0.1, 0.25, 0.25]) # define size and location of reward history plot
        ax_final2.plot(step_rewards, 'b-')
        ax_final2.set_title("Reward History")
        ax_final2.set_xlabel("Step")
        ax_final2.set_ylabel("Reward")
        plt.tight_layout()
        plt.show()

        # Print final results
        print(f"Final total reward: {total_reward:.2f}")
        print(f"Collision occurred: {env.collision}")
        landed_count = sum(1 for p in env.planes if not p.active and not env.collision)
        print(f"Landed: {landed_count}/{n_planes}")

if __name__ == "__main__":
    visualize_dqn_policy(model_path='policy_dqn_test.pth', n_planes=5, seed=5) # make sure n_planes and seed is same as what you trained with