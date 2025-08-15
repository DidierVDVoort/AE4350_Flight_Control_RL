import torch
import numpy as np
import matplotlib.pyplot as plt
from sim import PlaneSim
from dqn_agent import DQNAgent

def visualize_dqn_policy(model_path='policy_dqn.pth', n_planes=1):
    """
    Visualize the trained DQN policy in the PlaneSim environment.
    Works with the dqn_agent.py and dqn_train.py
    """
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
    action_names = {0: "Straight", 1: "Left15", 2: "Right15", 3: "Left30", 4: "Right30"}
    
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
            
            # Loop over the planes to get rewards
            for plane_idx in range(env.n_planes):
                if env.planes[plane_idx].active:
                    reward = env.get_simple_reward(plane_idx)
                    step_reward += reward
                    total_reward += reward
            
            # Save rewards of current step for live history plotting
            step_rewards.append(step_reward)

            # Check if terminated
            terminated = env.done or (step == env.max_steps - 1)
            
            # Render environment
            ax.clear()
            env.render(ax=ax)
            
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
                metrics_text += (
                    f"\nPlane {i}:\n"
                    f"  Action: {action_names.get(actions[i])}\n"
                    f"  Heading: {np.degrees(plane.heading):.1f}°\n"
                    f"  Position: ({plane.position[0]:.1f}, {plane.position[1]:.1f})\n"
                    f"  Status: {status}"
                )
            
            # Set title of the plot
            ax.set_title("Flight Control - DQN Policy Visualization")

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
    
    finally:
        plt.ioff() # interactive mode off
        plt.show()

        # Print final results
        print(f"Final total reward: {total_reward:.2f}")
        print(f"Collision occurred: {env.collision}")
        print(f"Landed: {sum(1 for p in env.planes if not p.active and not env.collision)}/{n_planes}")

if __name__ == "__main__":
    visualize_dqn_policy(model_path='policy_dqn_test.pth', n_planes=3)