import numpy as np
import torch
import matplotlib.pyplot as plt
from sim import PlaneSim
from agent import DQNAgent

# Function to visualise the agent AFTER training in the PlaneSim environment
def visualise(model_path='flight_control_agent_relative.pth'):
    # Initialize environment and agent
    env = PlaneSim(n_planes=2)
    agent = DQNAgent(env.state_size, env.action_space, env.n_planes)
    
    # Load trained weights
    agent.model.load_state_dict(torch.load(model_path))
    agent.epsilon = 0.0 # disable exploration, so full exploitation
    
    # Define turn increments
    turn_increments = {
        1: -np.pi/12,   # -15deg (left)
        2: np.pi/12,    # +15deg (right)
        3: -np.pi/6,    # -30deg (sharper left)
        4: np.pi/6      # +30deg (sharper right)
    }
    
    # Setup interactive plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    env.reset()
    
    # Initialize rewards
    total_reward = 0
    step_rewards = [] # store rewards for plotting
    
    try:
        for step in range(env.max_steps):
            # Get state and actions (one per plane)
            state = env.get_state()
            actions = agent.act(state)

            # Apply heading changes for each plane
            for i, plane in enumerate(env.planes):
                action = actions[i]
                if action != 0:
                    increment = turn_increments.get(action, 0)
                    plane.heading += increment
                    plane.heading = (plane.heading + np.pi) % (2*np.pi) - np.pi
            
            # Step the environment
            env.step()
            reward, breakdown = env.get_reward()

            # Additional penalty: penalize turning slightly (action 0 is preferred)
            if action != 0:
                reward -= 0.02  # You can tune this value (e.g., 0.005 or 0.02)
            
            total_reward += reward
            step_rewards.append(reward)
            
            # Render with live info on rewards
            ax.clear()
            env.render(ax=ax)
            
            # Display live parameters
            action_names = {0:"No-op", 1:"Left15", 2:"Right15", 3:"Left30", 4:"Right30"}

            metrics_text = f"Step: {step}\nTotal Reward: {total_reward:.2f}\n"
            for i, plane in enumerate(env.planes):
                heading_deg = np.degrees(plane.heading)
                action_str = action_names.get(actions[i], '?')
                metrics_text += f"Plane {i}: {action_str}, Heading: {heading_deg:.1f}deg\n"
            ax.set_title(metrics_text)
            
            # Plot reward history in corner, being updated live
            ax2 = ax.inset_axes([0.7, 0.7, 0.25, 0.25]) # [x, y, width, height]
            ax2.plot(step_rewards, 'b-')
            ax2.set_title("Reward History")
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Reward")
            
            fig.canvas.draw()
            plt.pause(0.05)
            
            if not plt.fignum_exists(fig.number): # Check if the figure is closed
                break
    
    # Handle keyboard interrupt properly
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()

if __name__ == "__main__":
    visualise()