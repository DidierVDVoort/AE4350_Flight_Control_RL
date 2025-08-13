import torch
import numpy as np
import matplotlib.pyplot as plt
from sim import PlaneSim
from q_learning_agent import QLearningAgent
from collections import defaultdict

def visualize_q_policy(model_path='policy.pth', n_planes=1):
    """
    Visualize the trained Q-learning policy in the PlaneSim environment.
    Works with the gym_agent.py and gym_train.py files.
    """
    # Initialize environment
    env = PlaneSim(n_planes=n_planes)
    
    # Create agent with dummy parameters (full exploitation)
    agent = QLearningAgent(
        env=env,
        learning_rate=0.01,
        initial_epsilon=0.0, # full exploitation
        epsilon_decay=0.0,
        final_epsilon=0.0 
    )

    # Loading the policy data (not really secure, but if I trust the source then it is fine)
    loaded_data = torch.load(model_path, weights_only=False)
    if isinstance(loaded_data, dict): # check if loaded data is a dictionary
         agent.q_values = defaultdict(agent.zero_values, loaded_data)
    else: # if loaded data is not a dictionary
        agent.q_values = loaded_data
    
    agent.epsilon = 0.0  # No exploration during visualization
    
    # Define turn increments (same as in training code)
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
            # Get current state
            raw_state = env.get_simple_state()
            state = tuple(env.discretize_state(raw_state))

            # Select action using the Q-table (no exploration)
            action = agent.get_action(state)
            
            # Apply action to all active planes
            for plane in env.planes:
                if plane.active and action != 0:
                    plane.heading += turn_increments[action]
                    plane.heading = (plane.heading + np.pi) % (2 * np.pi) - np.pi # clamp heading to [-pi, pi]

            # Step the environment
            env.step()
            reward = env.get_simple_reward(action)
            total_reward += reward
            step_rewards.append(reward)

            # Check if terminated
            terminated = env.done or (step == env.max_steps - 1)
            
            # Render environment
            ax.clear()
            env.render(ax=ax)
            
            # Display metrics on the plot
            metrics_text = (
                f"Step: {step}/{env.max_steps}\n"
                f"Total reward: {total_reward:.2f}\n"
                f"Current reward: {reward:.2f}\n"
                f"Collision: {env.collision}\n"
                f"Action: {action_names.get(action)}\n"
                f"Q-value: {agent.q_values.get(state, [0]*5)[action]:.2f}\n" # display Q-value for the chosen action ([0,0,0,0,0] if state is new)
            )
            
            # Add info for each plane
            for i, plane in enumerate(env.planes):
                status = "Active" if plane.active else "Landed/Crashed"
                metrics_text += (
                    f"\nPlane {i}:\n"
                    f"  Heading: {np.degrees(plane.heading):.1f}deg\n"
                    f"  Position: ({plane.position[0]:.1f}, {plane.position[1]:.1f})\n"
                    f"  Status: {status}"
                )
            
            # Set metrics text as title of the plot
            ax.set_title(metrics_text, loc='left', fontsize=7, pad=0)
            
            # Plot reward history (live updated)
            ax2 = ax.inset_axes([0.7, 0.7, 0.25, 0.25])
            ax2.plot(step_rewards, 'b-')
            ax2.set_title("Reward History")
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Reward")
            
            fig.canvas.draw()
            plt.pause(0.05)
            
            # Check if figure is closed
            if not plt.fignum_exists(fig.number):
                break
                
            if terminated:
                break
    
    except KeyboardInterrupt:
        print("Visualization stopped")
    
    finally:
        plt.ioff() # interactive mode off
        plt.show()

        # Print final results
        print(f"Final total reward: {total_reward:.2f}")
        print(f"Collision occurred: {env.collision}")
        print(f"Number of planes landed: {sum(1 for p in env.planes if not p.active and not env.collision)}/{n_planes}")

if __name__ == "__main__":
    visualize_q_policy(model_path='policy.pth', n_planes=1)