import torch
import numpy as np
import matplotlib.pyplot as plt
from sim import PlaneSim
from q_learning_agent import QLearningAgent, QLearningMultiAgent
from collections import defaultdict

def visualize_q_policy(model_path='policy.pth', n_planes=1):
    """
    Visualize the trained Q-learning policy in the PlaneSim environment.
    Works with the gym_agent.py and gym_train.py files.
    """
    # Initialize environment
    env = PlaneSim(n_planes=n_planes)

    # Create agents with dummy parameters (full exploitation)
    agents = [QLearningMultiAgent(plane_idx=i, env=env,
        learning_rate=0.01,
        initial_epsilon=0.0,
        epsilon_decay=0.0,
        final_epsilon=0.0
    ) for i in range(env.n_planes)]

    # Load the policy data
    loaded_data = torch.load(model_path, weights_only=False)
    
    if isinstance(loaded_data, dict):
        # Multi-agent policy case: load Q-values of each agent
        for i, agent in enumerate(agents):
            if i in loaded_data: # check if this agent's policy exists
                agent.q_values = defaultdict(agent.zero_values, loaded_data[i])
                agent.epsilon = 0.0 # no exploration during visualization
    else:
        # Single-agent policy case
        if n_planes == 1:
            agents[0].q_values = defaultdict(agents[0].zero_values, loaded_data)
            agents[0].epsilon = 0.0
        else:
            raise ValueError("Loaded policy is not a multi-agent policy but n_planes > 1")
    
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
    total_reward = [0 for _ in range(env.n_planes)]
    step_rewards = [] # store rewards for each step (one entry for each agent)
    action_names = {0: "Straight", 1: "Left15", 2: "Right15", 3: "Left30", 4: "Right30"}
    
    try:
        # Main visualization loop
        for step in range(env.max_steps):
            # Initialization
            obs = []
            actions = []
            step_reward = 0

            # Select action using the Q-table (no exploration)
            for agent in agents:
                raw_state = env.get_simple_state(agent.plane_idx)
                state = tuple(env.discretize_state(raw_state))
                obs.append(state)

                action = agent.get_action(state)
                actions.append(action)
            
                # Apply action to plane
                if env.planes[agent.plane_idx].active and action != 0:
                    env.planes[agent.plane_idx].heading += turn_increments[action]
                    env.planes[agent.plane_idx].heading = (env.planes[agent.plane_idx].heading + np.pi) % (2 * np.pi) - np.pi # clamp heading to [-pi, pi]

            # Step the environment
            env.step()

            # Loop over agents (planes) to get rewards
            for agent in agents:
                reward = env.get_simple_reward(agent.plane_idx)
                step_reward += reward
                total_reward[agent.plane_idx] += reward
            
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
                f"Total reward: {sum(total_reward):.2f}\n"
                f"Current reward: {step_reward:.2f}\n"
                f"Collision: {env.collision}\n"
            )
            
            # Add info for each plane
            for i, plane in enumerate(env.planes):
                status = "Active" if plane.active else "Landed/Crashed"
                metrics_text += (
                    f"\nPlane {i}:\n"
                    f"  Action: {action_names.get(actions[i])}\n"
                    f"  Heading: {np.degrees(plane.heading):.1f}deg\n"
                    f"  Position: ({plane.position[0]:.1f}, {plane.position[1]:.1f})\n"
                    f"  Status: {status}"
                )

                # Display Q-value only for active planes with valid actions (not working properly yet I think)
                if plane.active:
                    q_values = agents[i].q_values.get(obs[i], np.zeros(5)) # display Q-value for the chosen action ([0,0,0,0,0] if state is new)
                    metrics_text += f"\n  Q-value: {float(q_values[actions[i]]):.2f}"

            # Set title of the plot
            ax.set_title("Flight Control Game - Q-Learning Final Policy Visualization")

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
            
            # Check if figure is closed
            if not plt.fignum_exists(fig.number) or terminated:
                break
    
    except KeyboardInterrupt:
        print("Visualization stopped")
    
    finally:
        plt.ioff() # interactive mode off
        plt.show()

        # Print final results
        print(f"Final total reward: {sum(total_reward):.2f}")
        print(f"Collision occurred: {env.collision}")
        print(f"Number of planes landed: {sum(1 for p in env.planes if not p.active and not env.collision)}/{n_planes}")

if __name__ == "__main__":
    visualize_q_policy(model_path='policy_multi_agent_3_planes.pth', n_planes=3)