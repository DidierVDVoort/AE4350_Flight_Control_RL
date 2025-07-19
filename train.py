from agent import DQNAgent
from sim import PlaneSim
import matplotlib.pyplot as plt
import numpy as np
import torch

# Function to train the DQN agent in the PlaneSim environment
def train():
    # Initialize environment and agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Try to use GPU if available
    env = PlaneSim(n_planes=2)
    agent = DQNAgent(env.state_size, env.action_space, env.n_planes)

    # Part of the hyperparameters
    batch_size = 32
    epochs = 50

    # Possible actions for relative heading changes (0 = straight ahead, 1-4 = left/right turns)
    turn_increments = {  # Define turn angles in radians
        1: -np.pi/12,    # -15deg (left)
        2: np.pi/12,     # +15deg (right)
        3: -np.pi/6,     # -30deg (sharper left)
        4: np.pi/6       # +30deg (sharper right)
    }
    
    rewards_history = []

    # Training loop
    for e in range(epochs):
        # Get initial state
        state = env.get_state()
        state = torch.FloatTensor(state).to(device)
        total_reward = 0
        env.reset()
        
        # Loop through simulation steps
        plt.ion()
        for step in range(env.max_steps):
            # Get actions (for each plane)
            actions = agent.act(state)
            
            # Execute relative heading change (action) for each plane
            for i, action in enumerate(actions):
                if action != 0 and env.planes[i].active:
                    env.planes[i].heading += turn_increments[action]
                    # Normalize heading to [-pi, pi]
                    env.planes[i].heading = (env.planes[i].heading + np.pi) % (2 * np.pi) - np.pi
            
            # Step simulation
            env.step()
            # env.render() # Uncomment to visualize during training (becomes very slow though)

            # Get next state after stepping
            next_state = env.get_state()
            next_state = torch.FloatTensor(next_state).to(device)

            # Get reward and breakdown (for logging)
            reward, breakdown = env.get_reward()

            # Extra reward: penalize turning slightly (action 0 is preferred)
            for i, action in enumerate(actions):
                if action != 0:
                    reward -= 0.02
            
            # print(f"Reward: {reward:.2f} (Land: {breakdown['landing']:.2f} | "
            #     f"Col: {breakdown['collision']:.2f} | Edge: {breakdown['edge']:.2f} | "
            #     f"Prog: {breakdown['progress']:.2f})")
            done = step == env.max_steps - 1 # done if max steps reached

            # Store experience in memory
            agent.remember(state, actions, reward, next_state, done)

            # Update state for next iteration
            state = next_state
            total_reward += reward
            
            if done:
                print(f"Episode: {e+1}/{epochs}, Reward: {total_reward}, Epsilon: {agent.epsilon}")
                rewards_history.append(total_reward)
                break
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size) # train the agent with a batch of experiences
        plt.ioff()

    plt.close() # close plot from render
    torch.save(agent.model.state_dict(), 'flight_control_agent_relative.pth')
    plt.plot(rewards_history)
    plt.show()

if __name__ == "__main__":
    train()
