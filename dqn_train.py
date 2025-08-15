import torch
from tqdm import tqdm
from sim import PlaneSim
from dqn_agent import DQNAgent
from itertools import count
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Number of states and actions
state_size = 6 # matches get_dqn_state() output
action_size = 5 # [none, left15, right15, left30, right30]

# Training hyperparameters
n_episodes = 200
n_planes = 3
max_replay_memory = 10000
discount_factor = 0.99
learning_rate = 0.001
start_epsilon = 1.0 # start with 100% random actions
final_epsilon = 0.01 # always keep some exploration
epsilon_decay = 0.995
batch_size = 64

# Initialize environment and agent
env = PlaneSim(n_planes=n_planes)
agent = DQNAgent(state_size,
                 action_size,
                 max_replay_memory,
                 discount_factor,
                 learning_rate,
                 start_epsilon,
                 final_epsilon,
                 epsilon_decay,
                 batch_size)

# Set up matplotlib for plotting
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion() # enable interactive mode

# Use GPU if available
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Live reward plotting function
def plot_rewards(show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float) # convert to tensor
    
    # Plot rewards
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    
    # Take 100 episode averages and plot them too (only after 100th episode of course)
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001) # pause so that plots are updated

    # Clear output for next plot
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

episode_rewards = [] # store episode rewards

# Training loop
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
                turn_increment = { # Define turn angles in radians
                    1: -np.pi/12,  # -15deg (left)
                    2: np.pi/12,   # +15deg (right)
                    3: -np.pi/6,   # -30deg (sharper left)
                    4: np.pi/6     # +30deg (sharper right)
                }[action]
                env.planes[i].heading += turn_increment

        # Update the environment
        env.step()
        terminated = env.done or t == env.max_steps

        # Loop over all planes and update the agent with the new experiences
        for i in range(env.n_planes):
            next_state = env.get_dqn_state(i)
            reward = env.get_simple_reward(i)

            # Store experience in replay memory
            agent.remember(states[i], actions[i], reward, next_state, terminated)

            # Update state
            states[i] = next_state
            total_reward += reward

        agent.replay() # train the model using the replay buffer (also decays epsilon)

        if terminated:
            episode_rewards.append(total_reward)
            print(f"Planes landed: {sum(1 for p in env.planes if not p.active and not env.collision)}/{env.n_planes}")
            plot_rewards()
            break
    
    # Update target model every 10 episodes
    if episode % 10 == 0:
        agent.update_target_model()  # Sync target network
        print(f"Episode: {episode}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

# Save the model (policy)
agent.save("policy_dqn_test.pth")