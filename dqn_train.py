import torch
import time
from tqdm import tqdm
from sim import PlaneSim
from dqn_agent import DQNAgent
from itertools import count
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random

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

# Number of states and actions
state_size = 6 # matches get_dqn_state() output
action_size = 5 # [none, left15, right15, left30, right30]

# Training hyperparameters
seed = 5
n_episodes = 200
n_planes = 5
max_replay_memory = 10000
discount_factor = 0.99
learning_rate = 0.01
start_epsilon = 1.0 # start with 100% random actions
final_epsilon = 0.01 # always keep some exploration
epsilon_decay = 0.95
batch_size = 64

# Set all seeds
set_all_seeds(seed)

# Possible actions for relative heading changes (0 = straight ahead, 1-4 = left/right turns)
turn_increments = { # Define turn angles in radians
    1: -np.pi/12,   # -15deg (left)
    2: np.pi/12,    # +15deg (right)
    3: -np.pi/6,    # -30deg (sharper left)
    4: np.pi/6      # +30deg (sharper right)
}

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

# Live reward and epsilon plotting function
def plot_rewards_and_epsilon(show_result=False):
    plt.figure(1, figsize=(10, 8))
    
    # Create two subplots next to each other
    ax1 = plt.subplot(2, 1, 1) # reward plot
    ax2 = plt.subplot(2, 1, 2) # epsilon plot
    
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float) # convert to tensor
    epsilon_t = torch.tensor(episode_epsilons, dtype=torch.float) # convert to tensor
    
    # Plot rewards
    if show_result:
        ax1.set_title('Rewards Over Time for DQN Algorithm')
    else:
        ax1.clear()
        ax1.set_title('Training...')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.plot(rewards_t.numpy())

    # Take 50 episode averages and plot them too (only after 50th episode of course)
    if len(rewards_t) >= 50:
        means = rewards_t.unfold(0, 50, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(49), means))
        ax1.plot(means.numpy(), label='Reward moving average with window=50')

    # Plot epsilon
    if show_result:
        ax2.set_title('Epsilon Over Time for Classic Q-Learning Algorithm')
    else:
        ax2.clear()
        ax2.set_title('Epsilon Over Time for Classic Q-Learning Algorithm')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.plot(epsilon_t.numpy())
    ax2.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.pause(0.001) # pause so that plots are updated

    # Clear output for next plot
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

episode_rewards = [] # store episode rewards
episode_epsilons = [] # store episode epsilons
episode_stats = { # store some more episode statistics
    'steps_per_episode': [],
    'success_rate': [],
    'collisions': []
}

# Start timing
start_time = time.time()

# Training loop
for episode in tqdm(range(n_episodes)):
    # Reset environment
    env.reset()
    total_reward = 0
    states = [] # initialize states (one entry for each plane)

    # Store epsilon value at the start of the episode
    episode_epsilons.append(agent.epsilon)

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
            episode_rewards.append(total_reward)
            landed = sum(1 for p in env.planes if not p.active and not env.collision)
            print(f"Planes landed: {landed}/{env.n_planes}")

            # Check for collision
            if env.collision:
                collided = 1
            else:
                collided = 0

            # Save episode statistics
            episode_stats['steps_per_episode'].append(t + 1) # +1 because steps start at 0
            episode_stats['success_rate'].append(landed / n_planes)
            episode_stats['collisions'].append(collided)

            plot_rewards_and_epsilon()
            break

    # Decay exploration rate at end of episode (agent becomes less random over time)
    agent.decay_epsilon()

    # Update target model every 10 episodes
    if episode % 10 == 0:
        agent.update_target_model() # sync target network

# End timing
end_time = time.time()
training_time = end_time - start_time

print(f"DQN training completed in {training_time:.2f} seconds")

# Show final results after training completes
plt.close('all')
plot_rewards_and_epsilon(show_result=True)

# Keep the plot open
if not is_ipython:
    plt.ioff()
    plt.show()

# Save the model (policy)
agent.save("policy_dqn_test.pth")

# Save training results for comparison
training_results = {
    'episode_rewards': torch.tensor(episode_rewards),
    'episode_epsilons': torch.tensor(episode_epsilons),
    'episode_stats': episode_stats,
    'algorithm': 'DQN',
    'parameters': {
        'n_episodes': n_episodes,
        'learning_rate': learning_rate,
        'discount_factor': discount_factor,
        'epsilon_decay': epsilon_decay
    },
    'seed': seed,
    'training_time': training_time
}

# torch.save(training_results, "dqn_seed_5.pth")