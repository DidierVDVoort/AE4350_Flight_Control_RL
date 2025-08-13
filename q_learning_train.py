from tqdm import tqdm # progress bar
import matplotlib.pyplot as plt
from sim import PlaneSim
import matplotlib
import numpy as np
import torch
from q_learning_agent import QLearningAgent
from itertools import count

# Training hyperparameters
learning_rate = 0.01
n_episodes = 500
start_epsilon = 1.0 # start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2) # reduces exploration over time
final_epsilon = 0.1 # always keep some exploration

# Possible actions for relative heading changes (0 = straight ahead, 1-4 = left/right turns)
turn_increments = { # Define turn angles in radians
    1: -np.pi/12,   # -15deg (left)
    2: np.pi/12,    # +15deg (right)
    3: -np.pi/6,    # -30deg (sharper left)
    4: np.pi/6      # +30deg (sharper right)
}

# Environment
env = PlaneSim(n_planes=1)

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

# Define agent
agent = QLearningAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
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
    done = False
    total_reward = 0

    # Log average reward every 100 episodes
    if episode % 100 == 0:
        print(f"Ep {episode}: Avg Reward {np.mean(episode_rewards[-100:])}")

    # Get initial observation
    raw_state = env.get_simple_state()
    obs = tuple(env.discretize_state(raw_state))

    # Play one complete simulation (until all planes landed, collided or max. steps is reached)
    for t in count():
        # Agent chooses action for each plane (initially random, gradually more intelligent)
        actions = agent.get_action(obs)

        # Apply the actions to the planes
        for i, action in enumerate([actions]):
            if action != 0 and env.planes[i].active:
                env.planes[i].heading += turn_increments[action]
                env.planes[i].heading = (env.planes[i].heading + np.pi) % (2 * np.pi) - np.pi # clamp heading to [-pi, pi]

        # Observe results of the taken actions
        env.step() # step the environment
        raw_next_obs = env.get_simple_state() # new state (raw)
        next_obs = tuple(env.discretize_state(raw_next_obs)) # new state (discretized)
        reward = env.get_simple_reward(action)
        total_reward += reward
        terminated = env.done or t == env.max_steps

        # Let the agent learn from this environment step
        agent.update(obs, action, reward, terminated, next_obs)

        # Move to next state
        obs = next_obs

        if terminated:
            episode_rewards.append(total_reward)
            plot_rewards()
            break

    # Reduce exploration rate (agent becomes less random over time)
    agent.decay_epsilon()

# Save the trained policy
torch.save(dict(agent.q_values), "policy.pth")