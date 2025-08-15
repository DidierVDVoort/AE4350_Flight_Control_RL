from tqdm import tqdm # progress bar
import matplotlib.pyplot as plt
from sim import PlaneSim
import matplotlib
import numpy as np
import torch
from q_learning_agent import QLearningAgent, QLearningMultiAgent
from itertools import count

# Training hyperparameters
learning_rate = 0.01
n_episodes = 2000
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
env = PlaneSim(n_planes=3)

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

# Define agents (one for each plane)
agents = [QLearningMultiAgent(plane_idx=i, env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon
) for i in range(env.n_planes)]

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
    obs = [] # initialize observations (for each agent)
    total_reward = [0 for _ in range(env.n_planes)] # list of total rewards (one entry per agent)

    # Log average reward every 100 episodes
    if episode % 100 == 0:
        print(f"Ep {episode}: Avg Reward {np.mean(episode_rewards[-100:])}")

    # Get initial observations
    for agent in agents:
        raw_state = env.get_simple_state(agent.plane_idx)
        obs.append(tuple(env.discretize_state(raw_state)))

    # Play one complete simulation (until all planes landed, collided or max. steps is reached)
    for t in count():
        actions = [] # reset actions every time (need to be stored for the agent.update() function)
        
        # Agents choose action for their plane (initially random, gradually more intelligent)
        for agent in agents:
            action = agent.get_action(obs[agent.plane_idx])
            actions.append(action)

            # Apply the action to the plane
            if action != 0 and env.planes[agent.plane_idx].active:
                env.planes[agent.plane_idx].heading += turn_increments[action]
                env.planes[agent.plane_idx].heading = (env.planes[agent.plane_idx].heading + np.pi) % (2 * np.pi) - np.pi # clamp heading to [-pi, pi]

        # Observe results of the taken actions
        env.step() # step the environment
        terminated = env.done or t == env.max_steps

        # Loop over agents (planes) and update them with the new observations and rewards
        for agent in agents:
            raw_next_obs = env.get_simple_state(agent.plane_idx) # new state (raw)
            next_obs = tuple(env.discretize_state(raw_next_obs)) # new state (discretized)
            reward = env.get_simple_reward(agent.plane_idx)
            total_reward[agent.plane_idx] += reward

            # Let the agent learn from this environment step
            agent.update(obs[agent.plane_idx], actions[agent.plane_idx], reward, terminated, next_obs)

            # Move to next state
            obs[agent.plane_idx] = next_obs

        if terminated:
            episode_rewards.append(sum(total_reward))
            print(f"Planes landed: {sum(1 for p in env.planes if not p.active and not env.collision)}/{env.n_planes}")
            plot_rewards()
            break

    # Reduce exploration rate (agent becomes less random over time)
    for agent in agents:
        agent.decay_epsilon()

# Save all policies (i.e. 1 for each agent) in a dictionary with their indices as keys
policies = {i: dict(agent.q_values) for i, agent in enumerate(agents)}
torch.save(policies, "policy_multi_agent_3_planes.pth")