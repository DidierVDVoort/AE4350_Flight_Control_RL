from collections import defaultdict
import numpy as np

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, initial_epsilon=1.0, 
                 epsilon_decay=0.9995, final_epsilon=0.01, discount_factor=0.99):
        self.env = env
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        
        # Initialize Q-table with zeros for all possible discrete states
        self.q_values = defaultdict(self.zero_values)
        self.state_bins = 10 # number of bins per state dimension (for discretization)

    def zero_values(self):
        """Return zero values for all actions (for initialization)"""
        return np.zeros((self.env.n_planes, 5)) # shape: (n_planes, 5 actions)
    
    def discretize_state(self, state):
        """Discretize the raw state"""
        return self.env.discretize_state(state, bins=self.state_bins)
    
    def get_action(self, raw_state):
        """Get action for a raw state (after discretizing)"""
        discrete_state = self.discretize_state(raw_state)
        actions = []
        
        for plane_idx in range(self.env.n_planes):
            # For active planes
            if np.random.random() < self.epsilon:
                actions.append(np.random.randint(0, 5))  # random action (exploration)
            else:
                actions.append(np.argmax(self.q_values[discrete_state][plane_idx]))  # exploit learned Q-values
        return actions

    def update(self, raw_state, actions, reward, terminated, next_raw_state):
        """Update Q-table with discretized states"""
        state = self.discretize_state(raw_state)
        next_state = self.discretize_state(next_raw_state)
        
        for plane_idx, action in enumerate(actions):
            current_q = self.q_values[state][plane_idx][action]
            max_future_q = np.max(self.q_values[next_state][plane_idx]) if not terminated else 0
            new_q = current_q + self.lr * (reward + self.discount_factor * max_future_q - current_q)
            self.q_values[state][plane_idx][action] = new_q

    def decay_epsilon(self):
        """Reduce exploration rate after each episode"""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

# QLearningMultiAgent is a version of QLearningAgent for multi-agent scenarios (i.e. when there are multiple planes)
# The setup is still independent, meaning that each plane is assigned a separate agent (perhaps not optimal, but easiest to implement)
class QLearningMultiAgent:
    def __init__(self, plane_idx, env, learning_rate=0.1, initial_epsilon=1.0, 
                 epsilon_decay=0.9995, final_epsilon=0.01, discount_factor=0.99):
        self.plane_idx = plane_idx
        self.env = env
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        
        # Initialize Q-table with zeros for all possible discrete states
        self.q_values = defaultdict(self.zero_values)
        self.state_bins = 10 # number of bins per state dimension (for discretization)

    def zero_values(self):
        """Return zero values for all actions (for initialization)"""
        return np.zeros(5)  # 5 actions
    
    def discretize_state(self, state):
        """Discretize the raw state"""
        return self.env.discretize_state(state, bins=self.state_bins)
    
    def get_action(self, raw_state):
        """Get action for a raw state (after discretizing)"""
        discrete_state = self.discretize_state(raw_state)
        
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 5) # random action (exploration)
        else:
            return np.argmax(self.q_values[discrete_state]) # exploit learned q-values

    def update(self, raw_state, action, reward, terminated, next_raw_state):
        """Update Q-table with discretized states"""
        state = self.discretize_state(raw_state)
        next_state = self.discretize_state(next_raw_state)

        # # Log Q-values if the plane is close to the landing zone (if Q-values are large for certain actions, the agent expects a high future reward, meaning it has learned something)
        # if raw_state[2] < 20: # distance to target < 20 units 
        #     print(f"Plane {self.plane_idx} | State: {state} | Q-values: {self.q_values[state]}")
        
        current_q = self.q_values[state][action]
        max_future_q = np.max(self.q_values[next_state]) if not terminated else 0
        new_q = current_q + self.lr * (reward + self.discount_factor * max_future_q - current_q) # Bellman equation
        self.q_values[state][action] = new_q

    def decay_epsilon(self):
        """Reduce exploration rate after each episode"""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)