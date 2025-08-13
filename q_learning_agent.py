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
        return np.zeros(5) # 5 possible actions
    
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
        
        current_q = self.q_values[state][action]
        max_future_q = np.max(self.q_values[next_state]) if not terminated else 0
        new_q = current_q + self.lr * (reward + self.discount_factor * max_future_q - current_q) # Bellman equation
        self.q_values[state][action] = new_q

    def decay_epsilon(self):
        """Reduce exploration rate after each episode"""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)