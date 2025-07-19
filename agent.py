import torch
import torch.nn as nn
import random
from collections import deque

# Try to use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Deep Q-Network (DQN) Agent
# Uses a neural network to approximate Q-values for each action
class DQNAgent:
    def __init__(self, state_size, action_size, n_planes):
        self.state_size = state_size
        self.action_size = action_size
        self.n_planes = n_planes
        self.memory = deque(maxlen=2000) # Experience replay memory
        # Part of hyperparameters
        self.learning_rate = 0.001
        self.gamma = 0.95 # discount factor for Bellman equation
        self.epsilon = 1.0 # start with full exploration
        self.epsilon_min = 0.1  # do not drop below 10% exploration 
        self.epsilon_decay = 0.999 # slower decay (original: 0.995 was too fast)

        self.device = device
        self.model = self.build_model().to(device) # initialize the neural network model
        self.criterion = nn.MSELoss() # loss function for Q-learning
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) # optimizer for the model

    # Simple neural network for Q-learning
    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.n_planes * self.action_size)
        )
        return model
    
    # Store experience in memory
    def remember(self, state, actions, reward, next_state, done):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        self.memory.append((state, actions, reward, next_state, done))

    # Choose action based on epsilon-greedy policy
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device) # convert to [1, state_size]
        
        # Get Q-values for all actions
        with torch.no_grad():
            q_values = self.model(state) # shape is [1, n_planes * action_size]
        
        # Reshape to [n_planes, action_size]
        q_values = q_values.view(self.n_planes, self.action_size)

        # Epsilon-greedy action selection
        actions = []
        for i in range(self.n_planes):
            if random.random() <= self.epsilon:
                actions.append(random.randrange(self.action_size)) # random action for plane i
            else:
                actions.append(torch.argmax(q_values[i]).item()) # greedy action for plane i
        return actions
    
    # Train the agent using a batch of experiences
    def replay(self, batch_size):
        # If not enough experiences, skip training
        if len(self.memory) < batch_size:
            return

        # Sample a random minibatch from memory
        minibatch = random.sample(self.memory, batch_size)

        # Loop through each experience in the minibatch
        for state, actions, reward, next_state, done in minibatch:
            # Convert to tensors
            state = state.unsqueeze(0).to(self.device) # shape is [1, state_size]
            next_state = next_state.unsqueeze(0).to(self.device) # shape is [1, state_size]

            # Current Q-values
            q_values = self.model(state).view(self.n_planes, self.action_size) # shape is [n_planes, action_space]
            q_next = self.model(next_state).view(self.n_planes, self.action_size) # shape is [n_planes, action_space]

            # Compute targets for each plane
            q_target = q_values.clone().detach() # do not backpropagate through target

            for i in range(self.n_planes):
                if done:
                    q_target[i][actions[i]] = reward # full reward if done
                else:
                    q_target[i][actions[i]] = reward + self.gamma * torch.max(q_next[i]).item() # Bellman update (agent expects more reward in future)

            # Compute loss and backpropagate
            self.optimizer.zero_grad()
            loss = self.criterion(q_values, q_target)
            loss.backward()
            self.optimizer.step()

        # Decay epsilon for more exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
