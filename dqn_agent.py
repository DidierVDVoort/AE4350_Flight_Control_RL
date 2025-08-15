import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    """Q-network with 3 fully connected layers."""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, max_replay_memory_size=10000, discount_factor=0.99, learning_rate=0.001,
                 initial_epsilon=1.0, final_epsilon=0.01, epsilon_decay=0.995, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=max_replay_memory_size) # replay buffer
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon # exploration rate
        self.epsilon_min = final_epsilon # minimum exploration rate
        self.epsilon_decay = epsilon_decay # decay rate for exploration
        self.batch_size = batch_size # size of mini-batch for training
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.update_target_model()

    def update_target_model(self):
        """Copy weights from policy network to target network."""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer (memory)"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Select action (epsilon-greedy)"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0) # convert state to tensor
        act_values = self.model(state)
        return torch.argmax(act_values).item() # else select best action based on Q-values

    def replay(self):
        """Train model on a batch of past experiences"""
        if len(self.memory) < self.batch_size: # memory not enough yet
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch) # unpack minibatch

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions) # actions are integers (discrete) hence LongTensor
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Compute Q-values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.discount_factor * next_q # Compute target Q-values

        # Compute loss and update model
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        """Save model weights to file"""
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """Load model weights from file"""
        self.model.load_state_dict(torch.load(path))