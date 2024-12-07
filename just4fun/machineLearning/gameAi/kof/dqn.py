import random
from collections import deque

import numpy as np
import torch
from torch import optim, nn

from machineLearning.gameAi.kof.reward import KOF98Env
from machineLearning.gameAi.kof.network import DQN


class DQNAgent:
    def __init__(self, state_shape, num_actions):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_shape, num_actions)
        self.target_model = DQN(state_shape, num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state)[0]).item()
            target_f = self.model(state).clone()
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def preprocess_state(state):
    return np.expand_dims(np.expand_dims(state, axis=0), axis=0)


if __name__ == "__main__":
    env = KOF98Env()
    state_shape = (1, 84, 84)
    num_actions = 8
    agent = DQNAgent(state_shape, num_actions)
    episodes = 1000
    batch_size = 32

    for e in range(episodes):
        state = preprocess_state(env.reset())
        for time_t in range(500):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = preprocess_state(next_state)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{episodes}, score: {time_t}, e: {agent.epsilon:.2}")
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if e % 10 == 0:
            agent.update_target_model()
