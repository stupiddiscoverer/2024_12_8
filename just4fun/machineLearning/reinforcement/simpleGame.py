import pygame
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# Game Constants
WIDTH, HEIGHT = 400, 600
PLAYER_WIDTH, PLAYER_HEIGHT = 50, 50
OBSTACLE_WIDTH, OBSTACLE_HEIGHT = 50, 50

# Initialize PyGame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()


# Deep Q-Network (DQN) Model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Reinforcement Learning Agent
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmax(action_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target += self.gamma * torch.max(self.model(next_state)).item()
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state).clone()
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = F.mse_loss(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        if os.path.exists(filepath):
            self.model.load_state_dict(torch.load(filepath))
            self.model.eval()


# Game Environment
class Game:
    def __init__(self):
        self.player_x = WIDTH // 2
        self.obstacle_x = random.randint(0, WIDTH - OBSTACLE_WIDTH)
        self.obstacle_y = 0
        self.score = 0

    def step(self, action):
        if action == 0:
            self.player_x = max(0, self.player_x - 5)
        elif action == 1:
            self.player_x = min(WIDTH - PLAYER_WIDTH, self.player_x + 5)
        # Action 2: Stay still (do nothing)

        self.obstacle_y += 5
        if self.obstacle_y > HEIGHT:
            self.obstacle_y = 0
            self.obstacle_x = random.randint(0, WIDTH - OBSTACLE_WIDTH)
            self.score += 1

        done = False
        if (self.obstacle_y + OBSTACLE_HEIGHT > HEIGHT - PLAYER_HEIGHT and
                self.obstacle_x < self.player_x + PLAYER_WIDTH and
                self.obstacle_x + OBSTACLE_WIDTH > self.player_x):
            done = True

        state = [self.player_x / WIDTH, self.obstacle_x / WIDTH, self.obstacle_y / HEIGHT]
        reward = 1 if not done else -10
        return np.array(state), reward, done

    def render(self):
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (0, 255, 0), (self.player_x, HEIGHT - PLAYER_HEIGHT, PLAYER_WIDTH, PLAYER_HEIGHT))
        pygame.draw.rect(screen, (255, 0, 0), (self.obstacle_x, self.obstacle_y, OBSTACLE_WIDTH, OBSTACLE_HEIGHT))
        pygame.display.flip()


# Training Loop
state_size = 3
action_size = 3  # Now includes 'stay still'
agent = Agent(state_size, action_size)
game = Game()
batch_size = 32
episodes = 5000
model_filepath = "dqn_model.pth"
# Load previous model if exists
agent.load_model(model_filepath)


def train():
    for episode in range(episodes):
        state = np.array([game.player_x / WIDTH, game.obstacle_x / WIDTH, game.obstacle_y / HEIGHT])
        done = False
        step_count = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            step_count += 1
            if game.score > 200:
                agent.save_model(model_filepath)
                print("score > 200! the end")
                render_game()
                return
        agent.replay(batch_size)
        print(f"Episode {episode + 1}/{episodes} - Score: {game.score}")

        # Only render every 100 episodes after the game ends
        if episode % 100 == 0:
            render_game()

        game.__init__()


def render_game():
    """Runs the game for one episode and renders it."""
    game.__init__()
    state = np.array([game.player_x / WIDTH, game.obstacle_x / WIDTH, game.obstacle_y / HEIGHT])
    done = False
    while not done:
        pygame.event.pump()
        action = agent.act(state)
        state, _, done = game.step(action)
        game.render()
        clock.tick(60)


train()
# render_game()
pygame.quit()