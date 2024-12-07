import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym


class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 9)  # 9 possible moves for Tic-Tac-Toe
        self.fc3 = nn.Linear(256, 1)  # Value head

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 128 * 3 * 3)
        x = torch.relu(self.fc1(x))
        policy_logits = self.fc2(x)
        value = torch.tanh(self.fc3(x))
        return policy_logits, value


def train_network():
    env = gym.make('TicTacToe-v0')  # Assume you have a custom Tic-Tac-Toe gym environment
    net = TicTacToeNet().cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    for episode in range(1000):  # Train for 1000 episodes
        state = env.reset()
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).cuda()
            policy_logits, value = net(state_tensor)
            action = torch.argmax(policy_logits).item()
            next_state, reward, done, _ = env.step(action)
            target_value = reward if done else value

            optimizer.zero_grad()
            policy_loss = criterion_policy(policy_logits, torch.LongTensor([action]).cuda())
            value_loss = criterion_value(value, torch.FloatTensor([target_value]).cuda())
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

            state = next_state
            if done:
                break


if __name__ == "__main__":
    train_network()
