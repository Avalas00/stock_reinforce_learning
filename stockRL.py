import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from FinMind.data import DataLoader
from collections import deque
import os
import csv

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class DQNTrader:
    def __init__(self, state_size, action_num=3, model_name="DQN_trader",
                 initial_balance=1000, trade_fee=0.001):
        self.state_size = state_size
        self.action_num = action_num
        self.memory = deque(maxlen=2000)
        self.inventory = []
        self.model_name = model_name

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.1
        self.epsilon_decay = 0.995

        self.model = self.model_dnn().to(device)
        self.target_model = self.model_dnn().to(device)  # 添加目标网络
        self.update_target_model()  # 初始化时更新目标网络
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

        self.balance = initial_balance
        self.trade_fee = trade_fee
        self.episode_rewards = []  # 累積每一年的獎勵

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


    def model_dnn(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_num)
        )
        return model

    def trade(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_num)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        actions = self.model(state)
        return actions.argmax().item()

    def record_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_weights(self):
        batch_size = len(self.memory)
        if batch_size == 0:
            return

        state_batch = torch.FloatTensor([s[0] for s in self.memory]).to(device)
        action_batch = torch.LongTensor([s[1] for s in self.memory]).to(device)
        reward_batch = torch.FloatTensor([s[2] for s in self.memory]).to(device)
        next_state_batch = torch.FloatTensor([s[3] for s in self.memory]).to(device)
        done_batch = torch.FloatTensor([s[4] for s in self.memory]).to(device)

        q_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_state_batch).max(1)[0]
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = self.criterion(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

        self.memory.clear()  # 清空记忆库
        self.update_target_model()  # 更新目标网络

    def save_model(self, episode):
        model_path = f'{self.model_name}_{episode + 1}.pth'
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved at {model_path}")

    def load_model(self, model_path):
        if os.path.isfile(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            print(f"Model loaded from {model_path}")
        else:
            print(f"No model found at {model_path}")

def get_data(stock_id, start_date, end_date):
    dl = DataLoader()
    data = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date, end_date=end_date)
    return data


def train_dqn(stock_id, start_date, end_date, episodes,
              initial_balance=1000, trade_fee=0.001,
              save_every=10, model_name=None, bank_rate=0.02, num_of_transactions=1):
    data = get_data(stock_id, start_date, end_date)
    data = data.drop(columns=['date'])
    data = data.astype(np.float32)
    state_size = len(data.columns)
    trader = DQNTrader(state_size, initial_balance=initial_balance, trade_fee=trade_fee)
    trade_log = []

    if model_name is not None:
        trader.load_model(model_name)
    start = time.time()
    for episode in range(episodes):
        state = data.iloc[0].values
        total_profit = 0
        trader.inventory = []
        trader.balance = initial_balance
        trade_count = 0
        episode_reward = 0  # 紀錄這一年的總獎勵

        for t in range(1, len(data)):
            next_state = data.iloc[t].values
            action = trader.trade(state)
            reward = 0

            # 執行交易邏輯
            if action == 1:  # Buy
                price = data.iloc[t]['close'] * (1 + trade_fee) * num_of_transactions
                if trader.balance >= price:
                    trader.inventory.append(price)
                    trader.balance -= price
                    print(f"Buy: {data.iloc[t]['close']}, Trade Fee: {price * trade_fee * num_of_transactions:.2f}, Total Expense: {price:.2f}, Balance: {trader.balance:.2f}")
                    trade_log.append((episode+1, t+1, 'Buy', data.iloc[t]['close'], price, price * trade_fee, trader.balance, None))
                    trade_count += 1
            elif action == 2 and len(trader.inventory) > 0:  # Sell
                bought_price = trader.inventory.pop(0)
                price = data.iloc[t]['close'] * (1 - trade_fee) * num_of_transactions
                reward = price - bought_price
                trader.balance += price
                total_profit += reward
                print(f"Sell: {data.iloc[t]['close']}, Trade Fee: {price * trade_fee * num_of_transactions:.2f}, Total Expense: {price:.2f},Profit: {reward:.2f}, Balance: {trader.balance:.2f}")
                trade_count += 1

                trade_log.append((episode+1, t+1, 'Sell', data.iloc[t]['close'], price, price * trade_fee, trader.balance, reward))

            else:
                reward -= (bank_rate * initial_balance / 365)

            episode_reward += reward  # 累積這一年的獎勵
            done = t == len(data) - 1
            trader.record_transition(state, action, reward, next_state, done)
            state = next_state

            if done:
                while len(trader.inventory) > 0:
                    bought_price = trader.inventory.pop(0)
                    price = data.iloc[t]['close'] * (1 - trade_fee) * num_of_transactions
                    reward = price - bought_price
                    trader.balance += price
                    total_profit += reward
                    trade_log.append((episode+1, t+1, 'Final Sell', data.iloc[t]['close'], price, trader.balance, reward))
                    episode_reward += reward
                    print(f"Final Sell: {data.iloc[t]['close']}, Trade Fee: {price * trade_fee:.2f}, Total Expense: {price:.2f}, Profit: {reward:.2f}, Balance: {trader.balance:.2f}")
                end = time.time()
                print(f"Total train time: {end-start:.2f}")
                print(f"Episode: {episode+1}/{episodes}, Total Profit: {total_profit:.2f}, Final Balance: {trader.balance:.2f}, Profit Rate: {((total_profit/initial_balance)*100):.2f}%, Trade Count: {trade_count}, Episode Reward: {episode_reward:.2f}")


        trader.episode_rewards.append(episode_reward)  # 保存這一年的總獎勵
        trader.update_weights()  # 在年度結束後更新權重

        if (episode+1) % save_every == 0:
            trader.save_model(episode)
    with open('trade_log.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['episode', 'date', 'action', 'close price', 'price', 'trade_fee', 'balance', 'profit'])
        writer.writerows(trade_log)
    print("Trade log saved to trade_log.csv")

    plt.plot(range(1, episodes + 1), trader.episode_rewards, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()
    plt.savefig('reward_per_episode.png')
    plt.show()
    print("Reward plot saved to reward_per_episode.png")
# 訓練模型
train_dqn('2330', '2020-01-01', '2021-01-01', episodes=100, initial_balance=10000,
          trade_fee=0.001, save_every=100, model_name='DQN_trader_1000.pth',
          bank_rate=0.02, num_of_transactions=5)
