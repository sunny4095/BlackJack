import gymnasium as gym
import numpy as np
from collections import defaultdict
import random


env = gym.make('Blackjack-v1', render_mode='human')

# Hyperparameters
alpha = 0.1  
gamma = 0.99  
epsilon = 1.0  
epsilon_decay = 0.9999
min_epsilon = 0.01
num_episodes = 500000

# Q-table initialized with zeros
Q = defaultdict(lambda: np.zeros(env.action_space.n))

def epsilon_greedy_policy(state, epsilon):
    state = state[0]
    
    if random.random() < epsilon:
        return env.action_space.sample()  
    else:
        return np.argmax(Q[state])  

def update_q_table(state, action, reward, next_state, done):

    state = state[0]
    next_state = next_state[0]
    
    best_next_action = np.argmax(Q[next_state])
    td_target = reward + (gamma * Q[next_state][best_next_action]) * (1 - done)
    td_delta = td_target - Q[state][action]
    Q[state][action] += alpha * td_delta


for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        action = epsilon_greedy_policy(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        
        update_q_table(state, action, reward, next_state, done)
        
        state = next_state
    
    
    epsilon = max(min_epsilon, epsilon * epsilon_decay)


def play_blackjack(env, num_games=10):
    for game in range(num_games):
        state = env.reset()
        done = False
        while not done:
            
            state_tuple = state[0]
            action = np.argmax(Q[state_tuple])  
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
        print(f"Game {game + 1}: Reward = {reward}")


play_blackjack(env)







