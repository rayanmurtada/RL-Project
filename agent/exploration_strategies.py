import numpy as np
import random

class EpsilonGreedy:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def select_action(self, q_values):
        if random.random() < self.epsilon:
            return random.randint(0, len(q_values) - 1)
        return np.argmax(q_values)

class Softmax:
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def select_action(self, q_values):
        exp_q = np.exp(q_values / self.temperature)
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(len(q_values), p=probs)

class UCB:
    def __init__(self, c=2):
        self.c = c
        self.action_count = np.zeros((1, len(q_values)))

    def select_action(self, q_values):
        total_counts = np.sum(self.action_count)
        if total_counts == 0:
            return random.randint(0, len(q_values) - 1)
        ucb_values = q_values + self.c * np.sqrt(np.log(total_counts) / (self.action_count + 1e-5))
        return np.argmax(ucb_values)

class Boltzmann:
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def select_action(self, q_values):
        exp_q = np.exp(q_values / self.temperature)
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(len(q_values), p=probs)

class Random:
    def select_action(self, q_values):
        return random.randint(0, len(q_values) - 1)
