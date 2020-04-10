from collections import deque
import numpy as np

from models import *


class DPPGAgent:
    def __init__(self, env, max_memory_size=50000):
        # Init Networks
        self.actor = Actor()
        self.critic = Critic()
        self.actor_target = Actor()
        self.actor_target = Critic()

        self.memory = Memory(max_memory_size)

    def get_action(self, state):
        action = self.actor.predict(state)
        return action


class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)
