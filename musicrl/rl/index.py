#from env import *
#env = MelEnvironment(None, mapper)
from agent import *
import numpy as np
import matplotlib.pyplot as plt

batch_size = 128
rewards = []
avg_rewards = []

class FakeEnv:
    def __init__(self, dim_seq):
        self.dim_seq = dim_seq
        self.action = np.ones(self.dim_seq)
        self.state = self.action

    def reset(self):
        return self.state

    def step(self, action):
        self.state = np.vstack((self.state,self.action))
        return self.state, 1, False, None

env = FakeEnv(10)
agent = DPPGAgent(env)

for episode in range(50):
    state = env.reset()
    episode_reward = 0
    action = agent.get_action(state)
    print(action)
    # todo: make action noisy?
    new_state, reward, done, info = env.step(action)

    agent.memory.push(state, action, reward, new_state, done)

    if len(agent.memory) > batch_size:
            agent.update(batch_size)

    state = new_state
    episode_reward += reward

    if done:
        sys.stdout.write(
            "episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2),  np.mean(rewards[-10:])))
        break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()