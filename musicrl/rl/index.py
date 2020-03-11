import sys
sys.path.append("../")
sys.path.append("../../")
from environment import *

from tensorflow.keras.models import Sequential, load_model
from musicrl import mel_lstm

from glob import glob




import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras import metrics

import gym

from musicrl.midi2vec import MidiVectorMapper
from musicrl.render import *
from musicrl.random_generator import generate_random_midi, resemble_midi
from musicrl.data import RandomMidiDataGenerator
from musicrl import mel_lstm


filepaths = list(glob('../../maestro-v2.0.0/2008/**.midi'))
real_midis = [pretty_midi.PrettyMIDI(i) for i in filepaths]
mapper = MidiVectorMapper(real_midis)

mapper = MidiVectorMapper(real_midis)
real_seq = mapper.midi2vec(real_midis[1])
real_seq.shape



discriminator = load_model("../../models/mel_lstm.h5")
env = MelEnvironment(discriminator, mel_lstm.preprocess_wav, mapper, 1000)


notes = []
for event in real_seq:
    if isinstance(mapper.action2note(event), pretty_midi.Note):
        notes.append(event)


for i, action in enumerate(notes):
    env.step(action)
    if i % 500 == 0:
        env.render()
        plt.show()
    if i > 500: break




from agent import *
import numpy as np
import matplotlib.pyplot as plt


batch_size = 128
rewards = []
avg_rewards = []

agent = DPPGAgent(env)

for episode in range(50):
    state = env.reset()
    episode_reward = 0
    action = agent.get_action(state)
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