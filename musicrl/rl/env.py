## Imports and data loading

import pdb
import sys
sys.path.append("../../")

import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras import metrics

from musicrl.midi2vec import MidiVectorMapper
from musicrl.render import *
from musicrl.random_generator import generate_random_midi, resemble_midi
from musicrl.data import RandomMidiDataGenerator

import pretty_midi
from glob import glob


REAL = 1
GEN = 0



filepaths = list(glob('../../maestro-v2.0.0/2008/**.midi'))
real_midis = [pretty_midi.PrettyMIDI(i) for i in filepaths]
mapper = MidiVectorMapper(real_midis)


mapper = MidiVectorMapper(real_midis)
real_seq = mapper.midi2vec(real_midis[1])
real_seq.shape


notes = []
for event in real_seq:
    if isinstance(mapper.action2note(event), pretty_midi.Note):
        notes.append(event)

import gym
import pretty_midi


class MelEnvironment(gym.Env):
    """We ignore control change events for now
    """

    def __init__(self, discriminator, mapper):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input:
        # self.observation_space = spaces.Box(low=0, high=255,
        #                                    shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
        self.discriminator = discriminator
        self.mapper = mapper
        self.fr = 44100
        self.current_seq = []
        self.current_midi = pretty_midi.PrettyMIDI(resolution=384, initial_tempo=300)
        self.current_midi.instruments.append(pretty_midi.Instrument(program=0))

    def step(self, action):
        self.current_seq.append(action)
        event = mapper.action2note(action)
        if isinstance(event, pretty_midi.Note):

            if len(self.current_midi.instruments[0].notes) == 1:
                # It is the first note, so we synthesize
                self.current_midi.instruments[0].notes.append(event)
                self.current_midi.instruments[0].synthesize(self.fr)
            else:
                pdb.set_trace()
                self.current_midi.instruments[0].append_and_synthesize(event)
        return None
        return observation, reward, done, info

    def reset(self):
        self.current_seq = []
        self.current_midi = pretty_midi.PrettyMIDI(resolution=384, initial_tempo=300)
        self.current_midi.instruments.append(pretty_midi.Instrument(program=0))
        return observation  # reward, done, info can't be included

    @property
    def current_wav(self):
        return self.current_midi.instruments[0].synthesized

    def render(self, mode='human'):
        pass

    def close(self):
        pass


env = MelEnvironment(None, mapper)

print("here")
for action in notes:
    print(action)
    env.step(action)

#display(Audio(env.current_wav, rate=44100))