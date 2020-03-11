import numpy as np
from matplotlib import pyplot as plt

import gym

from musicrl.render import *


na = None


class MelEnvironment(gym.Env):
    """Environment to grain generating midi data in a self defined
    vector space. The midi vector representation is defined via the
    mapper object.
    The waveform for the single instrument is then preprocessed for the
    discriminator, and at each time step, the discriminators final prediction
    serves as reward.
    The preprocessed waveform, i.e. the mel spectrogram, also serves as
    observation. The number of time frames that are used for the observation
    are defined by the constant `self.N_TIMESTEPS`.
    One session is understood as one song.
    
    Gets:
        discriminator: keras.Model: np.array(preprocessed) -> np.array(#time_steps, 1)
        preprocess: function: np.array(#actions): waveform -> np.array(preprocessed) : spectrogram
        mapper: musicrl.midi2vec.MidiVectorMapper
        N_TIMESTEPS: int: number of timesteps used to generate the observation
        MAX_NUM_ACTIONS: int: number of actions after which to end a trajectory
    """
    def __init__(self, discriminator, preprocess_wav, mapper, N_TIMESTEPS=100, MAX_NUM_ACTIONS=10000):
        super().__init__()
        # N_TIMESTEPS is used to define the observation:
        # This many timeframes of the spectrogram are fed
        # back to the generator
        self.N_TIMESTEPS = N_TIMESTEPS
        self.MAX_NUM_ACTIONS = MAX_NUM_ACTIONS
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = gym.spaces.Box(0, np.inf, shape=(mapper.dims,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                           shape=(self.N_TIMESTEPS, 128), dtype=np.float32)
        self.discriminator = discriminator
        self.preprocess_wav = preprocess_wav
        self.mapper = mapper
        self.fr = 44100
        self.rewards = []
        self.current_seq = []
        self.current_midi = pretty_midi.PrettyMIDI(resolution=384, initial_tempo=300)
        self.current_midi.instruments.append(pretty_midi.Instrument(program=0))
        self.current_observation = np.zeros((self.N_TIMESTEPS, 128))
     
    def _update_wav(self, action):
        self.current_seq.append(action)
        event = mapper.action2note(action)
        if isinstance(event, pretty_midi.Note):
            if len(self.current_midi.instruments[0].notes) == 1:
                # It is the first note, so we synthesize
                self.current_midi.instruments[0].notes.append(event)
                self.current_midi.instruments[0].synthesize(self.fr)
            else:
                self.current_midi.instruments[0].append_and_synthesize(event)
            return True
        
        
    def step(self, action):
        self._update_wav(action)
        preprocessed = self.preprocess_wav(self.current_wav, self.fr)[na]
        prediction = self.discriminator.predict_on_batch(preprocessed)
        observation = np.zeros((self.N_TIMESTEPS, 128))
        observation[-min(self.N_TIMESTEPS, len(preprocessed[0])):] = preprocessed[0, -self.N_TIMESTEPS:]
        self.current_observation = observation
        self.current_prediction = prediction
        reward = prediction[0, -1, 0]
        self.rewards.append(reward)
        # TODO: add a end token to mapper (issue #1)
        done = len(self.current_seq) >= self.MAX_NUM_ACTIONS
        return observation, reward, done, None

    def reset(self):
        self.current_seq = []
        self.current_midi = pretty_midi.PrettyMIDI(resolution=384, initial_tempo=300)
        self.current_midi.instruments.append(pretty_midi.Instrument(program=0))
        self.current_observation = np.zeros((self.N_TIMESTEPS, 128))
        self.rewards = []
        return self.current_observation
    
    @property
    def current_wav(self):
        return self.current_midi.instruments[0].synthesized

    def render(self, mode='human'):
        plot_spectro(self.current_observation.T, "Current observation")
    
    def close (self):
        pass
