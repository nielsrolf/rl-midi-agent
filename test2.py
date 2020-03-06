import numpy as np
from matplotlib import pyplot as plt

import pretty_midi
from glob import glob

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import TimeDistributed

from musicrl.midi2vec import MidiVectorMapper
from musicrl.render import *
from musicrl.random_generator import generate_random_midi, resemble_midi
from musicrl.data import RandomMidiDataGenerator
from musicrl import mel_lstm

filepaths = list(glob('maestro-v2.0.0/2008/**.midi'))
real_midis = [pretty_midi.PrettyMIDI(i) for i in filepaths]

mapper = MidiVectorMapper(real_midis)

model = mel_lstm.get_model()
train_generator = RandomMidiDataGenerator(real_midis[::2], mel_lstm.preprocess, mapper, 10, 8000)
test_generator = RandomMidiDataGenerator(real_midis[1::2], mel_lstm.preprocess, mapper, 10, 1000)
test_data = test_generator.compute_batch()
x_train, y_train = train_generator.compute_batch()

history = model.fit(x_train, y_train, 10, validation_data=test_data, epochs=30)
mel_lstm.plot_history(history)
plt.savefig("test_mel_lstm_training.png")