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

filepaths = list(glob("maestro-v2.0.0/2008/**.midi"))
real_midis = [pretty_midi.PrettyMIDI(i) for i in filepaths]

mapper = MidiVectorMapper(real_midis)

model = mel_lstm.get_model()

train_generator = RandomMidiDataGenerator(
    real_midis[:50], mel_lstm.preprocess, mapper, 4
)
test_generator = RandomMidiDataGenerator(
    real_midis[50:60], mel_lstm.preprocess, mapper, 10, max_num_timeframes=1000
)

history = model.fit_generator(
    train_generator(),
    validation_data=test_generator(),
    steps_per_epoch=train_generator.steps_per_epoch,
    epochs=2,
    validation_steps=test_generator.steps_per_epoch,
    validation_freq=1,
    verbose=1,
)
mel_lstm.plot_history(history)
plt.show()
