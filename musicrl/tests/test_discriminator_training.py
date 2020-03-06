import unittest
from glob import glob

import pretty_midi
import numpy as np

from musicrl.midi2vec import MidiVectorMapper
from musicrl.data import RandomMidiDataGenerator
from musicrl import seq_lstm, mel_lstm


filepaths = list(glob('maestro-v2.0.0/2008/**.midi'))[:5]
real_midis = [pretty_midi.PrettyMIDI(i) for i in filepaths]
mapper = MidiVectorMapper(real_midis)


class DiscriminatorTrainingTest(unittest.TestCase):
    def test_with_seq_lstm(self):
        """
        Test that a very short training runs without exceptions
        """
        model = seq_lstm.get_model()
        train_generator = RandomMidiDataGenerator(real_midis[::2], seq_lstm.make_preprocessor(mapper), mapper, 2, 100)
        test_generator = RandomMidiDataGenerator(real_midis[1::2], seq_lstm.make_preprocessor(mapper), mapper, 2, 100)

        history = model.fit_generator(train_generator(), validation_data=test_generator(),
                                    steps_per_epoch=1, epochs=2, validation_steps=1, validation_freq=1, verbose=1)
        seq_lstm.plot_history(history)
        plt.savefig("test_seq_lstm_training.png")
        self.assertTrue(len(history.history["loss"])>0)
        
    def test_with_mel_lstm(self):
        """
        Test that a very short training runs without exceptions
        """
        model = mel_lstm.get_model()
        train_generator = RandomMidiDataGenerator(real_midis[::2], mel_lstm.preprocess, mapper, 2, 100)
        test_generator = RandomMidiDataGenerator(real_midis[1::2], mel_lstm.preprocess, mapper, 2, 100)
        test_data = test_generator.compute_batch()

        history = model.fit_generator(train_generator(), validation_data=test_data,
                                    steps_per_epoch=1, epochs=2, validation_freq=1, verbose=1) # validation_steps=1, 
        mel_lstm.plot_history(history)
        plt.savefig("test_mel_lstm_training.png")
        self.assertTrue(len(history.history["loss"])>0)
        
    
if __name__ == "__main__":
    unittest.main()