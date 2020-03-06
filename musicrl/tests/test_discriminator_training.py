import unittest
from glob import glob

import pretty_midi
import numpy as np
from matplotlib import pyplot as plt

from musicrl.midi2vec import MidiVectorMapper
from musicrl.data import RandomMidiDataGenerator
from musicrl import seq_lstm, mel_lstm


filepaths = list(glob('maestro-v2.0.0/2008/**.midi'))[:5]
real_midis = [pretty_midi.PrettyMIDI(i) for i in filepaths]
mapper = MidiVectorMapper(real_midis)


class DiscriminatorTrainingWithGeneratorsTest(unittest.TestCase):
    def train(self, model, train_generator, test_generator):
        history = model.fit_generator(train_generator(), validation_data=test_generator(),
                                    steps_per_epoch=1, epochs=2, validation_steps=1, validation_freq=1, verbose=1)
        seq_lstm.plot_history(history)
        self.assertTrue(len(history.history["loss"])>0)

    def test_with_seq_lstm(self):
        """
        Test that a very short training runs without exceptions
        """
        model = seq_lstm.get_model()
        train_generator = RandomMidiDataGenerator(real_midis[::2], seq_lstm.make_preprocessor(mapper), mapper, 2, 100)
        test_generator = RandomMidiDataGenerator(real_midis[1::2], seq_lstm.make_preprocessor(mapper), mapper, 2, 100)
        self.train(model, train_generator, test_generator)
        plt.title("Seq LSTM, on generator - Accuracy")
        plt.savefig("test_seq_lstm_training_generator.png")
        
#     def test_with_mel_lstm(self):
#         """
#         Test that a very short training runs without exceptions
#         """
#         model = mel_lstm.get_model()
#         train_generator = RandomMidiDataGenerator(real_midis[::2], mel_lstm.preprocess, mapper, 2, 100)
#         test_generator = RandomMidiDataGenerator(real_midis[1::2], mel_lstm.preprocess, mapper, 2, 100)
#         test_data = test_generator.compute_batch()
#         self.train(model, train_generator, test_generator)
#         plt.title("Mel LSTM, on generator - Accuracy")
#         plt.savefig("test_seq_lstm_training_generator.png")

class DiscriminatorTrainingWithPrecomputedData(unittest.TestCase):
    def train(self, model, train_generator, test_generator):
        x_train, y_train = train_generator.compute_batch()
        test_data = test_generator.compute_batch()

        history = model.fit(x_train, y_train, 2, validation_data=test_data, epochs=10, validation_freq=1, verbose=1)
        seq_lstm.plot_history(history)
        self.assertTrue(len(history.history["loss"])>0)

    def test_with_seq_lstm(self):
        """
        Test that a very short training runs without exceptions
        """
        model = seq_lstm.get_model()
        train_generator = RandomMidiDataGenerator(real_midis[::2], seq_lstm.make_preprocessor(mapper), mapper, 2, 100)
        test_generator = RandomMidiDataGenerator(real_midis[1::2], seq_lstm.make_preprocessor(mapper), mapper, 2, 100)
        self.train(model, train_generator, test_generator)
        plt.title("Seq LSTM, precomputed - Accuracy")
        plt.savefig("test_seq_lstm_training_precomputed.png")
        
    def test_with_mel_lstm(self):
        """
        Test that a very short training runs without exceptions
        """
        model = mel_lstm.get_model()
        train_generator = RandomMidiDataGenerator(real_midis[::2], mel_lstm.preprocess, mapper, 2, 100)
        test_generator = RandomMidiDataGenerator(real_midis[1::2], mel_lstm.preprocess, mapper, 2, 100)
        self.train(model, train_generator, test_generator)
        plt.title("Mel LSTM, precomputed - Accuracy")
        plt.savefig("test_mel_lstm_training_precomputed.png")

    
if __name__ == "__main__":
    unittest.main()