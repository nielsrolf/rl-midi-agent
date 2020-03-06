import unittest
from glob import glob

import pretty_midi
import numpy as np

from musicrl.midi2vec import MidiVectorMapper
from musicrl.data import RandomMidiDataGenerator
from musicrl import mel_lstm, seq_lstm


filepaths = list(glob('maestro-v2.0.0/2008/**.midi'))[:5]
real_midis = [pretty_midi.PrettyMIDI(i) for i in filepaths]
mapper = MidiVectorMapper(real_midis)


class RandomMidiDataGeneratorTest(unittest.TestCase):
    def test_with_mel_lstm(self):
        generator = RandomMidiDataGenerator(real_midis, mel_lstm.preprocess, mapper, 2)

        got_a_batch = False
        for x, y in generator():
            got_a_batch = True
            self.assertFalse(np.any(np.isnan(x)))
            self.assertFalse(np.any(np.isnan(y)))
            if generator.epochs >= 1:
                break
        self.assertTrue(got_a_batch)

    def test_with_seq_lstm(self):
        generator = RandomMidiDataGenerator(real_midis, seq_lstm.make_preprocessor(mapper), mapper, 2)

        got_a_batch = False
        for x, y in generator():
            got_a_batch = True
            self.assertFalse(np.any(np.isnan(x)))
            self.assertFalse(np.any(np.isnan(y)))
            if generator.epochs >= 1:
                break
        self.assertTrue(got_a_batch)
        
    
if __name__ == "__main__":
    unittest.main()
    