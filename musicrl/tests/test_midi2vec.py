import unittest
from glob import glob

import numpy as np
import pretty_midi

from musicrl.midi2vec import MidiVectorMapper


class MidiVectorMapperTest(unittest.TestCase):
    """
    Test if it is possible to convert a midi to a sequence and back to midi
    without changing how it sounds when synthezising.
    """
    def test_midi2vec2midi(self):
        filepaths = list(glob('maestro-v2.0.0/2008/**.midi'))
        real_midis = [pretty_midi.PrettyMIDI(filepaths[1])]
        real_midi = real_midis[0]
        mapper = MidiVectorMapper(real_midis)
        real_seq = mapper.midi2vec(real_midi)
        reconstruction_midi = mapper.vec2midi(real_seq)
        real_wav = real_midi.synthesize(100)
        reconstruction_wav = reconstruction_midi.synthesize(100)
        self.assertTrue(np.allclose(real_wav, reconstruction_wav))

    
if __name__ == "__main__":
    unittest.main()
        