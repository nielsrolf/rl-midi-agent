"""
Data generators for training and validation data
"""
import numpy as np
from random import shuffle
from musicrl.render import midi2mel
from musicrl.midi2vec import PostProcessor, generate_random_sequence


REAL = 1
GEN = 0


class RandomSeqGenerator:
    """
    Take some real data and build a generator that can be used to train
    a discriminator on real data and random midi data generated by random sampling in the
    seq space.
    When creating batches, it cuts sequences to the shortest length of the batch.
    We could also pad it to the longest sequence, but it doesn't matter at this point.
    Gets:
        real_midis: List[pretty_midi.PrettyMidi]
        mapper: midi2vec.MidiVectorMapper
        batch_size: int
        max_num_timeframes: each batch is cut to at most this many timeframes. This can avoid memory issues,
            which result in an error like this
            `[1]    2097 bus error  /Users/nielswarncke/opt/anaconda3/envs/midi-rl/bin/python  --default --client`
    """

    def __init__(
        self, real_midis, mapper, batch_size, max_num_timeframes=8000
    ):
        self.real_midis = real_midis
        self.batch_size = batch_size
        self.mapper = mapper
        self.max_num_timeframes = max_num_timeframes
        self.idx = 0
        self.epochs = 0
        self.steps_per_epoch = len(self.real_midis) // (batch_size // 2)
        real_seqs = [mapper.midi2vec(i) for i in real_midis]
        self.generate_postprocess = PostProcessor(real_seqs)

    def compute_batch_of_reals(self, batch_size):
        real_midis = np.random.choice(self.real_midis, size=batch_size, replace=False)
        real_seqs = [self.mapper.midi2vec(midi) for midi in real_midis]
        min_length = min([mel.shape[0] for mel in real_seqs])
        min_length = min(min_length, self.max_num_timeframes)
        x = np.array([mel[:min_length] for mel in real_seqs])
        y = np.array([REAL] * (self.batch_size))
        return x, y

    def compute_batch_of_generated(self, batch_size):
        fake_seqs = [
            generate_random_sequence(self.generate_postprocess, self.max_num_timeframes)
            for _ in range(batch_size)
        ]
        min_length = min([mel.shape[0] for mel in fake_seqs])
        min_length = min(min_length, self.max_num_timeframes)
        x = np.array([mel[:min_length] for mel in fake_seqs])
        y = np.array([GEN] * (self.batch_size)).reshape((-1, 1))
        return x, y

    def compute_batch(self):
        if self.idx + self.batch_size // 2 >= len(self.real_midis):
            self.idx = 0
            shuffle(self.real_midis)
            self.epochs += 1
        real_midis = self.real_midis[self.idx : self.idx + self.batch_size // 2]
        real_seqs = [self.mapper.midi2vec(midi) for midi in real_midis]
        fake_seqs = [
            generate_random_sequence(self.generate_postprocess, self.max_num_timeframes)
            for _ in range(self.batch_size // 2)
        ]
        min_length = min([seq.shape[0] for seq in real_seqs + fake_seqs])
        min_length = min(min_length, self.max_num_timeframes)
        x = np.array(
            [seq[:min_length] for seq in real_seqs + fake_seqs]
        )
        y = np.array(
            [REAL] * (self.batch_size // 2) + [GEN] * (self.batch_size // 2)
        ).reshape((-1, 1))
        self.idx += self.batch_size // 2
        return x, y

    def __call__(self):
        while True:
            yield self.compute_batch()


class RandomMidiDataGenerator(RandomSeqGenerator):
    """
    Take some real data and build a generator that can be used to train
    a discriminator on real data and ranom midi data generated by random sampling
    in seq space and applying some postprocessing to it.
    When creating batches, it cuts sequences to the shortest length of the batch.
    We could also pad it to the longest sequence, but it doesn't matter at this point.
    Gets:
        real_midis: List[pretty_midi.PrettyMidi]
        preprocess: (pretty_midi.PrettyMidi) -> model input
        mapper: midi2vec.MidiVectorMapper which shall be used by resemble_midi
        batch_size: int
        max_num_timeframes: each batch is cut to at most this many timeframes. This can avoid memory issues,
            which result in an error like this
            `[1]    2097 bus error  /Users/nielswarncke/opt/anaconda3/envs/midi-rl/bin/python  --default --client`
    """

    def __init__(
        self, real_midis, preprocess, mapper, batch_size, max_num_timeframes=8000
    ):
        super().__init__(real_midis, mapper, batch_size, max_num_timeframes)
        self._preprocess = preprocess

    def _preprocess_batch_of_seqs(self, seqs):
        """Takes a list of seqs, applies the preprocessing and cuts them to a uniform
        length

        Gets:
            seqs: List[np.array: seq corresponding to self.mapper]
        
        Retruns:
            x: np.array
        """
        midis = [self.mapper.vec2midi(seq) for seq in seqs]
        x = [self._preprocess(midi) for midi in midis]
        t = min(*[len(i) for i in x])
        x = np.array([i[:t] for i in x])
        return x

    def compute_batch_of_reals(self, batch_size):
        """Compute a batch of real samples, without changing the index counter
        Not intended for use in training

        Gets:
            batch_size: int
        Returns:
            x: np.array of shape (batch_size, *self.preprocess().shape)
            y: np.array of (batch_size, 1)
        """
        seqs, y = super().compute_batch_of_reals(batch_size)
        x = self._preprocess_batch_of_seqs(seqs)
        return x, y

    def compute_batch_of_generated(self, batch_size):
        """Compute a batch of generated samples, without changing the index counter
        Not intended for use in training

        Gets:
            batch_size: int
        Returns:
            x: np.array of shape (batch_size, *self.preprocess().shape)
            y: np.array of (batch_size, 1)
        """
        seqs, y = super().compute_batch_of_generated(batch_size)
        x = self._preprocess_batch_of_seqs(seqs)
        return x, y

    def compute_batch(self):
        seqs, y = super().compute_batch()
        x = self._preprocess_batch_of_seqs(seqs)
        return x, y

