import numpy as np


def resemble_midi(real_midi, mapper):
    real_seq = mapper.midi2vec(real_midi)

    rand_seq = np.zeros(real_seq.shape)
    rand_seq[:,0] = np.where(np.random.random(shape=(len(real_seq)))>real_seq[:,0].mean(), 1, 0)
    rand_seq[rand_seq[:,0]>0.5,[1,2]] = np.random.multivariate_normal(
        real_seq[rand_seq[:,0]>0.5,[1,2]].mean(axis=0),
        np.diag(real_seq.std(axis=0)),
        size=int(rand_seq[:,0].sum()))
    return repair_generated_seq(rand_seq)


def repair_generated_seq(seq):
    seq = np.maximum(0, seq)
    seq[:,[1,2]] = np.clip(seq[:,[1,2]], 0, 127).astype(int)
    return seq
