import numpy as np


def generate_random_midi(seq, num_events=4000, seconds=300):
    """Generate a vectorized midi sequence that resembles a given sequence
    in certain statisics:
    - Ratio of notes to control change events is equal
    - Mean and standard deviation of pitch, velocity and duration is equal for notes, but
        approximated to be gaussian
    - Mean and standard deviation of value and the columns for number are equal,
        although this does not make incredibly much sense since the categorical
        features should be sampled such that the proportion of events is resembled
    Gets:
        seq: np.array; reference vectorized midi sequence
        num_events: int; number of events that will be created
        seconds: float; rough duration of the generated sequence
    
    """
    rand = np.zeros([num_events, seq.shape[1]])
    rand[:,0] = np.random.uniform(0, seconds, size=num_events) # start
    rand[:,1] = np.random.uniform(0,1,size=num_events) < seq[:,1].mean() # is_note
    # split notes and control change events
    rand_notes = rand[rand[:,1]==1] # select rows there is_note is true
    rand_cc = rand[rand[:,1]==0] # select rows there is_note is false
    seq_notes = seq[seq[:,1]==1] # select rows there is_note is true
    seq_cc = seq[seq[:,1]==0] # select rows there is_note is false
    # notes: pitch, velocity, duration/end
    rand_notes[:,2:5] = np.random.multivariate_normal(
        seq_notes[:,2:5].mean(axis=0),
        np.diag(seq_notes[:,2:5].std(axis=0)),
        size=len(rand_notes))
    rand_notes[:,4] = np.max(rand_notes[:,4], 0)
    # events: value, one hot encodings for number
    # it doesn't really make sense to use normal distributed values for one hot
    # encoding - it should be a distribution where p(rand_one_hot.argmax()) is distributed
    # like p(rand_one_hot.argmax()), but it doesn't really matter
    rand_cc[:,5:] = np.random.multivariate_normal(
        seq_cc[:,5:].mean(axis=0),
        np.diag(seq_cc[:,5:].std(axis=0)),
        size=len(rand_cc))
    # copy back
    rand[rand[:,1]==1] = rand_notes
    rand[rand[:,1]==0] = rand_cc
    # columns 2-5 are 7bit ints
    rand[:,[2,3,5]] = np.clip(rand[:,[2,3,5]], 0, 127).astype(int)
    # Done!
    # Done!
    return rand


def resemble_midi(real_midi, mapper):
    real_seq = mapper.midi2vec(real_midi)
    return generate_random_midi(real_seq, len(real_seq), real_midi.get_end_time())