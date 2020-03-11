import numpy as np
import pretty_midi


class MidiVectorMapper():
    """Map a PrettyMIDI object to a sequence of vectors and back.
    For single instrument midi tracks only.
    Gets:
        - dataset: List of PrettyMIDI objects, to check for the categorical features, which features exist
    """
    def __init__(self, samples, time_per_tick=0.00130208):
        self.time_per_tick = time_per_tick
        self.dims = 5
        self.column_meaning = [
            "is_note",
            "note_pitch",
            "note_velocity",
            "note_duration",
            "is_end"
        ]
        self.no_sound = np.zeros(5)
    
    def vec2midi(self, seq):
        """Map a vector to a PrettyMIDI object with a single piano
        """
        song = pretty_midi.PrettyMIDI(resolution=384, initial_tempo=300)
        piano = pretty_midi.Instrument(program=0)
        for i, event_vec in enumerate(seq):
            if event_vec[4] > 0.5: # end token
                break
            if event_vec[0] > 0.5: # note
                start = i*self.time_per_tick
                piano.notes.append(
                    self.action2note(event_vec, start)
                )
            else: # no_sound placeholder
               continue
        song.instruments.append(piano)
        return song

    def midi2vec(self, midi):
        notes = sorted(midi.instruments[0].notes, key=lambda note: note.start)
        seq = []
        for note in notes:
            position = int(note.start/self.time_per_tick)
            while len(seq) < position:
                seq.append(self.no_sound)
            seq.append(np.array([1, note.pitch, note.velocity, note.end-note.start, 0]))
        return np.array(seq)

    def action2note(self, event_vec, start):
        """Map a single action of the generstor to a note
        """
        if event_vec[0] > 0.5 and event_vec[4] < 0.5:
            return pretty_midi.Note(
                start=start,
                pitch=int(event_vec[1]),
                velocity=int(event_vec[2]),
                end=start+event_vec[3]
            )
        else:
            return None


class PostProcessor():
    """Postprocess to make a generating function that produces values
    around 0 makes some noise.
    If the generators produce only silence in the initial state, there won't be
    any useful gradients
    
    Gets:
        real_seqs: np.array, vectorized sequences of which to collect statistics
    """
    def __init__(self, real_seqs):
        real_seqs = np.concatenate(real_seqs)
        self.offset_0 = 0.4
        self.mean_notes = real_seqs[(real_seqs[:,0]>0.5)].mean(axis=0, keepdims=True)[:,[1,2,3]]
        
    def __call__(self, pred):
        pred = np.array(pred)
        pred[:,0] += self.offset_0
        pred[:,[1,2,3]] += self.mean_notes
        return repair_generated_seq(pred)


def generate_random_sequence(postprocess, num_events=15000):
    """Generate some Gaussian noise, remove end tokens and postprocess it
    to something playable.

    Gets:
        postprocess: function/callable
            np.array predictions -> np.array of hopefully playable sequence
        num_events: number of actions in the random sequence,
            linear relationship with expected wav lemngth

    Returns:
        rand_seq: np.array, input for a corresponding MidiVectorMapper.vec2midi
    """
    noise = np.random.normal(0.2, 1, size=(num_events, 5))
    noise[:,4] = 0 # remove any end token
    rand_seq = postprocess(noise)
    return rand_seq

   
# mapper = MidiVectorMapper(samples)
# seq = mapper.midi2vec(samples[0])
# reconstruction = mapper.vec2midi(seq)
# listen_to(reconstruction)