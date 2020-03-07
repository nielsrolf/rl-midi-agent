import numpy as np
import pretty_midi


class MidiVectorMapper():
    """Map a PrettyMIDI object to a sequence of vectors and back.
    For single instrument midi tracks only.
    Gets:
        - dataset: List of PrettyMIDI objects, to check for the categorical features, which features exist
    """
    def __init__(self, samples):
        """
        Dimensions:
            0: time
            1: is_note
            For notes only:
            2: pitch
            3: velocity
            4: duration
            For control changes only:
            5: value
            6-?: one hot encoding for control number
        """
        tmp = [[i.number for i in sample.instruments[0].control_changes] for sample in samples]
        sample_control_changes_number = [item for sublist in tmp for item in sublist]
        self.control_change_categories = sorted(set(sample_control_changes_number))
        self.dims = 6 + len(self.control_change_categories)
        self.column_meaning = [
            "time",
            "is_note",
            "note_pitch",
            "note_velocity",
            "note_duration",
            "control_change_value"
        ] + ["control_change_number"]*len(self.control_change_categories)
        
    def _timeof(self, event):
        """Return the start time for notes or the time for control change events
        """
        return event.start if isinstance(event, pretty_midi.Note) else event.time
    
    def midi2vec(self, sample):
        """Map a PrettyMIDI object to a sequence of vectors"""
        events = sorted(
            sample.instruments[0].notes +
            sample.instruments[0].control_changes,
            key=self._timeof
        )
        seq = np.zeros([len(events), self.dims])
        for i, event in enumerate(events):
            seq[i, 0] = self._timeof(event)
            if isinstance(event, pretty_midi.Note):
                seq[i, 1:5] = 1, event.pitch, event.velocity, event.end - event.start
            else:
                seq[i, 5] = event.value
                seq[i, 6+self.control_change_categories.index(event.number)] = 1
                
        return seq
    
    def vec2midi(self, seq):
        """Map a vector to a PrettyMIDI object with a single piano
        """
        song = pretty_midi.PrettyMIDI(resolution=384, initial_tempo=300)
        piano = pretty_midi.Instrument(program=0)
        for event_vec in seq:
            if event_vec[1] > 0.5:
                piano.notes.append(
                    pretty_midi.Note(
                        start=event_vec[0],
                        pitch=int(event_vec[2]),
                        velocity=int(event_vec[3]),
                        end=event_vec[0]+event_vec[4]
                    )
                )
            else:
                piano.control_changes.append(
                    pretty_midi.ControlChange(
                        time=event_vec[0],
                        value=int(event_vec[5]),
                        number=self.control_change_categories[np.argmax(event_vec[6:])]
                    )
                )
        song.instruments.append(piano)
        return song

    def action2note(self, event_vec):
        """Map a single action of the generstor to a note
        """
        if event_vec[1] > 0.5:
            return pretty_midi.Note(
                start=event_vec[0],
                pitch=int(event_vec[2]),
                velocity=int(event_vec[3]),
                end=event_vec[0]+event_vec[4]
            )
        else:
            return pretty_midi.ControlChange(
                time=event_vec[0],
                value=int(event_vec[5]),
                number=self.control_change_categories[np.argmax(event_vec[6:])]
            )

    
# mapper = MidiVectorMapper(samples)
# seq = mapper.midi2vec(samples[0])
# reconstruction = mapper.vec2midi(seq)
# listen_to(reconstruction)