from IPython.display import display, Audio
import soundfile as sf
import librosa
import numpy as np
from matplotlib import pyplot as plt


def midi2wav(sample_midi):
    """Generate an in-memory wav file from a PrettyMidi object
    Gets:
        sample: PrettMidi object
    Returns:
        data: np.array with 1 dimension, waveform
        rate: int, sample rate
    """
    return sample_midi.synthesize(fs=44100), 44100

def listen_to(sample_midi):
    """Create a audio player that renders a PrettyMidi object"""
    sample_wav, rate = midi2wav(sample_midi)
    display(Audio(data=sample_wav, rate=rate))
    
def save_as_wav(sample_midi, filename):
    sample_wav, rate = midi2wav(sample_midi)
    sf.write(filename, sample_wav, rate)

def wav2mel(wav, rate):
    mel = librosa.feature.melspectrogram(y=wav, sr=rate, hop_length=512, n_mels=128, n_fft=2048)
    return librosa.power_to_db(mel, ref=np.max)

def midi2mel(sample_midi):
    return wav2mel(*midi2wav(sample_midi))

def plot_spectro(sample_mel, title):
    plt.figure(figsize=(25, 8))
    plt.title(title)
    return plt.imshow(sample_mel[:,::20])