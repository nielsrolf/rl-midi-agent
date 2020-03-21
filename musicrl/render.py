from IPython.display import display, Audio
import soundfile as sf
import librosa
import librosa.display
import pretty_midi
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
    mel = librosa.feature.melspectrogram(
        y=wav, sr=rate, hop_length=512, n_mels=128, n_fft=2048
    )
    return librosa.power_to_db(mel, ref=np.max)


def midi2mel(sample_midi):
    return wav2mel(*midi2wav(sample_midi))


def plot_spectro(sample_mel, title):
    plt.figure(figsize=(25, 8))
    plt.title(title)
    return plt.imshow(sample_mel[:, ::20])


def plot_piano_roll(pm, start_pitch=56, end_pitch=70, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(
        pm.get_piano_roll(fs)[start_pitch:end_pitch],
        hop_length=1,
        sr=fs,
        x_axis="time",
        y_axis="cqt_note",
        fmin=pretty_midi.note_number_to_hz(start_pitch),
    )


def plot_predictions_over_time(model, reals, generateds):
    plt.figure(figsize=(18, 9))

    plt.subplot(121, ylim=(0, 1))
    pred_reals = model.predict(reals)[:2340, :, 0]
    for pred_real in pred_reals:
        plt.plot(pred_real, color="blue", linewidth=0.5, alpha=0.5)

    pred_rands = model.predict(generateds)[:2340, :, 0]
    for pred_rand in pred_rands:
        plt.plot(pred_rand, color="green", linewidth=0.5, alpha=0.5)

    plt.plot(0 * pred_real + 0.5, linewidth=3)
    plt.legend()

    plt.subplot(122, ylim=(0, 1))
    plt.hist(
        pred_reals.mean(0),
        label="Real",
        color="blue",
        orientation="horizontal",
        alpha=0.5,
    )
    plt.hist(
        pred_rands.mean(0),
        label="Generated",
        color="green",
        orientation="horizontal",
        alpha=0.5,
    )
    plt.legend()
    plt.show()
