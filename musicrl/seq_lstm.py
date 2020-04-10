"""
This model is currently not really used besides in the discriminator.ipynb
for evaluation.
"""


from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras import metrics

from musicrl.render import midi2mel


def get_model(input_dims):
    model = Sequential()
    model.add(LSTM(128, input_shape=(None, input_dims), return_sequences=True))  #
    model.add(TimeDistributed(Dense(128, activation="sigmoid")))
    model.add(TimeDistributed(Dense(1, activation="sigmoid")))
    model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=[metrics.binary_accuracy]
    )
    return model


"""
train like this:
history = model.fit_generator(train_generator(), validation_data=test_generator(),
                              steps_per_epoch=1, epochs=10, validation_steps=1, validation_freq=1)
"""


def make_preprocessor(mapper):
    """
    Gets:
        mapper: musicrl.midi2vec.MidiVectorMapper
    Returns:
        preprocess: (PrettyMidi) -> np.array of shape (#time, mapper.dims)
    """

    def preprocess(midi):
        """
        Gets:
            pretty_midi.PrettyMidi
        Returns:
            np.array of shape (#time, #frequencies) of mel spectrograms
        """
        return mapper.midi2vec(midi)

    return preprocess


def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.title("Loss")
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Test")
    plt.legend()
    plt.subplot(122)
    plt.title("Accuracy")
    plt.plot(history.history["val_binary_accuracy"], label="Train")
    plt.plot(history.history["binary_accuracy"], label="Test")
    plt.legend()
