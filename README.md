# Reinforcement learning for generating music
Inspired by this article of deepmind, we want to try to train an agent to generate music. When humans create music, they neither generate pure waveforms or spectrograms, instead, they choose a couple of sounds or instruments and experiment on a higher level with them. Midi data is a good abstraction for this. We can try to mimic this process by having an agent generate the chords, and simultanously training a discriminator. The agents reward will be the log likelihood that  𝐷  predicts about being music.

## Setup
Make sure to have python and conda installed, then source the setup.sh script:
`source setup.sh`

Then start jupyter `jupyter notebook`

## Running tests
Either run one of the test files in `musicrl/tests/` or `py.test musicrl`.

## Naming convention

Let's use a naming schema that works like this:
- `rand_midi`: random sequence of type PrettyMidi
- `rand_seq`: random sequence of vectorized midi
- `rand_wav`: random sequence as synthesized waveform
- `rand_mel`: mel spectrogram of rand_wav

And for collections, we use `rand_midis`, `rand_seqs`, `rand_wavs` and `rand_mels`.

For non generated sequences, let's use `real_...`.

For predictions on `something_mel`, we use `pred_something` and for labels, we use `y_something`.

## Developer notes

When this error occurs:
```
...
  File "/Users/nielswarncke/opt/anaconda3/envs/midi-rl/lib/python3.6/site-packages/pretty_midi/instrument.py", line 334, in synthesize
    fade_out = np.linspace(1, 0, .1*fs)
  File "<__array_function__ internals>", line 6, in linspace
  File "/Users/nielswarncke/opt/anaconda3/envs/midi-rl/lib/python3.6/site-packages/numpy/core/function_base.py", line 121, in linspace
    .format(type(num)))
TypeError: object of type <class 'float'> cannot be safely interpreted as an integer.
```

You have to modify `anaconda3/envs/midi-rl/lib/python3.6/site-packages/pretty_midi/instrument.py line 334` from `fade_out = np.linspace(1, 0, .1*fs)` to `fade_out = np.linspace(1, 0, fs//10)`