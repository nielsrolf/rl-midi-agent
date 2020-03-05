# Reinforcement learning for generating music
Inspired by this article of deepmind, we want to try to train an agent to generate music. When humans create music, they neither generate pure waveforms or spectrograms, instead, they choose a couple of sounds or instruments and experiment on a higher level with them. Midi data is a good abstraction for this. We can try to mimic this process by having an agent generate the chords, and simultanously training a discriminator. The agents reward will be the log likelihood that  𝐷  predicts about being music.

## Setup
Make sure to have python and conda installed, then source the setup.sh script:
`source setup.sh`

Then start jupyter `jupyter notebook`

## Naming convention

Let's use a naming schema that works like this:
- `rand_midi`: random sequence of type PrettyMidi
- `rand_seq`: random sequence of vectorized midi
- `rand_wav`: random sequence as synthesized waveform
- `rand_mel`: mel spectrogram of rand_wav

And for collections, we use `rand_midis`, `rand_seqs`, `rand_wavs` and `rand_mels`.

For non generated sequences, let's use `real_...`.

For predictions on `something_mel`, we use `pred_something` and for labels, we use `y_something`.