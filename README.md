# Reinforcement learning for generating music
Inspired by this article of deepmind, we want to try to train an agent to generate music. When humans create music, they neither generate pure waveforms or spectrograms, instead, they choose a couple of sounds or instruments and experiment on a higher level with them. Midi data is a good abstraction for this. We can try to mimic this process by having an agent generate the chords, and simultanously training a discriminator. The agents reward will be the log likelihood that  𝐷  predicts about being music.

## Setup
Make sure to have python and conda installed, then source the setup.sh script:
`source setup.sh`

Then start jupyter `jupyter notebook`