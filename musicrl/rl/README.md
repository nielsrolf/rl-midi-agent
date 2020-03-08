#Reinforcement Learning
The RL-Midi-Agent implements a Deep Deterministic Policy Gradient method:

* [Deep Deterministic Policy Gradients Explained](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b)
* [GitHub Repository: germain-hug/Deep-RL-Keras](https://github.com/germain-hug/Deep-RL-Keras)

The following components are needed:
* Q Network (Critic): calculates estimated Q-value of the current state and of the action given by the actor
* policy network (Actor): maps state (sequence of seq) to actions (seq)

* target Q Network: 
* target policy network



TODO:
* sample randomly from experience replay to avoid temporally-correlated simulated trajectories and to reduce variance 

