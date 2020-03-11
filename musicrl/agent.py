from collections import deque
import numpy as np

from musicrl.models import *
from musicrl.midi2vec import MidiVectorMapper

from musicrl.utils.memory_buffer import MemoryBuffer

class DPPGAgent:
    def __init__(self, env, act_dim, max_memory_size=50000):
        buffer_size = 20000
        gamma =  0.9
        self.act_dim = act_dim


        #Init Networks
        self.critic = Critic()

        inp_dim =
        out_dim =
        act_range =
        lr = 0.001
        tau = 0.02
        self.actor = Actor(act_dim,inp_dim, out_dim, act_range, lr, tau)
        self.actor_target = Actor(act_dim,inp_dim, out_dim, act_range, lr, tau)

            self.critic_target = Critic()
        self.buffer = MemoryBuffer(buffer_size)
        self.gamma = gamma



        self.memory = Memory(max_memory_size)

    def get_action(self, state):
        action = self.actor.predict(state)
        return action


    def update_models(self, states, actions, critic_target):
        """ Update actor and critic networks from sampled experience
        """
        # Train critic
        self.critic.train_on_batch(states, actions, critic_target)
        # Q-Value Gradients under Current Policy
        actions = self.actor.predict(states)
        grads = self.critic.gradients(states, actions)
        # Train actor
        import pdb
        pdb.set_trace()
        self.actor.train(states, actions, np.array(grads).reshape((-1, self.act_dim)))
        # Transfer weights to target networks at rate Tau
        self.actor.transfer_weights()
        self.critic.transfer_weights()

    def policy_action(self, s):
        """ Use the actor to predict value
        """
        return self.actor.predict(s) #[0] ?

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """
        self.buffer.memorize(state, action, reward, done, new_state)

    def sample_batch(self, batch_size):
        return self.buffer.sample_batch(batch_size)


    def bellman(self, rewards, q_values, dones):
        """ Use the Bellman Equation to compute the critic target
        """
        critic_target = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                critic_target[i] = rewards[i]
            else:
                critic_target[i] = rewards[i] + self.gamma * q_values[i]
        return critic_target



class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)