from collections import deque
import numpy as np

from musicrl.models import *


class DDPG:
    """ Deep Deterministic Policy Gradient (DDPG) Helper Class
    """

    def __init__(
        self,
        act_dim,
        env_dim,
        act_range,
        batch_size,
        gamma=0.99,
        lr=0.00005,
        tau=0.001,
    ):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.act_range = act_range
        self.env_dim = env_dim
        self.gamma = gamma
        self.lr = lr
        # Create actor and critic networks
        self.critic = Critic(self.env_dim, act_dim, lr, tau, batch_size=batch_size)
        self.actor = Actor(
            self.env_dim,
            act_dim,
            act_range,
            0.1 * lr,
            tau,
            critic=self.critic.model,
            batch_size=batch_size,
        )

    def reset(self):
        """Resets the model states - should happen before a new episode starts
        """
        self.agent.actor.train_model.reset()
        self.critic.train_model.reset()

    def policy_action(self, s):
        """ Use the actor to predict value
        """
        return self.actor.predict(s)

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

    def update_models(self, states, actions, critic_target):
        """ Update actor and critic networks from sampled experience
        """
        # Train critic
        critic_loss = self.critic.train_on_batch(states, actions, critic_target)
        # Q-Value Gradients under Current Policy
        actions = self.actor.predict(states)
        # Train actor
        actor_loss = self.actor.train(states)
        # Transfer weights to target networks at rate Tau
        self.actor.transfer_weights()
        self.critic.transfer_weights()
        return critic_loss, actor_loss

    def save_weights(self, path):
        path += "_LR_{}".format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)
