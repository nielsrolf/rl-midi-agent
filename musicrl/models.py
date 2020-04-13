import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Input,
    Dense,
    concatenate,
    LSTM,
    Reshape,
    BatchNormalization,
    Lambda,
    GaussianNoise,
    Flatten,
)

# import tensorflow.compat.v1 as tf

# tf.disable_v2_behavior()


class Actor:
    """ Actor Network for the DDPG Algorithm
    """

    def __init__(self, inp_dim, out_dim, act_range, lr, tau, batch_size, critic):
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.act_range = act_range
        self.tau = tau
        self.lr = lr
        self.batch_size = batch_size
        actor = self.network()
        self.target_model = self.network()
        inp = Input((self.env_dim))
        out = actor(inp)
        q_estimates = critic([inp, out])
        critic.trainable = False
        self.train_model = Model(inp, q_estimates)
        self.predict_model = Model(inp, out)

        def loss(dummy, q_estimates):
            return -K.mean(q_estimates)

        self.train_model.compile(Adam(lr), loss=loss)

    def network(self):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """

        inp = Input(batch_shape=(self.batch_size, self.env_dim))
        x = Reshape((1, self.env_dim))(inp)
        x = LSTM(128, stateful=True)(x)
        x = GaussianNoise(1.0)(x)
        #
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        x = GaussianNoise(1.0)(x)
        # NOTE: maybe exchange with linear layer without tanh
        out = Dense(
            self.act_dim, activation="tanh", kernel_initializer=RandomUniform()
        )(x)
        out = Lambda(lambda i: i * self.act_range)(out)
        #
        return Model(inp, out)

    def predict(self, states):
        """ Action prediction
        """
        return self.predict_model.predict(states, batch_size=self.batch_size)

    def target_predict(self, inp):
        """ Action prediction (target network)
        """
        return self.target_model.predict(inp, batch_size=self.batch_size)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.predict_model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau) * target_W[i]
        self.target_model.set_weights(target_W)

    def train(self, states):
        """ Actor Training
        """
        dummies = np.zeros((len(states), 1))
        return self.train_model.train_on_batch(states, dummies)

    def save(self, path):
        self.train_model.save_weights(path + "_actor.h5")

    def load_weights(self, path):
        self.train_model.load_weights(path)
        actual_tau = self.tau
        self.tau = 1
        self.transfer_weights()
        self.tau = actual_tau


class Critic:
    """ Critic for the DDPG Algorithm, Q-Value function approximator
    """

    def __init__(self, inp_dim, out_dim, lr, tau, batch_size):
        # Dimensions and Hyperparams
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.tau, self.lr = tau, lr
        self.batch_size = batch_size
        # Build models and target models
        self.model = self.network()
        self.target_model = self.network()
        self.model.compile(Adam(self.lr), "mse")
        self.target_model.compile(Adam(self.lr), "mse")

    def network(self):
        """ Assemble Critic network to predict q-values
        """
        state = Input((self.env_dim), name="state_input")
        action = Input(batch_shape=(self.batch_size, self.act_dim), name="action_input")
        x = Reshape((1, self.act_dim))(action)
        x = LSTM(128, stateful=True)(x)
        x = concatenate([Flatten()(x), state])
        x = Dense(128, activation="relu")(x)
        out = Dense(1, activation="linear", kernel_initializer=RandomUniform())(x)
        return Model([state, action], out)

    def target_predict(self, inp):
        """ Predict Q-Values using the target network
        """
        return self.target_model.predict(inp, batch_size=self.batch_size)

    def train_on_batch(self, states, actions, critic_target):
        """ Train the critic network on batch of sampled experience
        """
        return self.model.train_on_batch([states, actions], critic_target)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau) * target_W[i]
        self.target_model.set_weights(target_W)

    def save(self, path):
        self.model.save_weights(path + "_critic.h5")

    def load_weights(self, path):
        self.model.load_weights(path)
