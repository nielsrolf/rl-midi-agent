import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, LSTM, Lambda, BatchNormalization, GaussianNoise, Flatten


import sys
import numpy as np
sys.path.append("../../")


class Actor():

    def dummy_network(self):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """
        inp = Input((self.env_dim))
        #
        x = Dense(256, activation='relu')(inp)
        x = GaussianNoise(1.0)(x)
        #
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = GaussianNoise(1.0)(x)
        #
        out = Dense(self.act_dim, activation='tanh', kernel_initializer=RandomUniform())(x)
        out = Lambda(lambda i: i * self.act_range)(out)
        #
        return Model(inp, out)



    def __init__(self, inp_dim, out_dim, act_range, lr, tau):
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.act_range = act_range
        self.tau = tau
        self.lr = lr
        self.model = self.network()
        self.target_model = self.network()
        self.adam_optimizer = self.optimizer()


    '''
    def train(self, state, action):
        """
        :param state: list of seq
        :param action: seq
        :return:
        """
        pass
    '''


    def train(self, states, actions, grads):
        """ Actor Training
        """
        self.adam_optimizer([states, grads])

    def predict(self, state):
        """
        :param state: list of seq
        :return: seq
        """
        action = np.ones(9)
        return action

    def optimizer(self):
        """ Actor Optimizer
        """
        action_gdts = K.placeholder(shape=(None, self.act_dim))
        params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.model.trainable_weights)
        return K.function([self.model.input, action_gdts], [tf.train.AdamOptimizer(self.lr).apply_gradients(grads)])


"""
generate a temporal-difference (TD) error signal each time step
"""
class Critic():

    def __init__(self):
        #self.action_grads = K.function([self.model.input[0], self.model.input[1]], K.gradients(self.model.output, [self.model.input[1]]))
        pass

    def train(self, state, action, q_value):
        """

        :param state: list of seq
        :param action: seq
        :param q_value:
        :return:
        """
        pass

    def predict(self, state, action):
        """
        :param state: list of seq
        :param action: seq
        :return: q value
        """
        return np.ones(state.shape[0])


    def train_on_batch(self, states, actions, critic_target):
        """ Train the critic network on batch of sampled experience
        """
        pass
        #return self.model.train_on_batch([states, actions], critic_target)


    def gradients(self, states, actions):
        """ Compute Q-value gradients w.r.t. states and policy-actions
        """
        pdb.set_trace()
        return np.ones( (states.shape[1], actions.shape[1]  ))
        #return self.action_grads([states, actions])