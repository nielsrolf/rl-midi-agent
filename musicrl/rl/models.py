import sys
sys.path.append("../../")


class Actor():
    def __init__(self):
        pass

    def train(self, state, action):
        """
        :param state: list of seq
        :param action: seq
        :return:
        """
        pass

    def predict(self, state):
        """
        :param state: list of seq
        :return: seq
        """
        return



"""
generate a temporal-difference (TD) error signal each time step
"""
class Critic():
    def __int__(self):
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
        return 1