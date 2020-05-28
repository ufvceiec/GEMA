import numpy as np

from .map import Map


class IterativeSOM:
    @staticmethod
    def __init__(data,
                 period,
                 initial_lr,
                 range_from=np.array([0, 0]),
                 try_best=False,
                 give_best=False):

        if range_from == np.array([0, 0]):
            range_from = IterativeSOM.calculate_range(data, range_from)

        for x in range_from:
            map[x] = Map.train(data, x, period, initial_lr)

    def calculate_range(data, range_from):
        pass
