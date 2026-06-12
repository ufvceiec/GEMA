import numpy as np

from .map import Map


class IterativeSOM:
    def __init__(self,
                 data,
                 period,
                 initial_lr,
                 range_from=np.array([0, 0]),
                 try_best=False,
                 give_best=False):

        self.maps = {}
        if np.array_equal(range_from, np.array([0, 0])):
            range_from = IterativeSOM.calculate_range(data, range_from)

        for x in range_from:
            self.maps[x] = Map(data=data, size=x, period=period, initial_lr=initial_lr)

    @staticmethod
    def calculate_range(data, range_from):
        pass
