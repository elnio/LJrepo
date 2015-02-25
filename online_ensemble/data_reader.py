from itertools import izip
import numpy as np

__all__ = ["DataReader"]


class DataReader:
    def __init__(self,
                 num,
                 data_path,
                 target_path):
        self.num = num
        self.data_in = open(data_path)
        self.target_in = open(target_path)
        self.idx = 0

    def read_data_point(self):
        if self.idx >= self.num:
            raise ValueError('no more data point')
        x = np.fromstring(string=self.data_in.readline(), sep=',')
        y = np.array([float(self.target_in.readline())])
        self.idx += 1
        return {'x': x, 'y': y, 'idx': self.idx}

    def close(self):
        self.data_in.close()
        self.target_in.close()