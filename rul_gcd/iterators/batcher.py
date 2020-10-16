import math
import numpy as np


class Batcher:
    def __init__(self, iterator, batch_size, restart_at_end=True):
        self.iterator = iterator
        self.batch_size = batch_size
        self.restart_at_end = restart_at_end
        self.stop = False  
    
    def __len__(self):
        return math.ceil(len(self.iterator) / self.batch_size)

    def __iter__(self):
        self.iterator.__iter__()
        return self 

    def __next__(self):
        X = []
        y = []  
        if self.stop:
            raise StopIteration
        if self.iterator.at_end():
            if self.restart_at_end:
                self.__iter__()         
            else:
                raise StopIteration
        try:
            for _ in range(self.batch_size):            
                X_t, y_t = next(self.iterator)
                X.append(np.expand_dims(X_t,axis=0))
                y.append(y_t)
        except StopIteration:
            pass
        X = np.concatenate(X,axis=0)
        y = np.array(y)
        return X.astype(np.float32), y.astype(np.float32)