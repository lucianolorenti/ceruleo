import numpy as np


class LRUDataCache:
    def __init__(self, max_elem):
        self.max_elem = max_elem
        self.data = {}

    def get(self, key):
        self.data[key]['hit'] += 1

        return self.data[key]['elem']

    def add(self, key, elem):
        if len(self.data) == self.max_elem:
            keys = list(self.data.keys())
            key_to_remove = np.argmin([self.data[k]['hit'] for k in keys])
            del self.data[keys[key_to_remove]]
        self.data[key] = {
            'elem':  elem,
            'hit': 0
        }
        return elem

    def __len__(self):
        return len(self.data)
