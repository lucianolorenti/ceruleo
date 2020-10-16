import unittest
from rul_gcd.utils.lrucache import LRUDataCache


class LRUCache(unittest.TestCase):

    def test_limit_size(self):
        cache = LRUDataCache(3)
        cache.add('A', 5)
        cache.add('B', 6)
        cache.add('C', 7)
        cache.add('D', 5)
        self.assertTrue(len(cache) == 3)

    def test_lru(self):
        cache = LRUDataCache(3)
        cache.add('A', 5)
        cache.add('B', 6)
        cache.add('C', 7)
        self.assertTrue(len(cache) == 3)
        cache.get('A')
        cache.get('A')
        cache.get('A')
        cache.get('B')
        self.assertTrue(cache.data['A']['hit'] == 3)
        self.assertTrue(cache.data['B']['hit'] == 1)
        cache.add('D', 8)
        self.assertTrue(len(cache) == 3)
        self.assertTrue('D' in cache.data)
        self.assertTrue('C' not in cache.data)
        cache.get('D')
        cache.get('D')
        cache.add('E', 9)        
        self.assertTrue('B' not in cache.data)
