
from ceruleo.utils.lrucache import LRUDataCache


class TestLRUCache():

    def test_limit_size(self):
        cache = LRUDataCache(3)
        cache.add('A', 5)
        cache.add('B', 6)
        cache.add('C', 7)
        cache.add('D', 5)
        assert len(cache) == 3

    def test_lru(self):
        cache = LRUDataCache(3)
        cache.add('A', 5)
        cache.add('B', 6)
        cache.add('C', 7)
        assert len(cache) == 3
        cache.get('A')
        cache.get('A')
        cache.get('A')
        cache.get('B')
        assert cache.data['A']['hit'] == 3
        assert cache.data['B']['hit'] == 1
        cache.add('D', 8)
        assert len(cache) == 3
        assert 'D' in cache.data
        assert 'C' not in cache.data
        cache.get('D')
        cache.get('D')
        cache.add('E', 9)
        assert 'B' not in cache.data
