import shelve
import shutil
import uuid
from enum import Enum
from pathlib import Path

from ceruleo import CACHE_PATH


class GraphTraversalAbstractStore:
    """Abstract Cache for the graph traversal
    """
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def keys(self):
        raise NotImplementedError

    def pop(self, k):
        raise NotImplementedError


class GraphTraversalCacheShelveStore:
    """Cache all the intermediate steps in a Shelve Store

    Parameters:

        cache_path: Path where the case is stored
    """
    def __init__(self, cache_path: Path = CACHE_PATH):
        filename = "".join(str(uuid.uuid4()).split("-"))
        self.cache_path = cache_path / "GraphTraversalCache" / filename / "data"
        self.cache_path.parent.mkdir(exist_ok=True, parents=True)
        self.transformed_cache = shelve.open(str(self.cache_path))

    def __getitem__(self, k):
        return self.transformed_cache[k]

    def __setitem__(self, k, v):
        self.transformed_cache[k] = v

    def close(self):
        if self.cache_path.parent.is_dir():
            self.transformed_cache.close()
            shutil.rmtree(self.cache_path.parent)

    def reset(self):
        self.transformed_cache.close()
        self.transformed_cache = shelve.open(str(self.cache_path))

    def keys(self):
        return self.transformed_cache.keys()

    def pop(self, k):
        return self.transformed_cache.pop(k)


class GraphTraversalCacheMemory:
    """Cache all the intermediate steps in RAM
    """
    def __init__(self):
        self.store = {}

    def __getitem__(self, k):
        return self.store[k]

    def __setitem__(self, k, v):
        self.store[k] = v

    def close(self):
        self.store = {}

    def reset(self):
        self.store = {}

    def keys(self):
        return self.store.keys()

    def pop(self, k):
        return self.store.pop(k)


class CacheStoreType(Enum):
    """Cache store modes
    
    Values:

        SHELVE = 1
        MEMORY = 2    
    """
    SHELVE = 1
    MEMORY = 2
