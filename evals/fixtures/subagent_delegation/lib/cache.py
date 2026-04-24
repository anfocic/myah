"""An LRU cache implementation backed by `collections.OrderedDict`.
Used by the request pipeline to memoize expensive DB lookups."""
from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity: int = 128):
        self._store: OrderedDict = OrderedDict()
        self._cap = capacity

    def get(self, key):
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key, value):
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self._cap:
            self._store.popitem(last=False)
