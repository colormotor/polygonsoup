#!/usr/bin/env python3

import numpy as np
from collections import defaultdict, namedtuple

class DoublyLinkedList:
    class Node:
        ''' Generic linked list node
        Either define a class with prev, next or use this and populate'''
        def __init__(self):
            self.prev = None
            self.next = None

    def __init__(self):
        self.clear()

    def clear(self):
        self.front = None
        self.back = None

    def add(self, elem, at=None):
        elem.prev = None
        elem.next = None
        if self.back is None: # Empty
            self.back = self.front = elem
            return
        if at is None:
            at = self.back
            self.back = elem
            elem.next = None
        else:
            elem.next = at.next

        elem.prev = at
        at.next = elem

    def insert(self, elem, at=None):
        elem.prev = None
        elem.next = None
        if self.back is None: # Empty
            self.back = self.front = elem
            return
        if at is None:
            at = self.front
            self.back = elem
            elem.prev = None
        else:
            elem.prev = at.prev

        elem.next = at
        at.prev = elem

    def size(self):
        l = 0
        e = self.front
        while e:
            e = e.next
            l += 1
        return l

    def nodes(self):
        e = self.front
        while e:
            yield e
            e = e.next

    def pop_front(self):
        if self.front is not None:
            res = self.front
            if self.front == self.back:
                self.clear()
            else:
                self.front.next.prev = None
                self.front = self.front.next

            res.prev = res.next = None
            return res
        else:
            return None

    def pop_back(self):
        if self.back is not None:
            res = self.back
            if self.front == self.back:  # no more elements left
                self.clear()
            else:
                self.back.prev.next = None
                self.back = self.back.prev
            res.prev = res.next = 0
            return res
        else:
            return None

    def remove(self, elem):
        if elem == self.front:
            return self.pop_front()
        if elem == self.back:
            return self.pop_back()
        # should have prev and next since it isnt front or back
        if elem.next:
            elem.next.prev = elem.prev

        if elem.prev:
            elem.prev.next = elem.next

        elem.prev = elem.next = 0
        return elem


class UnionFind:
    """Union-find data structure.

    From https://www.ics.uci.edu/~eppstein/PADS/UnionFind.py
    Union-find data structure. Based on Josiah Carlson's code,
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/215912
    with significant additional changes by D. Eppstein.

    Each unionFind instance X maintains a family of disjoint sets of
    hashable objects, supporting the following two methods:

    - X[item] returns a name for the set containing the given item.
      Each set is named by an arbitrarily-chosen one of its members; as
      long as the set remains unchanged it will keep the same name. If
      the item is not yet part of a set in X, a new singleton set is
      created for it.

    - X.union(item1, item2, ...) merges the sets containing each item
      into a single larger set.  If any item is not yet part of a set
      in X, it is added to X as one of the members of the merged set.

    Example usage:

    uf = UnionFind()
    for loop i, j:
        uf[i]
        uf[j]

    groups = {}
    for i in uf:
        set_id = uf[i]
        if not set_id in groups:
            groups[set_id] = []
        groups[set_id].append(i)

    """

    def __init__(self):
        """Create a new empty union-find structure."""
        self.weights = {}
        self.parents = {}

    def __getitem__(self, object):
        """Find and return the name of the set containing the object."""

        # check for previously unknown object
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        """Iterate through all items ever found or unioned by this structure."""
        return iter(self.parents)

    def union(self, *objects):
        """Find the sets containing the objects and merge them all."""
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r],r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest

    def get_sets(self):
        """Return a dict with all sets"""
        sets = defaultdict(list)
        for item in iter(self.parents):
            sets[self.__getitem__(item)].append(item)
        return sets
