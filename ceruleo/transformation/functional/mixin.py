import hashlib
import uuid
from typing import List, Optional, Union

from numpy.lib.arraysetops import isin
from sklearn.base import BaseEstimator


class TransformerStepMixin(BaseEstimator):
    def __init__(self, *, name: Optional[str] = None, prefer_partial_fit:bool =True):
        self.name_ = name
        self.previous = []
        self.next = []
        self.uuid = "".join(str(uuid.uuid4()).split("-"))
        self.prefer_partial_fit = prefer_partial_fit


    @property
    def name(self):
        if self.name_ is not None:
            return self.name_
        else:
            return self.__class__.__name__

    @name.setter
    def name(self, value: str):
        self.name_ = value

    def __call__(
        self, prev: Union[List["TransformerStepMixin"], "TransformerStepMixin"]
    ):
        self._add_previous(prev)
        return self

    def _add_previous(
        self, prev: Union[List["TransformerStepMixin"], "TransformerStepMixin"]
    ):
        if not isinstance(prev, list):
            prev = [prev]
        for p in prev:
            p.add_next(self)

    def remove_previous(self, node):
        self.previous.remove(node)

    def disconnect(self, node):
        if node in self.previous:
            self.previous.remove(node)
            node.disconnect(self)
        elif node in self.next:
            self.next.remove(node)
            node.disconnect(self)

    def add_next(
        self, nodes: Union[List["TransformerStepMixin"], "TransformerStepMixin"]
    ):
        if not isinstance(nodes, list):
            nodes = [nodes]
        for n in nodes:
            self.next.append(n)
            n.previous.append(self)

    def description(self):
        return self.name

    def __str__(self):
        return self.name
