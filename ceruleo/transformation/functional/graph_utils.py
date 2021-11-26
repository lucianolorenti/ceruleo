from copy import copy
from typing import Union


def root_nodes(step_or_pipe: Union["TemporisPipeline", "TransformerStep"]):
    from ceruleo.transformation.functional.pipeline.pipeline import Pipeline

    if isinstance(step_or_pipe, Pipeline):
        final_step = step_or_pipe.final_step
    else:
        final_step = step_or_pipe
    final_step = final_step
    visited = set([final_step])
    to_process = copy(final_step.previous)

    while len(to_process) > 0:
        t = to_process.pop()
        if t not in visited:
            visited.add(t)
            to_process.extend(t.previous)

    return [n for n in visited if len(n.previous) == 0]


def dfs_iterator(step_or_pipe: Union["TemporisPipeline", "TransformerStep"]):

    from ceruleo.transformation.functional.pipeline.pipeline import Pipeline

    if isinstance(step_or_pipe, Pipeline):
        final_step = step_or_pipe.final_step
    else:
        final_step = step_or_pipe
    final_step = final_step
    visited = set([])
    Q = copy(root_nodes(final_step))
    while len(Q) > 0:
        node = Q.pop()
        if node in visited:
            continue
        visited.add(node)

        yield node

        if len(node.next) == 0:
            continue
        for n in node.next:
            Q.append(n)


class topological_sort_iterator:
    def __init__(self, step_or_pipe: Union["TemporisPipeline", "TransformerStep"]):
        from ceruleo.transformation.functional.pipeline.pipeline import Pipeline

        if isinstance(step_or_pipe, Pipeline):
            final_step = step_or_pipe.final_step
        else:
            final_step = step_or_pipe
        self.final_step = final_step
        self.in_degree = {}

    def _initialize_degrees(self):
        for node in dfs_iterator(self.final_step):
            if node not in self.in_degree:
                self.in_degree[node] = 0
            if len(node.next) == 0:
                continue
            for n in node.next:
                if n not in self.in_degree:
                    self.in_degree[n] = 0
                self.in_degree[n] += 1

    def __iter__(self):
        self.Q = root_nodes(self.final_step)
        self._initialize_degrees()
        self.current_node = None
        self.traversed = []
        return self

    def graph_updated(self):
        self.in_degree = {}
        self._initialize_degrees()
        for node in self.traversed:
            for n in node.next:
                self.in_degree[n] -= 1

    def __next__(self):
        if self.current_node is not None and len(self.current_node.next) > 0:

            for n in self.current_node.next:
                if n not in self.in_degree:
                    self.graph_updated()
                    break
                else:
                    self.in_degree[n] -= 1
            for n in self.current_node.next:
                if self.in_degree[n] == 0:
                    self.Q.append(n)
        if len(self.Q) == 0:
            raise StopIteration
        self.current_node = self.Q.pop(0)
        self.traversed.append(self.current_node)

        return self.current_node


def nodes(pipe : "TemporisPipeline"):
    _nodes = set()
    for root in root_nodes(pipe):        
        for node in topological_sort_iterator(root):
            _nodes.add(node)
    return list(_nodes)


def edges(pipe: "TemporisPipeline"):
    _edges = set()
    for n in nodes(pipe):
        for n_next in n.next:
            if (n, n_next) not in _edges and (n_next, n) not in _edges:
                _edges.add((n, n_next))
    return _edges