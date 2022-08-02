from typing import Tuple

from temporis.transformation.functional.graph_utils import edges, nodes
from temporis.transformation.functional.transformerstep import TransformerStep


def encode_tuple(tup: Tuple):
    def get_hash(x):
        if isinstance(x, TransformerStep):
            return hash(x)
        else:
            return hash(x)

    return ",".join(list(map(lambda x: str(get_hash(x)), tup)))


def decode_tuple(s: str):
    return s.split(",")


def make_pipeline(*steps):
    from temporis.transformation.functional.pipeline.pipeline import \
        TemporisPipeline

    for s in range(1, len(steps)):
        steps[s](steps[s - 1])
    return TemporisPipeline(steps[-1])


def plot_pipeline(pipe: "TemporisPipeline", name: str):
    import graphviz
    from temporis.transformation.functional.pipeline.pipeline import \
        TemporisPipeline

    dot = graphviz.Digraph(name, comment="Transformation graph")

    node_name = {}
    for i, node in enumerate(nodes(pipe)):
        node_name[node] = str(i) + node.name
        dot.node(str(i) + node.name, label=str(node))

    for (e1, e2) in edges(pipe):
        dot.edge(node_name[e1], node_name[e2])

    return dot
