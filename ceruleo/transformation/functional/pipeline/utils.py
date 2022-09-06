from typing import Tuple

from ceruleo.transformation.functional.graph_utils import edges, nodes
from ceruleo.transformation.functional.transformerstep import TransformerStep


def encode_tuple(tup: Tuple):
    def get_hash(x):
        if isinstance(x, TransformerStep):
            return hash(x)
        else:
            return hash(x)

    return ",".join(list(map(lambda x: str(get_hash(x)), tup)))


def decode_tuple(s: str):
    return s.split(",")


def plot_pipeline(pipe: "TemporisPipeline", name: str):
    """Plot the transformation pipeline

    Parameters:
    
        pipe: The pipeline
        name: Title of the graphic

    Returns:

        graphic: the diagram
        
    """
    import graphviz
    from ceruleo.transformation.functional.pipeline.pipeline import \
        Pipeline

    dot = graphviz.Digraph(name, comment="Transformation graph")

    node_name = {}
    for i, node in enumerate(nodes(pipe)):
        node_name[node] = str(i) + node.name
        dot.node(str(i) + node.name, label=str(node))

    for (e1, e2) in edges(pipe):
        dot.edge(node_name[e1], node_name[e2])

    return dot
