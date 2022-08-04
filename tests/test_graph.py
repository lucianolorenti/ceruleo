from ceruleo.transformation.functional.graph_utils import topological_sort_iterator
from ceruleo.transformation.functional.transformerstep import TransformerStep


class VisitableNode(TransformerStep):
    def visit(self):
        pass


class A(VisitableNode):
    pass


class B(VisitableNode):
    class B1(VisitableNode):
        pass

    class B2(VisitableNode):
        pass

    def __init__(self, *, name):
        super().__init__(name=name)
        self.b1 = B.B1(name="B1")
        self.b2 = B.B2(name="B2")(self.b1)


class VisitableNode(TransformerStep):
    def visit(self):
        pass


class A(VisitableNode):
    pass


class B(VisitableNode):
    class B1(VisitableNode):
        pass

    class B2(VisitableNode):
        pass

    def __init__(self, *, name: str):
        super().__init__(name=name)
        self.b1 = B.B1(name="B1")
        self.b2 = B.B2(name="B2")(self.b1)

    def visit(self):
        for n in self.next:
            self.disconnect(n)
            self.b2.add_next(n)
        self.b1(self)


class C(VisitableNode):
    pass


class Node(TransformerStep):
    pass

    def visit(self):
        self.disconnect(self.next)
        self.b2.add_next(self.next)
        self.b1(self)


class C(VisitableNode):
    pass


class Node(TransformerStep):
    pass


class TestGraph:
    def test_simple(self):
        pipe = Node(name="A")
        pipe = Node(name="B")(pipe)
        pipe = Node(name="C")(pipe)

        assert pipe.previous[0].name == "B"
        assert pipe.previous[0].previous[0].name == "A"

        topological_sort_iterator

    def test_graph_updating(self):
        pipe = A(name="A")
        pipe = B(name="B")(pipe)
        pipe = C(name="C")(pipe)

        pipe.previous[0].visit()
        assert pipe.previous[0].name == "B2"
        assert pipe.previous[0].previous[0].name == "B1"
        assert pipe.previous[0].previous[0].previous[0].name == "B"

        pipe = A(name="A")
        pipe = B(name="B")(pipe)
        pipe = C(name="C")(pipe)

        result = []
        for a in topological_sort_iterator(pipe):
            a.visit()
            result.append(a.name)

        assert result == ["A", "B", "B1", "B2", "C"]

    def test_diamond(self):
        pipe = Node(name="A")
        pipeB = Node(name="B")(pipe)
        pipeC = Node(name="C")(pipe)
        pipeD = Node(name="D")(pipe)
        pipe = Node(name="E")([pipeB, pipeC, pipeD])
