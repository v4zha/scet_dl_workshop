from typing import Union, Optional, Tuple, Set
from graphviz import Digraph


class Value:
    def __init__(self, value: float, label: Optional[str] = "", _op: Optional[str] = "", _children=()):
        self.value = value
        self.grad = 0
        self._prev = set(_children)
        self.label = label
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(val={self.value}, grad={self.grad})"

    def __add__(self, other: Union[int, float, 'Value']) -> "Value":
        if isinstance(other, (int, float)):
            other = Value(value=other)
        return Value(value=self.value + other.value, _op="+", _children=(self, other))

    def __mul__(self, other: Union[int, float, 'Value']) -> "Value":
        if isinstance(other, (int, float)):
            other = Value(value=other)
        else:
            return Value(value=self.value * other.value, _op="*", _children=(self, other))

    def __radd__(self, other: Union[int, float, 'Value']) -> "Value":
        return self + other

    def __rmul__(self, other: Union[int, float, 'Value']) -> "Value":
        return self * other

    def _get_edges_and_nodes(self) -> Tuple[Set['Value'], Set['Value']]:
        edges, nodes = set(), set()
        # DFS to get edges and nodes

        def dfs_ne(node: 'Value'):
            if node not in nodes:
                nodes.add(node)
                for child in node._prev:
                    edges.add((child, node))
                    dfs_ne(child)
        dfs_ne(self)
        return edges, nodes

    def digraph(self):
        # rankdir : graph will be drawn from left to right
        graph = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
        # get edges and nodes
        edges, nodes = self._get_edges_and_nodes()
        for node in nodes:
            node_id = str(id(node))
            graph.node(
                name=node_id, label=f"{node.label} | {node.value:.2f}", shape="rectangle")
            if node._op:
                graph.node(name=node_id+node._op,
                           label=node._op, shape="circle")
                graph.edge(node_id+node._op, node_id)

        for node1, node2 in edges:
            graph.edge(str(id(node1)), str(id(node2))+node2._op)

        return graph
