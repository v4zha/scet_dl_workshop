import numpy as np
from graphviz import Digraph
from typing import Union, Optional, Tuple, Set, List


class Value:
    def __init__(self, value: float, label: Optional[str] = "", _op: Optional[str] = "", _children=()):
        self.value = value
        self.grad = 0
        self._prev = set(_children)
        self.label = label
        self._op = _op
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(val={self.value}, grad={self.grad})"

    def __add__(self, other: Union[int, float, 'Value']) -> "Value":
        if isinstance(other, (int, float)):
            other = Value(value=other)
        out = Value(value=self.value + other.value,
                    _op="+", _children=(self, other))

        def _backward():
            self.grad += 1*out.grad
            other.grad += 1*out.grad
        out._backward = _backward
        return out

    def __mul__(self, other: Union[int, float, 'Value']) -> "Value":
        if isinstance(other, (int, float)):
            other = Value(value=other)
        out = Value(value=self.value * other.value,
                    _op="*", _children=(self, other))

        def _backward():
            self.grad += other.value*out.grad
            other.grad += self.value*out.grad
        out._backward = _backward
        return out

    def __radd__(self, other: Union[int, float, 'Value']) -> "Value":
        return self + other

    def __rmul__(self, other: Union[int, float, 'Value']) -> "Value":
        return self * other

    def __sub__(self, other: Union[int, float, 'Value']) -> "Value":
        return self+ (other*-1)

    def __rsub__(self, other: Union[int, float, 'Value']) -> "Value":
        return other + (self*-1)

    def __pow__(self, other: Union[int, float, 'Value']) -> "Value":
        if isinstance(other, (int, float)):
            other = Value(value=other)
        out = Value(value=self.value**other.value,
                    _op="**", _children=(self, other))
        def _backward():
            self.grad+=other.value*self.value**(other.value-1)*out.grad
            other.grad+=self.value**other.value*np.log(self.value)*out.grad
        out._backward = _backward
        return out
    

    def __true_div__(self, other: Union[int, float, 'Value']) -> "Value":
        out = self*other**-1
        return out
    def __rtrue_div__(self, other: Union[int, float, 'Value']) -> "Value":
        out = other*self**-1
        return out

    def relu(self)->'Value':
        out=Value(value=max(0,self.value),_op="relu",_children=(self))
        def _backward():
            self.grad+=1 if self.value>0 else 0 * out.grad
        out._backward=_backward
    
    def exp(self)->'Value':
        out=Value(value=np.exp(self.value),_op="exp",_children=(self))
        def _backward():
            self.grad+=np.exp(self.value)*out.grad
        out._backward=_backward
        return out

    def tanh(self)->'Value':
        out=Value(value=np.tanh, _op="tanh",_children=(self))
        def _backward():
            self.grad+=(1-np.tanh(self.value)**2)*out.grad
        out._backward=_backward
        return out

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
        graph = Digraph(format='svg', graph_attr={
                        'rankdir': 'LR'})
        # get edges and nodes
        edges, nodes = self._get_edges_and_nodes()
        for node in nodes:
            node_id = str(id(node))
            graph.node(
                name=node_id, label=f"{node.label} | value : {node.value:.2f} | grad : {node.grad:.2f}", shape="rectangle")
            if node._op:
                graph.node(name=node_id+node._op,
                           label=node._op, shape="circle", color="red")
                graph.edge(node_id+node._op, node_id)

        for node1, node2 in edges:
            graph.edge(str(id(node1)), str(id(node2))+node2._op)

        return graph

    def _topo_sort(self) -> List['Value']:
        visited = set()
        stack = []
        # use DFS for topological sorting 
        def visit(node: 'Value'):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    visit(child)
                stack.append(node)
        visit(self)
        return reversed(stack)

    def backward(self):
        self.grad = 1
        for node in self._topo_sort():
            node._backward()
