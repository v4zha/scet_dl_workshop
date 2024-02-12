import random
from typing import List
from scet_grad import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self,input_size:int):
        self.weights = [Value(random.uniform(-1,1)) for _ in range(input_size)]
        self.bias = Value(random.uniform(-1,1))

    def __call__(self,x:List[Value])->Value:
        r1 = sum([xi*wi for xi,wi in zip(x,self.weights)],self.bias) 
        res=r1.sigmoid()
        return res  
    
    def parameters(self):
        return self.weights + [self.bias]
    
    def __repr__(self):
        return f"Neuron({len(self.weights)})"
    
class Layer:
    def __init__(self,input_size:int,output_size:int):
        self.neurons = [Neuron(input_size) for _ in range(output_size)]
    
    def __call__(self,x:List[Value])->List[Value]:
        out=[n(x) for n in self.neurons]
        return out[0] if len(out)==1 else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return "Layer: "+", ".join([str(n) for n in self.neurons])

class Sequential(Module):
    def __init__(self,nin,nouts ):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]


    def __call__(self,x:List[Value])->Value:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return "Sequential:\n"+"\n".join([str(l) for l in self.layers])