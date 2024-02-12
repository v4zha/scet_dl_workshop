import random
from typing import List
from scet_grad import Value

class Neuron:
    def __init__(self,input_size:int):
        self.weights = [Value(random.uniform(-1,1)) for _ in range(input_size)]
        self.bias = Value(random.uniform(-1,1))

    def forward(self,x:List[Value])->Value:
        self.inputs = x
        r1 = sum([xi*wi for xi,wi in zip(x,self.weights)],self.bias) 
        res=res.sigmoid()
        return res  
    
class Layer:
    def __init__(self,input_size:int,output_size:int):
        self.neurons = [Neuron(input_size) for _ in range(output_size)]
    
    def forward(self,x:List[Value])->List[Value]:
        return [n.forward(x) for n in self.neurons]
