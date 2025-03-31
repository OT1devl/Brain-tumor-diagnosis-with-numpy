import numpy as np

class Activation:
    def forward(self):
        raise NotImplementedError('forward() is not implemented.')
    
    def backward(self):
        raise NotImplementedError('backward() is not implemented.')
    
class ReLU(Activation):
    def forward(self, x):
        self.inputs = x
        self.outputs = np.maximum(x, 0)
        return self.outputs
    
    def backward(self, dout):
        return dout * np.where(self.inputs>0, 1, 0)
    
class LeakyReLU(Activation):
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def forward(self, x):
        self.inputs = x
        self.outputs = np.where(x>0, x, x*self.alpha)
        return self.outputs

    def backward(self, dout):
        return dout * np.where(self.inputs>0, 1, self.alpha)
    
class Sigmoid(Activation):
    def forward(self, x):
        self.inputs = x
        self.outputs = 1 / (1 + np.exp(-x))
        return self.outputs
    
    def backward(self, dout):
        return dout * (self.outputs * (1 - self.outputs))