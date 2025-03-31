import numpy as np

class Loss:
    def __init__(self, mode='sum'):
        modes = 'sum', 'mean'
        mode = mode.lower()
        if mode in modes:
            self.mode = mode
        else:
            raise ValueError(f'{mode} is not in {modes}')
        
    def forward(self, y_true, y_pred):
        raise NotImplementedError('forward() is not implemented.')
    
    def calculate(self, y_true, y_pred):
        func = np.sum if self.mode == 'sum' else np.mean
        return func(self.forward(y_true, y_pred))

    def backward(self, y_true, y_pred):
        raise NotImplementedError('backward() is not implemented.')
    
class BinaryCrossEntropy(Loss):
    def __init__(self, mode='sum', epsilon=1e-8):
        super().__init__(mode)
        self.epsilon = epsilon

    def forward(self, y_true, y_pred):
        return -y_true*np.log(y_pred + self.epsilon) - (1 - y_true)*np.log(1 - y_pred + self.epsilon)
    
    def backward(self, y_true, y_pred):
        return -y_true/(y_pred + self.epsilon)+(1 - y_true)/(1 - y_pred + self.epsilon)