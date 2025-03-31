import numpy as np

class Accuracy:
    def calculate(self, y_true, y_pred):
        raise NotImplementedError('calculate() is not implemented.')
    
class BinaryAccuracy(Accuracy):
    def calculate(self, y_true, y_pred):
        return y_true == y_pred
    
class FactorAccuracy(Accuracy):
    def __init__(self, factor=0.5):
        self.factor = factor

    def calculate(self, y_true, y_pred):
        return (y_pred > self.factor) == y_true