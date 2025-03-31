import numpy as np
from activations import Activation
from utils import fast_convolution, fast_convolution_backprop, fast_maxpool, fast_maxpool_backprop

DTYPE = np.float32

class Layer:
    count = 0
    def __init__(self, name=None):
        if name:
            self.name = name
        else:
            self.name = f'{self.__class__.__name__}_{__class__.count}'

        __class__.count += 1

class Dense(Layer):
    def __init__(self, neurons_in, neurons_out, activation: Activation=None, name=None):
        super().__init__(name)
        self.W = np.random.randn(neurons_in, neurons_out).astype(DTYPE) * np.sqrt(2/neurons_in)
        self.b = np.zeros((1, neurons_out)).astype(DTYPE)
        self.activation = activation

    def forward(self, x):
        self.inputs = x
        self.outputs = x @ self.W + self.b

        if self.activation:
            self.outputs = self.activation.forward(self.outputs)

        return self.outputs
    
    def backward(self, dout):
        if self.activation:
            dout = self.activation.backward(dout)

        self.dW = self.inputs.T @ dout / dout.shape[0]
        self.db = dout.sum(axis=0, keepdims=True) / dout.shape[0]

        return dout @ self.W.T
    
    def give_optimizer(self):
        key_params = 'W', 'b'
        key_grads = 'dW', 'db'
        params = {}
        grads = {}

        for k, v in self.__dict__.items():
            if k in key_params:
                params[f'{self.name}_{k}'] = v
            elif k in key_grads:
                grads[f'{self.name}_{k}'] = v

        return params, grads
        
class Conv2d(Layer):
    def __init__(self, channels_in, channels_out, kernel=(3, 3), padding=1, stride=1, activation: Activation=None, name=None):
        super().__init__(name)
        self.W = np.random.randn(kernel[0], kernel[1], channels_in, channels_out).astype(DTYPE) * np.sqrt(2/(kernel[0]*kernel[1]*channels_in))
        self.b = np.zeros((1, 1, 1, channels_out)).astype(DTYPE)
        self.padding = padding
        self.stride = stride
        self.activation = activation

    def forward(self, x):
        self.inputs = x
        self.outputs = fast_convolution(x, self.W, self.b, self.padding, self.stride)

        if self.activation:
            self.outputs = self.activation.forward(self.outputs)

        return self.outputs
    
    def backward(self, dout):
        if self.activation:
            dout = self.activation.backward(dout)
        
        dout, self.dW, self.db = fast_convolution_backprop(self.inputs, self.W, dout, self.padding, self.stride)

        return dout
    
    def give_optimizer(self):
        key_params = 'W', 'b'
        key_grads = 'dW', 'db'
        params = {}
        grads = {}

        for k, v in self.__dict__.items():
            if k in key_params:
                params[f'{self.name}_{k}'] = v
            elif k in key_grads:
                grads[f'{self.name}_{k}'] = v

        return params, grads
    
class MaxPool2d(Layer):
    def __init__(self, pool_height, pool_width, padding=1, stride=1, name=None):
        super().__init__(name)
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.padding = padding
        self.stride = stride

    def forward(self, x):
        self.inputs = x
        self.outputs, self.x_strided = fast_maxpool(x, self.pool_height, self.pool_width, self.stride, self.padding)
        return self.outputs
    
    def backward(self, dout):
        return fast_maxpool_backprop(self.inputs, self.pool_height, self.pool_width, self.stride, self.padding, dout, self.x_strided)

class Flatten(Layer):
    def __init__(self, input_shape=None, name=None):
        super().__init__(name)
        self.input_shape = input_shape

    def forward(self, x):
        self.input_shape = x.shape
        self.outputs = x.reshape(x.shape[0], -1)
        return self.outputs
    
    def backward(self, dout):
        return dout.reshape(self.input_shape)