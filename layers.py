import numpy as np
from utils import fast_convolution, fast_convolution_backprop, fast_maxpool, fast_maxpool_backprop
from activations import *

DTYPE = np.float32

class Layer:
    count = 0
    def __init__(self, name=None):
        if name:
            self.name = name
        else:
            self.name = f'{self.__class__.__name__}_{self.__class__.count}'
        __class__.count += 1

    def re_init(self):
        raise NotImplementedError('re_init() is not implemented.')

    def forward(self, x):
        raise NotImplementedError('forward() is not implemented.')
    
    def backward(self, delta, m):
        raise NotImplementedError('backward() is not implemented.')
    
    def give_optimizer(self):
        # Claves básicas para parámetros y gradientes
        param_keys = ('W', 'b')
        grad_keys = ('dW', 'db')
        parameters = {}
        gradients = {}
        
        for key, val in self.__dict__.items():
            if key in param_keys:
                # Incluir el nombre de la capa para evitar conflictos
                parameters[f'{self.name}_{key}'] = val
            elif key in grad_keys:
                gradients[f'{self.name}_{key}'] = val

        return parameters, gradients

    
class Dense(Layer):
    def __init__(self, neurons_in, neurons_out, activation=None, name=None):
        super().__init__(name)
        self.activation = activation
        self.W = np.random.randn(neurons_in, neurons_out).astype(DTYPE) * np.sqrt(2/neurons_in)
        self.b = np.zeros((1, neurons_out)).astype(DTYPE)

    @classmethod
    def re_init(cls, name, W, b, activation):
        new_cls = cls(1, 1, name=name)
        new_cls.W = W
        new_cls.b = b

        if activation:
            activation_name, activation_params = list(activation.items())[0]
            activation_class: Activation = globals().get(activation_name)
            new_cls.activation = activation_class(**activation_params)
        else:
            new_cls.activation = None

        return new_cls

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
    
    def save_parameters(self):
        exceptions = 'outputs', 'inputs', 'dW', 'db'
        params = {}

        for param_name, param_val in self.__dict__.items():
            if param_name in exceptions: continue

            if isinstance(param_val, Activation):
                params[param_name] = param_val.save_parameters()
            else:
                params[param_name] = param_val

        return {self.__class__.__name__: params}
    
class Conv2d(Layer):
    def __init__(self, in_channels, filters, kernel_size=(3, 3), stride=1, padding=1, name=None):
        super().__init__(name)
        self.W = np.random.randn(kernel_size[0], kernel_size[1], in_channels, filters).astype(DTYPE) * np.sqrt(2/(kernel_size[0]*kernel_size[1]*in_channels))
        self.b = np.zeros((1, 1, 1, filters)).astype(DTYPE)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        self.inputs = x
        self.outputs = fast_convolution(x, self.W, self.b, self.padding, self.stride)
        return self.outputs
    
    def backward(self, dout):
        dout, self.dW, self.db = fast_convolution_backprop(self.inputs, self.W, dout, self.padding, self.stride)
        return dout
    
    def save_parameters(self):
        exceptions = 'outputs', 'inputs', 'dW', 'db'
        params = {}

        for param_name, param_val in self.__dict__.items():
            if param_name in exceptions: continue

            params[param_name] = param_val

        return {self.__class__.__name__: params}

class MaxPool2d(Layer):
    def __init__(self, pool_height, pool_width, stride=1, padding=1, name=None):
        super().__init__(name)
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.stride = stride
        self.padding = padding
    
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

    @classmethod
    def re_init(cls, input_shape, name):
        new_cls = cls(input_shape, name)
        return new_cls

    def forward(self, x):
        self.input_shape = x.shape
        self.outputs = x.reshape(x.shape[0], -1)
        return self.outputs
    
    def backward(self, dout):
        return dout.reshape(self.input_shape)
    
    def save_parameters(self):
        exceptions = 'outputs'
        params = {}

        for param_name, param_val in self.__dict__.items():
            if param_name in exceptions: continue

            params[param_name] = param_val

        return {self.__class__.__name__: params}