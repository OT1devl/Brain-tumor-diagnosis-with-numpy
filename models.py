import numpy as np
import pickle
from activations import *
from layers import *
from accuracies import *
from optimizers import *
from losses import *
import time

class Model:
    count = 0
    def __init__(self, name=None):
        if name:
            self.name = name
        else:
            self.name = f'{self.__class__.__name__}_{self.__class__.count}'
        __class__.count += 1

    def compile(self, loss: Loss, optimizer: Optimizer, accuracy: Accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def forward(self, x):
        raise NotImplementedError('forward() is not implemented.')
    
    def backward(self, delta, m):
        raise NotImplementedError('backward() is not implemented.')
    
    def save_model(self, path):
        params = {}
        
        params['type'] = self.__class__.__name__
        params['name'] = self.name

        if hasattr(self, 'layers'):
            params['layers'] = [layer.save_parameters() for layer in self.layers]

        if hasattr(self, 'loss'):
            params['loss'] = self.loss.save_parameters()

        if hasattr(self, 'optimizer'):
            params['optimizer'] = self.optimizer.save_parameters()

        if hasattr(self, 'accuracy'):
            params['accuracy'] = self.accuracy.save_parameters()
        
        with open(path, 'wb') as f:
            pickle.dump(params, f)

    @staticmethod
    def load_model(path):
        with open(path, 'rb') as f:
            model_dict : dict = pickle.load(f)
        # print(model_dict)
        name = model_dict['name']
        type_model = model_dict['type']

        layers_list = model_dict['layers']
        
        layers = []
        for layer in layers_list:
            layer_name, layer_params = list(layer.items())[0]
            layer_class : Layer = globals().get(layer_name)
            try:
                layer_layer = layer_class.re_init(**layer_params) # Normal layer
            except:
                layer_layer = layer_class(**layer_params) # Activation
            layers.append(layer_layer)

        loss_dict: dict = model_dict.get('loss', None)

        loss = None
        if loss_dict:
            loss_name, loss_params = list(loss_dict.items())[0]
            loss_class : Loss = globals().get(loss_name)
            if loss_class:
                loss = loss_class(**loss_params)

        optimizer_dict: dict = model_dict.get('optimizer', None)

        optimizer = None
        if optimizer_dict:
            optimizer_name, optimizer_params = list(optimizer_dict.items())[0]
            optimizer_class: Optimizer = globals().get(optimizer_name)
            if optimizer_class:
                optimizer = optimizer_class.re_init(**optimizer_params)
        
        accuracy_name = model_dict.get('accuracy', None)
        
        accuracy = None
        if accuracy_name:
            accuracy: Accuracy = globals().get(accuracy_name)()

        model_class = globals().get(type_model, -1)
        if model_class == -1: raise ValueError('No model found.')
        model: Model = model_class(name=name)
        if hasattr(model, 'layers'):
            model.layers = layers
        model.compile(loss=loss, optimizer=optimizer, accuracy=accuracy)
        return model
    
class PipeLine(Model):
    def __init__(self, layers: tuple=None, name=None):
        super().__init__(name)
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, delta):

        for layer in reversed(self.layers):
            delta = layer.backward(delta)

        return delta

    def update_params(self):
        for layer in self.layers:
            if hasattr(layer, 'give_optimizer'):
                self.optimizer.update_params(*layer.give_optimizer())

    def train(self, x, y, epochs=10, batch_size=32, print_every=0.1, shuffle=True, x_test=None, y_test=None):
        loss_history, acc_history = [], []

        for ep in range(1, epochs+1):
            if shuffle:
                KEYS = np.arange(x.shape[0])
                x = x[KEYS]
                y = y[KEYS]
            
            loss_ep = 0.0
            acc_ep = 0.0
            start = time.time()

            for i in range(0, x.shape[0], batch_size):
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                predictions = self.forward(x_batch)
                loss_ep += self.loss.calculate(y_batch, predictions)
                acc_ep += self.accuracy.calculate(y_batch, predictions)
                
                dL = self.loss.backward(y_batch, predictions)
                self.backward(dL)

                self.optimizer.prev_update()
                self.update_params()
                self.optimizer.step()
                print(f'Epoch: {ep} | Batch: [{i//batch_size+1}/{x.shape[0]//batch_size}]', end='\r')
            avg_loss = loss_ep / x.shape[0]
            avg_acc = acc_ep / x.shape[0]

            loss_history.append(avg_loss)
            acc_history.append(avg_acc)

            if ep % max(1, int(epochs*print_every)) == 0:
                message = f'Epoch: [{ep}/{epochs}] time: {time.time()-start:.2f}> Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}'
                if x_test is not None and y_test is not None:
                    loss_ep = 0.0
                    acc_ep = 0.0
                    for i in range(0, x_test.shape[0], 64):
                        x_batch = x[i:i+batch_size]
                        y_batch = y[i:i+batch_size]
                        predictions = self.forward(x_batch)
                        loss_ep += self.loss.calculate(y_batch, predictions)
                        acc_ep += self.accuracy.calculate(y_batch, predictions)

                    avg_loss = loss_ep / x_test.shape[0]
                    avg_acc = acc_ep / x_test.shape[0]
                    message += f' | Loss Test: {avg_loss:.4f}, Acc Test: {avg_acc:.4f}'
                print(message)
        return loss_history, acc_history