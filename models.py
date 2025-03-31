import numpy as np
import time
from losses import Loss
from optimizers import Optimizer
from accuracies import Accuracy
from utils import shuffle_data

class Model:
    pass

class PipeLine(Model):
    def __init__(self, layers: tuple):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def compile(self, loss: Loss, optimizer: Optimizer, accuracy: Accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def update_params(self):
        self.optimizer.prev_update()
        for layer in self.layers:
            if hasattr(layer, 'give_optimizer'):
                self.optimizer.update_params(*layer.give_optimizer())
        self.optimizer.step()

    def train(self, x, y, epochs=10, batch_size=32, shuffle=True, verbose=True, print_every=0.1, x_test=None, y_test=None):
        
        history_loss, history_acc = [], []

        for ep in range(1, epochs+1):
            
            if shuffle:
                x, y = shuffle_data(x, y)

            loss_ep = 0.0
            acc_ep = 0.0
            start = time.time()

            for i in range(0, x.shape[0], batch_size):
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                predictions = self.forward(x_batch)
                loss_ep += self.loss.calculate(y_batch, predictions)
                acc_ep += np.sum(self.accuracy.calculate(y_batch, predictions))
                self.backward(self.loss.backward(y_batch, predictions))
                self.update_params()
                print(f'EPOCH: {ep}, batch: [{i//batch_size+1}/{x.shape[0]//batch_size+1}]', end='\r')
            
            avg_loss = loss_ep / x.shape[0]
            avg_acc = acc_ep / x.shape[0]
            avg_time = time.time()-start

            history_loss.append(avg_loss)
            history_acc.append(avg_acc)

            if verbose and ep % max(1, int(epochs*print_every)) == 0:
                message = f'Epoch: [{ep}/{epochs}] time: {avg_time:.2f} seconds > Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}'

                if x_test is not None and y_test is not None:

                    loss_test = 0.0
                    acc_test = 0.0

                    for i in range(0, x_test.shape[0], batch_size):
                        x_batch = x_test[i:i+batch_size]
                        y_batch = y_test[i:i+batch_size]

                        predictions = self.forward(x_batch)
                        loss_test += self.loss.calculate(y_batch, predictions)
                        acc_test += np.sum(self.accuracy.calculate(y_batch, predictions))
                    
                    avg_loss_test = loss_test / x_test.shape[0]
                    avg_acc_test = acc_test / x_test.shape[0]

                    message += f', Test Loss: {avg_loss_test:.4f}, Test Acc: {avg_acc_test:.4f}'
                
                print(message)

        return history_loss, history_acc