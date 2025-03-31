import numpy as np

class Optimizer:
    def __init__(self):
        raise NotImplementedError(f'{self.__class__.__name__} is not implemented.')
    
    def prev_update(self):
        raise NotImplementedError(f'prev_update() is not implemented.')
    
    def update_params(self, params: dict, grads: dict):
        raise NotImplementedError(f'update_params() is not implemented.')
    
    def step(self):
        raise NotImplementedError(f'step() is not implemented.')
    
class Adam(Optimizer):
    def __init__(self, lr=0.001, decay=0, betas=(0.9, 0.999), epsilon=1e-8):
        self.lr = lr
        self.current_lr = lr
        self.decay = decay
        self.beta_1 = betas[0]
        self.beta_2 = betas[1]
        self.epsilon = epsilon
        self.t = 1
        self.m = {}
        self.v = {}
    
    def prev_update(self):
        if self.decay:
            self.current_lr = self.lr * (1 / (1 + self.decay * self.t))

    def update_params(self, params: dict, grads: dict):
        if self.t == 1:
            for name, val in params.items():
                if name not in self.m:
                    self.m[name] = np.zeros_like(val)
                if name not in self.v:
                    self.v[name] = np.zeros_like(val)

        for param_name, grad_name in zip(params.keys(), grads.keys()):
            self.m[param_name] = self.beta_1 * self.m[param_name] + (1 - self.beta_1) * grads[grad_name]
            self.v[param_name] = self.beta_2 * self.v[param_name] + (1 - self.beta_2) * grads[grad_name]**2
            m_h = self.m[param_name] / (1 - self.beta_1**self.t)
            v_h = self.v[param_name] / (1 - self.beta_2**self.t)
            params[param_name] -= self.current_lr * (m_h / (np.sqrt(v_h) + self.epsilon))

    def step(self):
        self.t += 1