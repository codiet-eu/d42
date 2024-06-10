import random
import numpy as np

class GenericBackend:
    def __init__(self, backend='numpy'):
        self.backends = {
            'numpy': np,
            'torch': 'torch',
            'tensorflow': 'tensorflow.experimental.numpy',
            'cupy': 'cupy'
        }
        self.set_backend(backend)

    def set_backend(self, backend):
        if backend in self.backends:
            self.np = __import__(self.backends[backend])
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        self.backend = backend

    def array(self, data, dtype=None):
        return self.np.array(data, dtype=dtype)

    def mean(self, data):
        return self.np.mean(data)

    def std(self, data, ddof=1):
        return self.np.std(data, ddof=ddof)

    def random_normal(self, mu, sigma, n_samples):
        return self.np.random.normal(mu, sigma, n_samples)

    def random_binomial(self, n, p, size):
        return self.np.random.binomial(n, p, size)

