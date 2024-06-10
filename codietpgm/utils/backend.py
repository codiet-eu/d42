from enum import Enum


class BackendType(Enum):
    """
    Enumeration class representing avaiable backend types.

    Attributes:
        NUMPY: Numpy package.
        TORCH: Pytorch package.
        TENSORFLOW: Tensorflow package.
        CUPY: CUPY package.
    """
    NUMPY = "np"
    TORCH = "torch"
    TENSORFLOW = "tensorflow.experimental.numpy"
    CUPY = "cupy"


class GenericBackend:
    def __init__(self, backend=BackendType.NUMPY):
        self._np = None
        self._backend = None
        self.set_backend(backend)

    def set_backend(self, backend):
        if backend in BackendType:
            self._np = __import__(self.backends[backend])
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        self._backend = backend

    def array(self, data, dtype=None):
        return self._np.array(data, dtype=dtype)

    def mean(self, data):
        return self._np.mean(data)

    def std(self, data, ddof=1):
        return self._np.std(data, ddof=ddof)

    def random_normal(self, mu, sigma, n_samples):
        return self._np.random.normal(mu, sigma, n_samples)

    def random_binomial(self, n, p, size):
        return self._np.random.binomial(n, p, size)

