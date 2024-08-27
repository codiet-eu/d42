from abc import ABC, abstractmethod
import numpy as np

class Sampler:
    def __init__(self, dimn, distribution, hyperparameters=None, custom_sample=None):
        """
        Initialize the Sampler with specified distribution and hyperparameters.

        Parameters:
        - dimn: Dimensionality of the input space.
        - distribution: A string specifying the type of distribution ('Gaussian', 'Dirichlet', etc.).
        - hyperparameters: A dictionary of hyperparameters for the distribution.
        - custom_sample: A user-defined function for custom sampling (used if distribution is 'Custom').
        """
        self.dimn = dimn  # Dimensionality of the input space
        self.distribution = distribution.lower()  # Convert to lowercase for consistency
        self.hyperparameters = hyperparameters if hyperparameters is not None else {}
        self.custom_sample = custom_sample
        self.chain = []  # List to store the samples

        # Initialize default hyperparameters based on distribution type
        self._initialize_hyperparameters()

    def _initialize_hyperparameters(self):
        """Initialize default hyperparameters based on the distribution type."""
        if self.distribution == 'gaussian':
            self.hyperparameters.setdefault('mean', np.random.uniform(-1, 1, self.dimn))
            self.hyperparameters.setdefault('std', np.random.uniform(0.1, 2, self.dimn))
        elif self.distribution == 'dirichlet':
            self.hyperparameters.setdefault('alpha', np.random.uniform(0.1, 2, self.dimn))
        elif self.distribution == 'bernoulli':
            self.hyperparameters.setdefault('p', np.random.uniform(0, 1, self.dimn))
        elif self.distribution == 'multinomial':
            self.hyperparameters.setdefault('n', np.random.randint(1, 10))
            self.hyperparameters.setdefault('pvals', np.random.dirichlet(np.ones(self.dimn)))
        elif self.distribution == 'beta':
            self.hyperparameters.setdefault('alpha', np.random.uniform(0.1, 2, self.dimn))
            self.hyperparameters.setdefault('beta', np.random.uniform(0.1, 2, self.dimn))
        elif self.distribution == 'wishart':
            self.hyperparameters.setdefault('df', self.dimn + np.random.randint(1, 5))
            self.hyperparameters.setdefault('scale', np.eye(self.dimn))
        elif self.distribution == 'chisquare':
            self.hyperparameters.setdefault('df', np.random.uniform(1, 10, self.dimn))
        elif self.distribution == 'custom' and self.custom_sample is None:
            raise ValueError("Custom sampling function must be provided for 'Custom' distribution.")

    def update_hyperparameters(self, hyperparameters):
        """Update the hyperparameters for the distribution."""
        self.hyperparameters.update(hyperparameters)

    def sample_parameters(self):
        """Sample parameters based on the specified distribution."""
        if self.distribution == 'gaussian':
            sample = np.random.normal(self.hyperparameters['mean'], self.hyperparameters['std'])
        elif self.distribution == 'dirichlet':
            sample = np.random.dirichlet(self.hyperparameters['alpha'])
        elif self.distribution == 'bernoulli':
            sample = np.random.binomial(1, self.hyperparameters['p'])
        elif self.distribution == 'multinomial':
            sample = np.random.multinomial(self.hyperparameters['n'], self.hyperparameters['pvals'])
        elif self.distribution == 'beta':
            sample = np.random.beta(self.hyperparameters['alpha'], self.hyperparameters['beta'])
        elif self.distribution == 'wishart':
            sample = np.random.wishart(self.hyperparameters['df'], self.hyperparameters['scale'])
        elif self.distribution == 'chisquare':
            sample = np.random.chisquare(self.hyperparameters['df'])
        elif self.distribution == 'custom' and self.custom_sample is not None:
            sample = self.custom_sample(self.hyperparameters)
        else:
            raise ValueError(f"Unsupported distribution type: {self.distribution}")

        self.chain.append(sample)
        return sample

    def reset_chain(self):
        """Reset the sampling chain."""
        self.chain = []

    def get_chain(self, burnin=0):
        """Get the sampling chain with optional burn-in."""
        return self.chain[burnin:]

