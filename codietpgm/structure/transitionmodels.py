from abc import ABC, abstractmethod
import numpy as np
from codietpgm.utils.statstools import Sampler

import statsmodels.api as sm
from itertools import combinations


class StatisticalModel(ABC):
    models = [
        ['Bernoulli', 'Linear'], ['Bernoulli', 'CPD'], ['Bernoulli', 'NoisyOr'], ['Bernoulli', 'Logit'],
        ['Multinomial', 'Linear'], ['Multinomial', 'CPD'], ['Multinomial', 'Logit'],
        ['Mixed', 'GLM'], ['Gaussian', 'Gaussian'], ['Real', 'LSEM'], ['Real', 'General'],
        ['Custom', 'Custom']
    ]

    def __init__(self, input_nodes=None, output_node=None, distribution='custom', hyperparameters=None):
        self.input_nodes = input_nodes if input_nodes else []
        self.output_node = output_node
        self.current_params = {}
        self.sampler = Sampler(len(self.input_nodes), distribution=distribution, hyperparameters=hyperparameters)

    def update_params(self, new_params):
        self.current_params = new_params

    def update_sampler(self, new_hyperparams):
        self.sampler.update_hyperparameters(new_hyperparams)

    def update_struct(self, input_nodes):
        self.input_nodes = input_nodes
        self.init_params()

    @abstractmethod
    def init_params(self):
        pass

    @abstractmethod
    def evaluate(self, input_values):
        pass


class GaussianModel(StatisticalModel):
    def __init__(self, input_nodes=None, output_node=None):
        super().__init__(input_nodes, output_node, distribution='gaussian')
        self.init_params()

    def init_params(self):
        self.current_params = {node.name: np.random.normal(0, 1) for node in self.input_nodes}

    def evaluate(self, input_values):
        if any(val is None for val in input_values):
            raise ValueError("Input values contain None, cannot perform evaluation.")
        params = [self.current_params[node.name] for node in self.input_nodes if node.name in self.current_params]
        if any(param is None for param in params):
            raise ValueError("Model parameters not properly initialized.")
        return np.dot(params, input_values)


class LinearBinaryModel(StatisticalModel):
    def __init__(self, input_nodes=None, output_node=None):
        super().__init__(input_nodes, output_node, distribution='Bernoulli')
        self.init_params()

    def init_params(self):
        self.current_params = {node.name: np.random.lognormal(0, 1) for node in self.input_nodes}

    def evaluate(self, input_values):
        linear_combination = np.dot(input_values, np.array([self.current_params[node.name] for node in self.input_nodes]))
        probability = 1 / (1 + np.exp(-linear_combination))
        return np.random.binomial(1, probability)

    def sample_and_evaluate(self, input_values):
        sampled_params = self.sampler.sample_parameters()
        for i, node in enumerate(self.input_nodes):
            self.current_params[node.name] = sampled_params[i]
        return self.evaluate(input_values)


class CPDBinaryModel(StatisticalModel):
    def __init__(self, input_nodes=None, output_node=None):
        super().__init__(input_nodes, output_node, distribution='Bernoulli')
        self.init_params()

    def init_params(self):
        self.current_params = {}
        all_combinations = sum([list(combinations([node.name for node in self.input_nodes], i))
                                for i in range(1, len(self.input_nodes) + 1)], [])
        for combination in all_combinations:
            self.current_params[combination] = np.random.uniform(-1, 1)

    def evaluate(self, input_values):
        value_dict = {node.name: val for node, val in zip(self.input_nodes, input_values)}
        intercept = self.current_params.get(tuple(), 0)
        linear_sum = sum(self.current_params[comb] * np.prod([value_dict[n] for n in comb]) for comb in self.current_params)
        probability = 1 / (1 + np.exp(-(intercept + linear_sum)))
        return np.random.binomial(1, probability, 1)

    def sample_and_evaluate(self, input_values):
        sampled_params = self.sampler.sample_parameters()
        for comb in self.current_params.keys():
            self.current_params[comb] = sampled_params[0]  # Assuming same beta parameter for simplicity
        return self.evaluate(input_values)


class GeneralizedLinearModel(StatisticalModel):
    def __init__(self, family=sm.families.Gaussian, link=sm.families.links.identity, input_nodes=None, output_node=None):
        super().__init__(input_nodes, output_node, distribution='LSEM')
        self.family = family
        self.link = link
        self.model = None
        self.init_params()

    def init_params(self):
        exog = np.column_stack([np.random.normal(size=100) for _ in self.input_nodes])
        endog = np.random.normal(size=100)
        self.model = sm.GLM(endog, exog, family=self.family(self.link)).fit()

    def evaluate(self, input_values):
        return self.model.predict(input_values)

    def sample_and_evaluate(self, input_values):
        sampled_params = self.sampler.sample_parameters()
        # Assuming a linear combination for GLM simplification
        self.model.params[:] = sampled_params
        return self.evaluate(input_values)


class LinearStructuralEquationModel(StatisticalModel):
    def __init__(self, input_nodes=None, output_node=None):
        super().__init__(input_nodes, output_node, distribution='LSEM')
        self.init_params()

    def init_params(self):
        self.current_params = {node.name: np.random.normal(0, 1) for node in self.input_nodes}

    def evaluate(self, input_values):
        result = np.dot([self.current_params[node.name] for node in self.input_nodes], input_values)
        return result

    def sample_and_evaluate(self, input_values):
        sampled_params = self.sampler.sample_parameters()
        for i, node in enumerate(self.input_nodes):
            self.current_params[node.name] = sampled_params[i]
        return self.evaluate(input_values)


class CustomModel(StatisticalModel):
    def __init__(self, input_nodes=None, output_node=None, custom_function=None):
        super().__init__(input_nodes, output_node, distribution='custom', custom_sample=custom_function)
        self.custom_function = custom_function
        self.init_params()

    def init_params(self):
        pass

    def set_custom_function(self, function):
        self.custom_function = function
        self.sampler.custom_sample = function

    def evaluate(self, input_values):
        if self.custom_function:
            return self.custom_function(input_values)
        raise ValueError("No custom function has been set for evaluation.")

    def sample_and_evaluate(self, input_values):
        sampled_params = self.sampler.sample_parameters()
        for i, node in enumerate(self.input_nodes):
            self.current_params[node.name] = sampled_params[i]
        return self.evaluate(input_values)


class NoisyOrModel(StatisticalModel):
    def __init__(self, input_nodes=None, output_node=None):
        super().__init__(input_nodes, output_node, distribution='beta')
        self.init_params()

    def init_params(self):
        self.current_params = {node.name: np.random.beta(1, 1) for node in self.input_nodes}

    def evaluate(self, input_values):
        noise = np.prod([1 - self.current_params[node.name] for node in self.input_nodes if input_values[node.name] == 1])
        return 1 - noise


class LogitModel(StatisticalModel):
    def __init__(self, input_nodes=None, output_node=None):
        super().__init__(input_nodes, output_node, distribution='gaussian')
        self.init_params()

    def init_params(self):
        self.current_params = {node.name: np.random.normal(0, 1) for node in self.input_nodes}

    def evaluate(self, input_values):
        linear_combination = np.dot([self.current_params[node.name] for node in self.input_nodes], input_values)
        probability = 1 / (1 + np.exp(-linear_combination))
        return np.random.binomial(1, probability)

    def sample_and_evaluate(self, input_values):
        sampled_params = self.sampler.sample_parameters()
        for i, node in enumerate(self.input_nodes):
            self.current_params[node.name] = sampled_params[i]
        return self.evaluate(input_values)


class MultinomialLinearModel(StatisticalModel):
    def __init__(self, input_nodes=None, output_node=None, hyperparameters=None):
        super().__init__(input_nodes, output_node, distribution='dirichlet', hyperparameters=hyperparameters)
        self.init_params()

    def init_params(self):
        """Initialize parameters using one-hot encoding for discrete sets."""
        self.current_params = {}
        for node in self.input_nodes:
            discrete_set = node.get_discrete_set()
            if discrete_set:
                self.current_params[node.name] = np.random.uniform(0, 1, len(discrete_set))
            else:
                raise ValueError(f"Input node {node.name} is not a discrete set.")

    def evaluate(self, input_values):
        """Evaluate model using one-hot encoded input values."""
        linear_combination = np.zeros(len(self.current_params[self.input_nodes[0].name]))
        for node, value in zip(self.input_nodes, input_values):
            one_hot_vector = np.zeros(len(self.current_params[node.name]))
            one_hot_vector[value] = 1  # One-hot encoding of the input value
            linear_combination += self.current_params[node.name] * one_hot_vector
        probabilities = np.exp(linear_combination) / np.sum(np.exp(linear_combination))
        return np.random.choice(len(probabilities), p=probabilities)

    def sample_and_evaluate(self, input_values):
        sampled_params = self.sampler.sample_parameters()
        for i, node in enumerate(self.input_nodes):
            self.current_params[node.name] = sampled_params[i]
        return self.evaluate(input_values)


class MultinomialCPDModel(StatisticalModel):
    def __init__(self, input_nodes=None, output_node=None, hyperparameters=None):
        super().__init__(input_nodes, output_node, distribution='dirichlet', hyperparameters=hyperparameters)
        self.init_params()

    def init_params(self):
        """Initialize parameters for conditional probability distribution."""
        self.current_params = {}
        all_combinations = sum([list(combinations([node.name for node in self.input_nodes], i))
                                for i in range(1, len(self.input_nodes) + 1)], [])
        for combination in all_combinations:
            self.current_params[combination] = np.random.dirichlet(np.ones(len(self.input_nodes)))

    def evaluate(self, input_values):
        """Evaluate model using the input values and CPD parameters."""
        value_dict = {node.name: val for node, val in zip(self.input_nodes, input_values)}
        combination_key = tuple(value_dict.keys())
        probabilities = self.current_params.get(combination_key, np.ones(len(self.input_nodes)) / len(self.input_nodes))
        return np.random.choice(len(probabilities), p=probabilities)

    def sample_and_evaluate(self, input_values):
        sampled_params = self.sampler.sample_parameters()
        for comb in self.current_params.keys():
            self.current_params[comb] = sampled_params
        return self.evaluate(input_values)


class MultinomialLogitModel(StatisticalModel):
    def __init__(self, input_nodes=None, output_node=None, hyperparameters=None):
        super().__init__(input_nodes, output_node, distribution='dirichlet', hyperparameters=hyperparameters)
        self.init_params()

    def init_params(self):
        """Initialize parameters for multinomial logit model."""
        self.current_params = {}
        for node in self.input_nodes:
            discrete_set = node.get_discrete_set()
            if discrete_set:
                self.current_params[node.name] = np.random.normal(0, 1, len(discrete_set))
            else:
                raise ValueError(f"Input node {node.name} is not a discrete set.")

    def evaluate(self, input_values):
        """Evaluate model using logit function and input values."""
        logits = np.zeros(len(self.current_params[self.input_nodes[0].name]))
        for node, value in zip(self.input_nodes, input_values):
            one_hot_vector = np.zeros(len(self.current_params[node.name]))
            one_hot_vector[value] = 1  # One-hot encoding of the input value
            logits += self.current_params[node.name] * one_hot_vector
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        return np.random.choice(len(probabilities), p=probabilities)

    def sample_and_evaluate(self, input_values):
        sampled_params = self.sampler.sample_parameters()
        for i, node in enumerate(self.input_nodes):
            self.current_params[node.name] = sampled_params[i]
        return self.evaluate(input_values)


class CustomModel(StatisticalModel):
    def __init__(self, input_nodes=None, output_node=None, custom_function=None, hyperparameters=None):
        """
        Initialize the CustomModel with user-defined sampling and evaluation functions.

        Parameters:
        - input_nodes: List of input nodes to the model.
        - output_node: The output node of the model.
        - custom_function: A user-defined function for custom evaluation.
        - hyperparameters: A dictionary of hyperparameters for the custom sampling function.
        """
        super().__init__(input_nodes, output_node, distribution='custom', hyperparameters=hyperparameters)
        self.custom_function = custom_function
        if custom_function is not None:
            self.sampler = Sampler(len(self.input_nodes), distribution='custom', hyperparameters=hyperparameters, custom_sample=custom_function)
        else:
            raise ValueError("A custom function must be provided for CustomModel.")

    def init_params(self):
        """
        Initialize model parameters.
        For a custom model, parameter initialization might depend on the custom sampling function.
        Here, we assume the user-defined function also handles initialization.
        """
        pass

    def set_custom_function(self, function):
        """
        Set or update the custom function used for evaluation and sampling.
        
        Parameters:
        - function: A user-defined function for custom evaluation and sampling.
        """
        self.custom_function = function
        self.sampler.custom_sample = function


# Updating Transition class to instantiate appropriate StatisticalModel subtype
class Transition:
    def __init__(self, model, input_nodes, hyperparameters=None):
        self.input_nodes = input_nodes
        self.hyperparameters = hyperparameters if hyperparameters else {}
        self.model = self.create_model_instance(model)

    def create_model_instance(self, model):
        variable, model_type = model
        if variable == 'Bernoulli' and model_type == 'Linear':
            return LinearBinaryModel(self.input_nodes)
        elif variable == 'Bernoulli' and model_type == 'CPD':
            return CPDBinaryModel(self.input_nodes)
        elif variable == 'Bernoulli' and model_type == 'NoisyOr':
            return NoisyOrModel(self.input_nodes)
        elif variable == 'Bernoulli' and model_type == 'Logit':
            return LogitModel(self.input_nodes)
        elif variable == 'Multinomial' and model_type == 'Linear':
            return MultinomialLinearModel(self.input_nodes, hyperparameters=self.hyperparameters)
        elif variable == 'Multinomial' and model_type == 'CPD':
            return MultinomialCPDModel(self.input_nodes, hyperparameters=self.hyperparameters)
        elif variable == 'Multinomial' and model_type == 'Logit':
            return MultinomialLogitModel(self.input_nodes, hyperparameters=self.hyperparameters)
        elif variable == 'Mixed' and model_type == 'GLM':
            return GeneralizedLinearModel(input_nodes=self.input_nodes)
        elif variable == 'Gaussian' and model_type == 'Gaussian':
            return GaussianModel(self.input_nodes)
        elif variable == 'Real' and model_type == 'LSEM':
            return LinearStructuralEquationModel(self.input_nodes)
        elif variable == 'Real' and model_type == 'General':
            return RealGeneralModel(self.input_nodes)
        elif variable == 'Custom' and model_type == 'Custom':
            return CustomModel(self.input_nodes)
        else:
            raise ValueError(f"Unknown model type: {model}")

    def update(self, new_input_nodes, model):
        if set(self.input_nodes) != set(new_input_nodes):
            model.update_struct(new_input_nodes)
            self.input_nodes = new_input_nodes
            self.model = model

    def evaluate(self, data):
        return self.model.evaluate(data)
