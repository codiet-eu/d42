#import pandas as pd
from types import MappingProxyType


class Sample:
    """
    Class to hold static variables for input to a Probabilistic Graphical Model,
    as well as a wrapper for time series data.
    """

    def __init__(self, data, static_variables):
        """
        Initialize PGMInput with time series data and static variables.

        Parameters:
        data (pd.DataFrame): DataFrame containing time series data.
                             Columns: 'time', 'variable1', 'variable2', ...
        static_variables (dict): Dictionary containing static variables for PGM input.
        """
        self._data = MappingProxyType(data)
        self._static_variables = static_variables

        # Make the class immutable by preventing attribute modification after initialization
        self.__dict__.setdefault("_immutable", True)

    def get_data(self):
        """
        Get the underlying DataFrame containing time series data.

        Returns:
        pd.DataFrame: DataFrame containing time series data.
        """
        return self._data

    def get_static_variables(self):
        """
        Get the static data dictionary. The returned view is immutable.

        :return: Immutable view of the static data dictionary.
        """
        return self._static_variables

    def get_static_variable(self, name):
        """
        Get the value of a static variable by its name.

        Parameters:
        name (str): Name of the static variable.

        Returns:
        Any: Value of the static variable.
        """
        return self._static_variables.get(name)

    def get_time_step(self, index):
        """
        Get a time step from the time series data.

        Parameters:
        index (int): Index of the time step.

        Returns:
        pd.Series: Series representing the row (time step).
        """
        return self._data.iloc[index]

    def get_variable(self, name):
        """
        Get a specific column (variable) from the time series data.

        Parameters:
        name (str): Name of the column (variable).

        Returns:
        pd.Series: Series representing the column (variable).
        """
        return self._data[name]

    def project(self, variables):
        """
        Returns a view on selected variables.
        :param variables: Variables to
        :return:
        """
        return Sample(self._data[list(variables.intersection(self._data.keys()))],
                      {key: self._static_variables[key] for key in variables if key in self._static_variables})