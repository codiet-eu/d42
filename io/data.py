import pandas as pd
from io.timeseries import Sample


class Data:
    def __init__(self, samples):
        """
        Initialize Data with a list of samples.

        Parameters:
        samples (list): List of Sample objects.
        """
        self._list = samples

    def add_sample(self, sample):
        """
        Add a sample to the list of samples.

        Parameters:
        sample (Sample): Sample object to add.
        """
        self._list.add(sample)

    def __iter__(self):
        """
        Make the Data object iterable.

        Returns:
        iterator: Iterator over the list of samples.
        """
        return iter(self._list)
