import pandas as pd
from io.timeseries import Sample


class Data:
    def __init__(self, samples, variables_annotation = {}):
        """
        Initialize Data with a list of samples.

        Parameters:
        samples (list): List of Sample objects.
        """
        self._list = samples
        self._variables_annotation = variables_annotation

    def add_sample(self, sample):
        """
        Add a sample to the list of samples.

        Parameters:
        sample (Sample): Sample object to add.
        """
        self._list.add(sample)

    def add_annotation(self, variable, *annotations):
        """
        Adds a new annotations to the variable.
        :param variable: Variable that should be annotated.
        :param annotations: Properties of the variable. See variableannotation module.
        """
        annot_set = self._variables_annotation.get(variable, set())
        for annotation in annotations:
            annot_set.add(annotation)

    def get_annotations(self, variable):
        """
        Returns unmodifiable view of the annotations of a variable.
        :param variable: Variable to query.
        :return: A set of annotations.
        """
        annotations = self._variables_annotation.get(variable, set())
        return frozenset(annotations)

    def __iter__(self):
        """
        Make the Data object iterable.

        Returns:
        iterator: Iterator over the list of samples.
        """
        return iter(self._list)
