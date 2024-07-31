from abc import ABC, abstractmethod


class Evaluator(ABC):
    """
    A class intended for evaluation of set of learners. It is presumed that the subclasses will get the data and all
    the settings in the constructor, and then evaluate a set of learners using the abstract methods below.
    """
    def evaluate(self, learners):
        results = []
        for learner in learners:
            results.append(self.evaluate_single(learner))
            # TODO some nice intervace
        return results

    @abstractmethod
    def evaluate_single(self, learner):
        return {}
