from enum import Enum


class VariableType(Enum):
    """
    Enumeration class representing variable types.

    Attributes:
        static: Represents a static variable that does not change over time.
        dynamic: Represents a dynamic variable that is time-dependent and may change over time.
    """
    STATIC = "static"
    DYNAMIC = "dynamic"


class StaticVariable(Enum):
    """
    Enumeration class representing static variable sub-types.

    Attributes:
        fixed: Represents a static variable that does not change over time at all, for example, height of an
    adult individual can be considered fixed.
        slow: Represents a dynamic variable that is changing over time but this
    change cannot be captured by the experiments.
    """
    FIXED = "fixed"
    SLOW = "slow"


class Source(Enum):
    """
    Enumeration class representing source of the data.
    """
    METABOLOMICS = "metabolomics"
    LIPIDOMICS = "lipidomics"
    DEMOGRAPHICS = "demographics"
    LIFESTYLE = "lifestyle"
    ANTHROPOMETRIC = "anthropometric"
    BIOCHEMICAL = "biochemical"
    DIETARY = "dietary"
    ACTIVITY = "activity"
    AGES = "ages"
    GENETICS = "genetics"
    MICROBIOME = "microbiome"
    CAMERA = "camera"


class Type(Enum):
    """
    Enumberation whether the variable is discrete or continous.
    """
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"

class Reliability(Enum):
    """
    Shows how reliable the measurements are - whether there is some error expected or not.
    """
    SUBJECTIVE = "subjective"
    EXACT = "exact"
    NOISY = "noisy"
