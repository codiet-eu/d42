from enum import Enum


class VariableType(Enum):
    """
    Enumeration class representing variable types.

    Attributes:
        static: Represents a static variable that does not change over time.
        dynamic: Represents a dynamic variable that is time-dependent and may change over time.
    """
    static = "static"
    dynamic = "dynamic"


class StaticVariable(Enum):
    """
    Enumeration class representing static variable sub-types.

    Attributes:
        fixed: Represents a static variable that does not change over time at all, for example, height of an
    adult individual can be considered fixed.
        slow: Represents a dynamic variable that is changing over time but this
    change cannot be captured by the experiments.
    """
    fixed = "fixed"
    slow = "slow"


class Source(Enum):
    """
    Enumeration class representing source of the data.
    """
    metabolomics = "metabolomics"
    sequencing = "sequencing"
    questionaire = "questionaire"
    lipidomics = "lipidomics"

