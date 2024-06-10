from abc import ABC
from codietpgm.PGM import PGM
import networkx as nx



class Variables():
    def __init__(self, sizen, variables):
       if(type(variables) == list): 


class DynamicBayesianNetwork(PGM, ABC):
    """Abstract class for dynamic Bayesian networks (DBNs). Generally, DBNs consist of initial Bayesian network which
    encodes prior distribution, and a second network which encodes transition probabilities from one time slice to another."""
    

    def __init__(self, sizen, vartype, variables, model):
        """
        Creates a new instance of the Dynamic Bayesian Network class.

        Parameters:

        """
        self.__sizen = sizen
        if(vartype == 'custom'): #list of Variable classes
             self.__variables = Variables(sizen, variables)
        else if(vartype == 'binary'): #binary variables, variables initializes
             self.__variables = Variables(sizen, 'binary')
        else if(vartype == 'discrete'): #discrete variables, variables initializes
             self.__variables = Variables(sizen, 'discrete')
        else if(vartype == 'gaussian'): #gaussian variables 
             self.__variables = Variables(sizen, 'gaussian')
        else if(vartype == 'lsem'): #continuous with linear model
             self.__variables = Variables(sizen, 'LSEM')
        else:
          raise RuntimeError("Illegal Variables Input.')     
        self.__model = model
        
        
        
        super().__init__(structure_and_weights)

