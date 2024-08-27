# CodietPGM

This repository represents a toolkit for structure learning of DBNs in the CoDiet [1]
project. Currently, this repository is under development. The goal is to finally
extend this repository into a toolkit that would complement `pgmpy` library [2] in the
case of Dynamic Bayesian Networks.

The repository contains several bundles of code. There are structured as follows:
* The `io` folder contains tools for loading and processing data. The organization
  of the data is focused on the types of data that will arise in the CoDiet project.
  The `Data` class contains a collection of `Samples`, which will represent individual
  patients. Each `Sample` has a set of static features, and a set of dynamic features
  that are stored in a data frame. For each of the variables, annotations are available,
  so that a relevant set of variables can be selected for an experiment.
* Once the data are loaded, the user can use one of the `learners` to learn the structure
  and parameters of the model. There are several learners available. For some, external
  libraries are used, as in the case `DyNoTearsDBN`, some are related to publication
  [3], as `MILPDBN`.
* The `structure` package contains a representation of the learned model. The model
  consists of instances of `Node`, which represent individual variables. Then, each
  node is attached transitions that have a probability distribution over `input_nodes`.
* In `utils`, you can find helper code. 
* The `evaluation` folder (under development) will contain tools to compare different
  results obtained by learners.

[1] https://www.codiet.eu/

[2] https://pgmpy.org/

[3] https://arxiv.org/abs/2406.17585