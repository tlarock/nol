# NOL\*: Network Online Learning

This repository contains code and data to reproduce the following paper:

Larock, Sakharov, Bahdra, and Eliassi-Rad. Understanding the Limitations of Network Online Learning. Forthcoming in Applied Network Science.

This code is meant for reproducibility and experimentation, not for general use. If you would like to use the method in a specific setting, we do not encourage you to try to use this code for that purpose, but instead use this implementation as a guide and reference for how to use the methodology. 

Scripts to reproduce the experiments from the paper are located in the reproduce/ directory. Scripts should automatically generate plots, but plotting code is also available in the plotting/ directory. 

*NOTE*: The experiments in this paper are very computationally expensive. We used a large computer with many processors to generate our results. Running these experiments, especially the parameter search, without multiprocessing is not recommended and will likely take days.
