# NOL\*: Network Online Learning

#### Purpose

This repository contains code and data to reproduce results from the following paper:

Larock, Sakharov, Bahdra, and Eliassi-Rad. Understanding the Limitations of Network Online Learning. Forthcoming in Applied Network Science. [ArXiv preprint.](https://arxiv.org/abs/2001.07607)

#### Instructions

The code in this repository is written in Python 3 and was developed using Python 3.5. It should work, but has not been tested, on any version 3.5+. 

No installation of the software in this repository is required, and the dependencies are fairly standard, most notably `numpy`/`scipy` and `sklearn`. Scripts to reproduce the experiments from the paper are located in the `reproduce/` directory, and are named for the figures they reproduce. Scripts should automatically generate plots, but in case of issues, plotting code is also available in the `plotting/` directory. 

#### Notes 

1. The experiments in this paper are computationally expensive and this code is written in pure python and not optimized for speed. We used a large computer with many processors over hours and days to generate our results. Running these experiments, especially the parameter search, without multiprocessing is not recommended and will likely take days on a single machine.
2. This code is meant for prototyping, experimentation, and now reproducibility, not necessarily for general use. If you would like to use the NOL* method in a specific setting, we encourage you to use this implementation as a guide or reference for how our methodology works, but it is likely that implementing the method for your own use case, and especially API endpoints, will be desirable in the long run.
3. Code for the iKNN-UCB baseline can be found [at this link](https://bitbucket.org/kau_mad/net_complete/src/master/mab_explorer/). Note that for our experiments we modified the random node sampling technique in their code to match our version. We also modified the implementation to handle networks with singleton nodes (nodes with 0 degree).
4. Our experiments using node2vec node embeddings rely on [the SNAP node2vec example code](https://github.com/snap-stanford/snap/tree/master/examples). Begin with [this line](https://github.com/tlarock/nol/blob/bf671b4817edd8d4fe38751ac8da9153c73b6ad2/nol/Node2VecFeatures.py#L19) to understand how we compute and read the embeddings. 

#### Contact

If you have any questions about this repository or run in to issues reproducing our work or using this code, please either open an issue or [contact Tim LaRock](mailto:timothylarock@gmail.com).

#### License

We are providing this code under the [MIT License](https://opensource.org/licenses/MIT). See [LICENSE](https://github.com/tlarock/nol/blob/master/LICENSE) for details.
