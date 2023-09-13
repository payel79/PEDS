# PEDS

<!---[![DOI](https://zenodo.org/badge/147731955.svg)](https://zenodo.org/badge/latestdoi/147731955)/--->

`PEDS` is a Julia package containing methodologies for Scientific Machine Learning surrogate models called Physics-Enhanced Deep Surrogates (PEDS) that is capable of saving at least two orders of magnitude in data needed to achieve a sufficient accuracy for design.

- [PEDS](#peds)
- [Overview](#overview)
- [Content](#content)
- [System Requirements](#system-requirements)
  - [Hardware requirements](#hardware-requirements)
  - [Software requirements](#software-requirements)
    - [OS Requirements](#os-requirements)
    - [Julia Dependencies](#julia-dependencies)
- [Installation Guide:](#installation-guide)
    - [Install from Github](#install-from-github)
- [Setting up the development environment:](#setting-up-the-development-environment)
- [Instruction for use](#instruction-for-use)
- [License](#license)

# Overview
``PEDS`` showcases examples of surrogates for the reaction-diffusion equations, the diffusion equation and Maxwell's equations. For clarity and simplicity, three notebooks illustrate the five surrogates from the manuscript, that can run on a laptop. However, the result from the manuscript were obtained by training on a High-Performance Computing (HPC) cluster with 320 CPUs, using the same code. 
The package can be run on all major platforms (e.g. BSD, GNU/Linux, OS X, Windows).
Example scripts to run the code on HPC are given in the notebook.

# Content

The folder `data/` contains datasets to reproduce the findings for the five surrogate models with about 1000 data points (with 10000 data points to generate training, validation, and test sets). The name convention is `X_#MODELNAME#_small.csv` and `y_#MODELNAME#_small.csv`. 

Traditional to Julia packages, the source code for PEDS is in a folder called `src/`, the most important julia file is `PEDS.jl` which calls all the other files. 

There are three illustrative notebooks in `demos/` folder:
1. `Example_reaction-diffusion_16.ipynb` contains the code for the surrogate of the reaction-diffusion equation with 16 pores. The same code can be use for a diffusion equation with 16 pores by replacing the training dataset, because they use the same approximate solver.
2. `Example_diffusion_25.ipynb` contains the code for the surrogate of the diffusion equation with 25 pores. The same code can be use for a reaction-diffusion equation with 25 pores by replacing the training dataset, because they use the same approximate solver.
3. `Example_Maxwell.ipynb` contains the code to train and evaluate the surrogate model for Maxwell's equation. 

For all the notebook, the code from the notebook can be copied into a julia `train_PEDS.jl` and run on a HPC cluster using the command

```
mpiexec -n 320 julia train_PEDS.jl
```

# System Requirements
## Hardware requirements
`PEDS` package can train and run on a standard computer with enough RAM to support the in-memory operations. However, for approximate solvers that are more time-intensive, the present code would take advantage of parallelization on multiple CPUs with High Performance Computing. 

## Software requirements
### OS Requirements
This package is supported for *macOS* and *Linux*. The package has been tested on the following systems:
+ macOS: Mojave (10.14.1)
+ Linux: Ubuntu 16.04

### Julia Dependencies
`PEDS` mainly depends on the Julia language (version 1.6 or later).

The packages, that were used to run the code, and their commit hashes are:
```
[deps]
BSON = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
ChangePrecision = "3cb15238-376d-56a3-8042-d33272777c9a"
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
```

# Installation Guide:

### Install from Github

```
git clone git@github.mit.edu:rpestour/PEDS.git
cd PEDS
```

# Setting up the development environment:
- Install Julia language from [julialang.org](https://julialang.org/downloads/). The installation may take a few minutes. Then install the dependencies using `] add #PACKAGENAME#`


- To run demo notebooks:
  - `cd demos`
  - `jupyter notebook --ip 0.0.0.0 --no-browser --allow-root`
  - Then copy the url it generates, it looks something like this: `http://(0de284ecf0cd or 127.0.0.1):8888/?token=e5a2541812d85e20026b1d04983dc8380055f2d16c28a6ad`
  - Edit this: `(0de284ecf0cd or 127.0.0.1)` to: `127.0.0.1`, in the above link and open it in your browser
  - Then open `Example_diffusion_25.ipynb`, `Example_Maxwell.ipynb`, or `Example_reaction-diffusion_16.ipynb`

# Instruction for use

By design of PEDS, the model depends on the engineering problem because the relevant approximate solver layer needs to be encoded in the deep surrogate. So the present code would not work with other data in a straighforward way.

# License

MIT License
