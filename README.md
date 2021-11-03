# Differentiable Trajectory Reweighting

Reference implementation of the Differentiable Trajectory Reweighting (DiffTRe) 
method as implemented in the paper "Learning neural network potentials from 
experimental data via Differentiable Trajectory Reweighting".

<p align="center">
<img src="https://drive.google.com/uc?export=view&id=1cWD0uKHZ4TnqfrC0DH5feFf7d3gPhsrI">
</p>


## Getting started
The Jupyter notebooks provide a guided tour through the DiffTRe method 
([double_well.ipynb](double_well.ipynb)) as well as its application to 
realistic systems in [diamond.ipynb](diamond.ipynb) and [CG_Water.ipynb](CG_Water.ipynb).

## Installation
All dependencies can be installed locally with pip:
```
pip install .
```

However, this only installs a CPU version of Jax. If you want to enable GPU 
support, please overwrite the jaxlib version:
```
pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

## Requirements
The repository uses the following packages:
```
    MDAnalysis
    jax>=0.2.12
    jaxlib>=0.1.65
    jax-md==0.1.13
    optax
    dm-haiku>=0.0.4
    sympy>=1.5
```
The code was tested with Python 3.6. The exact versions used in the paper 
are listed in ([requirements.txt](requirements.txt)). MDAnalysis is only used for loading of 
initial MD states and can be omitted in case of installation issues.

## Contact

For questions, please contact stephan.thaler@tum.de.

