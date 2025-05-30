# Diversity in a fire-plant feedback model

This repository provides the code to model plant-fire relationship in the Mediterranean and Boreal communities, as in the submitted article *"Functional and compositional diversity display a maximum at intermediate levels of fire frequency in a minimal plant-fire feedback model"*.

## Summary
This program models the dynamics of communities in fire prone environments. It includes 3 vegetation types that have different flammability (L), fire response (R), growth rate (c) and mortality rate (m). The model represents the time series of fractional vegetation cover (b) of the 3 plant types. The deterministic succession of plant (Tilman - Ecology, 1994) is perturbed by stochastic fires. Fire is represented as a non-stationary Poisson process, with average fire return time, 'tf'. The average fire return time depends on plant cover and flammability, leading to a fire-vegetation feedback.

## Content

The code is provided in Fortran and in Python language.

### Fortran Code

```simulation_Fortran/``` folder contains:

### Python Code

```simulation_Python/``` folder contains:

- ```biofire_main.py```: main file for generating the plant communities and simulating the time series of vegetation cover and fires.
- ```biofire_tools.py```: file containing the functions and subroutines used in the main.
- ```biofire_setting.json```: json file to set the simulation settings and the parameter values.

General command line:
```python generation.py -settings_file "biofire_setting.json"```

### Analyses

```analyses/``` folder contains:

- `build_dataframes.py`: Script to build dataframes from the simulation outputs.
- `diff_tilman.py`: Script to analyze the differences in vegetation dynamics compared to Tilman's model.
- `hypervolumes.py`: Script to calculate hypervolumes for different vegetation types.
- `ANALYSES_and_FIGURES.ipynb`: Jupyter notebook to generate the figured of the pubblished paper

## Access information

Data was derived from the following sources:

* Magnani, M., & Baudena, M. (2023). Fire Communities (FireCom) (v1.0). Zenodo. [https://doi.org/10.5281/zenodo.7763275](https://doi.org/10.5281/zenodo.7763275) (Open License)

## References and Contacts
This code has been developed within the manuscript: *"Functional and compositional diversity display a maximum at intermediate levels of fire frequency when modeling the plant-fire feedback"*

Please refer to the paper for the underlying assumptions and equations.

Authors: Matilde Torrassa and Gabriele Vissio

For more information please contact
- Matilde Torrassa (matilde.torrassa@cimafoundation.org) concerning the Python code and the analyses;
- Gabriele Vissio (gabriele.vissio@cnr.it) concerning the Fortran code.
