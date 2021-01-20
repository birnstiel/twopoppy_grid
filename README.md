# Twopoppy Grid Management

These scripts help run and analyze large grids of twopoppy runs.

- `model.py`: defines the model setup and how to run the model
- `grid_test.py`: an example setup of a grid: parameter combinations and other settings
- `run.py`: run this and pass a grid setup like `grid_test.py`, for example

        ./run.py -c 1 -t 0 grid_test.py

    to run just the first model of the grid for testing purposes. This will create `test.hdf5` containing all the model outputs.

- `analyze.py`: run this for example with this command:

        ./analyze.py -l 0.087 -c 1 -t 0 test.hdf5 

    this will loop through all simulation results and calculate the disk radii and fluxes. The result will be stored in a file `test_analysis_lam870_q3.5_f68.hdf5`, named with the wavelength, size distribution slope, and flux-fraction specified in the file name.

- `histograms.ipynb`: notebook to read, filter, and analyze/plot those values in terms of how well they reprodue the correlation from [Tripathi et al. 2017](https://doi.org/10.3847/1538-4357/aa7c62).

- `dust-line-mass-calibration.ipynb`: not yet done -- will test the mass determination from dust lines using the simulated observables of the grid.
    
