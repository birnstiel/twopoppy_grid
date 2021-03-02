# Twopoppy Grid Management

These scripts help run and analyze large grids of twopoppy runs.

- `model.py`: defines the model setup and how to run the model
- `grid_dustlines.py`: an example setup of a grid: parameter combinations and other settings
- `run_grid.py`: this is now part of dipsy and available from the command line as `run_grid`. Pass a grid setup like `grid_dustlines.py`, for example

        run_grid.py -c 1 -t 0 grid_test.py

    to run just the first model of the grid for testing purposes. This will create `test.hdf5` containing all the model outputs.

- `analyze_grid.py`: this is now also part of `dipsy` and available as shell command `analyze_grid`. It needs an analysis setup like `analisis_SLR.py` to do the work. run this example with the following command:

        analyze_grid analysis_SLR.py dustlines.hdf5  -c 1 -t 1 -l 0.087

    this will loop through all simulation results and calculate the disk radii and fluxes. The result will be stored in a file `dustlines_analysis_lam870_q3.5_f68.hdf5`, named with the wavelength, size distribution slope, and flux-fraction specified in the file name.

- `histograms.ipynb`: notebook to read, filter, and analyze/plot those values in terms of how well they reprodue the correlation from [Tripathi et al. 2017](https://doi.org/10.3847/1538-4357/aa7c62).

- `dust-line-mass-calibration.ipynb`: not yet done -- will test the mass determination from dust lines using the simulated observables of the grid.
    
