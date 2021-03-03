# Twopoppy Grid Management

These scripts help run and analyze large grids of twopoppy runs.

- `model.py`: defines the model setup and how to run the model
- `grid_SLR.py`: an example setup of a grid: parameter combinations and other settings
- `grid_dustlines.py`: basically the same with different name and less parameter values
- `analysis_SLR.py`: calculate the effective radius and mm luminosity as in Zormpas et al. 2021.
- `analysis_mass.py`: calculate the disk mass using the Powell method.

- `histograms.ipynb`: notebook to read, filter, and analyze/plot those values in terms of how well they reprodue the correlation from [Tripathi et al. 2017](https://doi.org/10.3847/1538-4357/aa7c62).

- `dust-line-mass-calibration.ipynb`: not yet done -- will test the mass determination from dust lines using the simulated observables of the grid.


## Running a grid

This means running a grid of many simulations and storing the results in HDF5. This is now part of dipsy and available from the command line as `run_grid`. Pass a grid setup like `grid_dustlines.py`, for example

        run_grid.py -c 1 -t 0 grid_SLR.py

    to run just the first model of the grid for testing purposes. This will create `test.hdf5` containing the model outputs.

## Analyzing a grid

This is now also part of `dipsy` and available as shell command `analyze_grid`. It needs an analysis setup like `analisis_SLR.py` to tell it what it should do for each simulation. As an example, this could be done with the following command:

        analyze_grid analysis_SLR.py SLR.hdf5  -c 1 -t 1 -l 0.087

    this will loop through all simulation results and calculate the disk radii and fluxes. The result will be stored in a file `SLR_analysis_lam870_q3.5_f68.hdf5`, named with the wavelength, size distribution slope, and flux-fraction specified in the file name.


## Notes

To loop over several wavelength, use bash:

```bash
for i in 0.087 0.1 0.13 0.8; do
	analyze_grid analysis_SLR.py -l $i -c 70 --flux-fraction 0.68 SLR.hdf5 
done	
```
