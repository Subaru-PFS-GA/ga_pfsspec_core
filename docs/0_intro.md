# Introduction

The `pfsspec-core` package provides basic functionality for simulating and processing spectra as well as various generic IO, grid interpolation, etc.

The main contents of the packages is as follows:

* Spectrum datasets to store a large number of spectra, observed or simulated, in HDF5 efficiently
* Grid objects to store and interpolate high dimensional model grids efficiently in HDF5 format. Interpolation methods include multivariate linear, nearest neighbor, 1D cubic splines and RBF. The library also supports data compression using PCA.
* Components to simulate spectra or process observations such as PSF convolution, resampling, S/N calculation, spectrum stacking etc.
* Plotting functions.
* Random sampling from various distributions with limits.
* Plugin-based command-line scripts to import, export and convert data, etc.
* Various utility functions for
    * caching
    * argument parsing
    * 1D and ND interpolation
    * memory mapped data access to HDF5 datasets
    * array filtering (hole filling, smoothing)
    * PCA-based convolution with varying kernels
    * logging
    * Jupyter notebook manipulation and execution
    * Parallel processing
    * Profiling
    * Astronomy and physics specific functions
    * tracing and debugging

